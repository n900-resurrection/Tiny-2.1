"""Train TINY version 2.

  python3 -I train2.py bench                       speed test of model sizes
  python3 -I train2.py train MINUTES OUT.pt [--from TINY.BIN]
  python3 -I train2.py eval  MODEL.pt|TINY.BIN     score on held-back questions
  python3 -I train2.py export MODEL.pt TINY.BIN    write the DOS weight file

The model reads and writes one letter at a time (ASCII 32..126 plus newline)."""
import sys, os, time, math, random, struct, re, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen2

VOCAB, CTX = 97, 128
torch.set_num_threads(2)


def encode(s):
    return [1 if c == '\n' else ord(c) - 30 for c in s]


def decode(ids):
    return ''.join('\n' if i == 1 else chr(i + 30) for i in ids if i > 0)


class Cfg:
    def __init__(self, D=96, L=4, H=4, HID=256):
        self.D, self.L, self.H, self.HID = D, L, H, HID

    def __repr__(self):
        return 'D=%d L=%d H=%d HID=%d' % (self.D, self.L, self.H, self.HID)


class Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype) * self.g


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.n1, self.n2 = Norm(c.D), Norm(c.D)
        self.q = nn.Linear(c.D, c.D, bias=False)
        self.k = nn.Linear(c.D, c.D, bias=False)
        self.v = nn.Linear(c.D, c.D, bias=False)
        self.o = nn.Linear(c.D, c.D, bias=False)
        self.f1 = nn.Linear(c.D, c.HID, bias=False)
        self.f2 = nn.Linear(c.HID, c.D, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H = self.c.H
        h = self.n1(x)
        sp = lambda t: t.view(B, T, H, D // H).transpose(1, 2)
        a = F.scaled_dot_product_attention(sp(self.q(h)), sp(self.k(h)), sp(self.v(h)), is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.f2(F.relu(self.f1(self.n2(x))))


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(VOCAB, c.D)
        self.pos = nn.Embedding(CTX, c.D)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.L)])
        self.nf = Norm(c.D)
        nn.init.normal_(self.emb.weight, std=0.05)
        nn.init.normal_(self.pos.weight, std=0.05)
        for b in self.blocks:
            for m in (b.q, b.k, b.v, b.o, b.f1, b.f2):
                nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features))

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.nf(x) @ self.emb.weight.t()


# ---------------------------------------------------------------- old TINY.BIN
def load_old_bin(path, D=96, L=3, HID=192):
    """Read the version-1 weight file (one scale per table) into a model."""
    b = open(path, 'rb').read()
    off = [0]

    def table(r, c):
        s = struct.unpack_from('<f', b, off[0])[0]
        w = np.frombuffer(b, np.int8, r * c, off[0] + 4).reshape(r, c).astype(np.float32) * s
        off[0] += 4 + r * c
        return torch.tensor(w)

    def gains():
        g = np.frombuffer(b, '<f4', D, off[0]).copy()
        off[0] += 4 * D
        return torch.tensor(g)

    m = Model(Cfg(D, L, 4, HID))
    with torch.no_grad():
        m.emb.weight.copy_(table(VOCAB, D))
        m.pos.weight.copy_(table(CTX, D))
        for blk in m.blocks:
            blk.n1.g.copy_(gains())
            for lin in (blk.q, blk.k, blk.v, blk.o):
                lin.weight.copy_(table(D, D))
            blk.n2.g.copy_(gains())
            blk.f1.weight.copy_(table(HID, D))
            blk.f2.weight.copy_(table(D, HID))
        m.nf.g.copy_(gains())
    assert off[0] == len(b), 'size mismatch'
    return m


def grow(old, c):
    """Bigger model that computes exactly the same function as `old` at the start:
    extra middle-layer units get zero output weights, extra layers have zero
    output projections, so they start as 'do nothing' and learn from there."""
    m = Model(c)
    with torch.no_grad():
        m.emb.weight.copy_(old.emb.weight)
        m.pos.weight.copy_(old.pos.weight)
        m.nf.g.copy_(old.nf.g)
        for i, blk in enumerate(m.blocks):
            if i < len(old.blocks):
                o = old.blocks[i]
                for n in ('n1', 'n2'):
                    getattr(blk, n).g.copy_(getattr(o, n).g)
                for n in ('q', 'k', 'v', 'o'):
                    getattr(blk, n).weight.copy_(getattr(o, n).weight)
                h = o.f1.weight.shape[0]
                blk.f1.weight[:h] = o.f1.weight
                blk.f2.weight.zero_()
                blk.f2.weight[:, :h] = o.f2.weight
            else:
                blk.o.weight.zero_()
                blk.f2.weight.zero_()
    return m


# ---------------------------------------------------------------- data
_POOL = []


def batch(rng, n):
    """Examples of similar length go together, so little time is spent on padding."""
    if not _POOL:
        ex = []
        for _ in range(n * 32):
            q, a, _ = gen2.sample(rng)
            ids = encode(q + '\n' + a + '\n')[:CTX]
            ex.append((ids, [0] * (len(q) + 1) + [1] * (len(ids) - len(q) - 1)))
        ex.sort(key=lambda e: len(e[0]))
        chunks = [ex[i:i + n] for i in range(0, len(ex), n)]
        rng.shuffle(chunks)
        _POOL.extend(chunks)
    seqs, masks = zip(*_POOL.pop())
    T = max(len(s) for s in seqs)
    x = torch.zeros(n, T, dtype=torch.long)
    m = torch.zeros(n, T)
    for i, (s, k) in enumerate(zip(seqs, masks)):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(k)] = torch.tensor(k, dtype=torch.float)
    return x, m


def loss_fn(model, x, m):
    logits = model(x[:, :-1])
    l = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1), reduction='none')
    w = m[:, 1:].reshape(-1)
    return (l * w).sum() / w.sum()


# ---------------------------------------------------------------- evaluation
STOP = set('a an the it is are was of in on and to its it\'s they there people speak use money capital city '
           'called comes after before has have says say lives eats one two answer number letter letters starts '
           'ends with month day the'.split())


def words_of(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def eval_set(n_per_kind=60, seed=123):
    rng = random.Random(seed)
    out = []
    for f in [f for f in gen2.TXT if f['test_q']]:
        out.append((f['test_q'][0], f['answers'], 'facts'))
    rng.shuffle(out)
    out = out[:120]
    for name, _, _ in gen2.GENS:
        if name == 'facts':
            continue
        seen = set()
        while sum(1 for o in out if o[2] == name) < n_per_kind:
            q, a, _ = gen2.sample(rng, True, name)
            if q.lower() not in seen:
                seen.add(q.lower())
                out.append((q, [a], name))
    return out


UNKNOWN_START = ('i do not', 'i have never', 'sorry', 'hmm')


def correct(q, gen, refs, kind=None):
    if kind == 'unknown' and refs[0].lower().startswith(UNKNOWN_START):
        if not gen.lower().startswith(UNKNOWN_START):
            return False
        names = [w for w in words_of(refs[0]) if w in words_of(q)]
        return all(w in words_of(gen) for w in names)
    g = set(words_of(gen))
    qw = set(words_of(q))
    for r in refs:
        rw = words_of(r)
        if rw and rw[0] in ('yes', 'no') and (not words_of(gen) or words_of(gen)[0] != rw[0]):
            continue
        key = [w for w in rw if w not in STOP and w not in qw]
        if not key:
            key = [w for w in rw if w not in STOP] or rw
        if all(w in g for w in key):
            return True
    return False


@torch.no_grad()
def answer_many(model, qs):
    """Greedy answers, batched by question length."""
    model.eval()
    res = {}
    bylen = collections.defaultdict(list)
    for q in qs:
        bylen[len(q)].append(q)
    for n, group in bylen.items():
        x = torch.tensor([encode(q + '\n') for q in group])
        done = [False] * len(group)
        outs = [[] for _ in group]
        while x.shape[1] < CTX and not all(done):
            nxt = model(x)[:, -1].argmax(-1)
            for i, t in enumerate(nxt.tolist()):
                if not done[i]:
                    if t <= 1:
                        done[i] = True
                    else:
                        outs[i].append(t)
            x = torch.cat([x, nxt[:, None]], 1)
        for q, o in zip(group, outs):
            res[q] = decode(o)
    model.train()
    return res


def evaluate(model, show=0):
    es = eval_set()
    ans = answer_many(model, [e[0] for e in es])
    by = collections.defaultdict(lambda: [0, 0])
    bad = []
    for q, refs, kind in es:
        ok = correct(q, ans[q], refs, kind)
        by[kind][0] += ok
        by[kind][1] += 1
        if not ok:
            bad.append((kind, q, ans[q], refs[0]))
    tot = sum(v[0] for v in by.values()) / sum(v[1] for v in by.values())
    line = '  '.join('%s %d%%' % (k, 100 * v[0] / v[1]) for k, v in by.items())
    if show:
        random.Random(0).shuffle(bad)
        for b in bad[:show]:
            print('   x [%s] %s -> %s   (want: %s)' % b)
    return tot, line


# ---------------------------------------------------------------- export
def export(model, path):
    """Version-2 weight file. Header: 'TNY2', D, L, H, HID (int16 each).
    Each table: rows x (float32 scale + cols int8)  (one scale per row).
    Gains: D float32."""
    c = model.c
    sd = {k: v.float() for k, v in model.state_dict().items()}
    with open(path, 'wb') as f:
        f.write(b'TNY2' + struct.pack('<4h', c.D, c.L, c.H, c.HID))

        def table(w):
            w = w.numpy()
            for row in w:
                s = max(float(np.abs(row).max()) / 127.0, 1e-8)
                f.write(struct.pack('<f', s))
                f.write(np.clip(np.round(row / s), -127, 127).astype(np.int8).tobytes())

        def gains(g):
            f.write(g.numpy().astype('<f4').tobytes())

        table(sd['emb.weight'])
        table(sd['pos.weight'])
        for i in range(c.L):
            p = 'blocks.%d.' % i
            gains(sd[p + 'n1.g'])
            for n in 'qkvo':
                table(sd[p + n + '.weight'])
            gains(sd[p + 'n2.g'])
            table(sd[p + 'f1.weight'])
            table(sd[p + 'f2.weight'])
        gains(sd['nf.g'])
    return os.path.getsize(path)


def quantized_copy(model):
    """The same model with weights rounded exactly as export() does."""
    m = Model(model.c)
    m.load_state_dict(model.state_dict())
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2:
                s = (p.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-8)
                p.copy_(torch.clamp(torch.round(p / s), -127, 127) * s)
    return m


def load_model(path):
    if path.upper().endswith('.BIN'):
        return load_old_bin(path)
    ck = torch.load(path, map_location='cpu')
    m = Model(Cfg(**ck['cfg']))
    m.load_state_dict(ck['sd'])
    return m


def save_model(m, path):
    torch.save(dict(cfg=vars(m.c), sd=m.state_dict()), path)


# ---------------------------------------------------------------- main
def train(minutes, out, start=None, cfg=Cfg(), peak=2e-3, bs=64, log_every=200, resume=None,
          decay_frac=None, eval_every=0):
    """decay_frac=None: cosine schedule. decay_frac=f: constant LR, then a linear
    decay to zero over the last fraction f of the time (lets us stop and polish)."""
    torch.manual_seed(0)
    rng = random.Random(int(time.time()))
    if resume:
        model = load_model(resume)
    else:
        model = grow(load_old_bin(start), cfg) if start else Model(cfg)
    cfg = model.c
    print('model', cfg, 'parameters', sum(p.numel() for p in model.parameters()), flush=True)
    decay = [p for p in model.parameters() if p.dim() == 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.01), dict(params=other, weight_decay=0.0)],
                            lr=peak, betas=(0.9, 0.98))
    t0, step, total = time.time(), 0, minutes * 60
    run = None
    eval_no = 0
    while True:
        el = time.time() - t0
        if el > total:
            break
        warm = min(1.0, (step + 1) / 200)
        if decay_frac is None:
            lr = peak * warm * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * el / total)))
        else:
            left = (total - el) / (decay_frac * total)
            lr = peak * warm * min(1.0, max(0.0, left))
        if eval_every and el > (eval_no + 1) * eval_every * 60:
            eval_no += 1
            save_model(model, out)
            tot, line = evaluate(quantized_copy(model))
            print('  EVAL at %.0f min: %.1f%%  %s' % (el / 60, 100 * tot, line), flush=True)
        for g in opt.param_groups:
            g['lr'] = lr
        x, m = batch(rng, bs)
        loss = loss_fn(model, x, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % log_every == 0:
            print('step %5d  %4.0fs  lr %.5f  loss %.3f' % (step, el, lr, run), flush=True)
    save_model(model, out)
    print('saved', out, 'steps', step, 'time %.0fs' % (time.time() - t0), flush=True)
    return model


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'bench':
        rng = random.Random(0)
        for c in [Cfg(96, 3, 4, 192), Cfg(96, 4, 4, 256), Cfg(112, 4, 4, 224)]:
            m = Model(c)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            for i in range(13):
                if i == 3:
                    t = time.time()
                x, k = batch(rng, 96)
                l = loss_fn(m, x, k); opt.zero_grad(); l.backward(); opt.step()
            print(c, sum(p.numel() for p in m.parameters()), '%.3f s/step' % ((time.time() - t) / 10), flush=True)
    elif cmd == 'train':
        arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
        m = train(float(sys.argv[2]), sys.argv[3], arg('--from'), resume=arg('--resume'),
                  peak=float(arg('--lr', 2e-3)),
                  decay_frac=float(arg('--decay')) if arg('--decay') else None,
                  eval_every=float(arg('--eval-every', 0)))
        tot, line = evaluate(m, show=0)
        print('eval (float) %.1f%%  %s' % (100 * tot, line))
        tot, line = evaluate(quantized_copy(m), show=25)
        print('eval (8-bit) %.1f%%  %s' % (100 * tot, line))
    elif cmd == 'eval':
        m = load_model(sys.argv[2])
        tot, line = evaluate(m, show=int(sys.argv[3]) if len(sys.argv) > 3 else 20)
        print('eval %.1f%%  %s' % (100 * tot, line))
    elif cmd == 'export':
        print('wrote', export(load_model(sys.argv[2]), sys.argv[3]), 'bytes')
