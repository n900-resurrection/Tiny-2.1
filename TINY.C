/* TINY.C - a small question answering neural network for DOS.  Version 2.1
   Written for Turbo C 2.0.  Compile with the LARGE memory model
   (Options, Compiler, Model, Large) and keep TINY.BIN next to the EXE.

   In DOS it shows a full-screen interface with PC-speaker sound and a
   robot voice.  With input redirected (TINY < QUESTIONS.TXT) or with /P
   it uses plain text.  Plain ANSI C apart from the DOS screen and sound
   parts, so it also builds with a modern compiler (plain mode) for tests.

   The network reads and writes one letter at a time.  Its size is read
   from the header of TINY.BIN, so a bigger or smaller network only needs
   a new TINY.BIN (up to MAXD wide, MAXHID middle, MAXL layers). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* The full-screen interface: Turbo C conio in DOS, or conio_ncurses.h (compile with -DCONIO_NCURSES) on a POSIX system. */
#if defined(__TURBOC__) || defined(CONIO_NCURSES)
#define HAVE_UI 1
#endif

#define VOCAB  97      /* 0 pad, 1 newline, 2.. the letters ' '..'~' */
#define CTX    128     /* longest question + answer, in letters      */
#define MAXD   128
#define MAXHID 512
#define MAXL   8

/* A table of weights: one byte per number, one scale per row. */
typedef struct {
    float *scale;
    signed char *w;
    int rows, cols;
} Table;

static int D, L, H, HD, HID;
static float att_div;                         /* sqrt(HD) */
static Table emb, pos;
static Table wq[MAXL], wk[MAXL], wv[MAXL], wo[MAXL], f1[MAXL], f2[MAXL];
static float *g1[MAXL], *g2[MAXL], *gf;

static void (*on_fail)(void);                 /* restores the screen  */
static void (*on_load)(int done, int total);  /* loading progress     */
static int tables_done, tables_total;

static void fail(const char *msg)
{
    if (on_fail)
        on_fail();
    printf("\nTINY: %s\n", msg);
    exit(1);
}

static void *get_memory(unsigned n)
{
    void *p = malloc(n);
    if (p == NULL)
        fail("not enough memory. TINY needs about 540 KB free (type MEM).\n"
             "Run TINY.EXE from the DOS prompt, not inside Turbo C.");
    return p;
}

static void damaged(void)
{
    fail("TINY.BIN is too short or damaged.");
}

static void load_table(FILE *fp, Table *t, int rows, int cols)
{
    int r;
    t->rows = rows;
    t->cols = cols;
    if ((long)rows * cols > 65000L)
        fail("a table in TINY.BIN is too big for DOS.");
    t->scale = (float *)get_memory((unsigned)rows * sizeof(float));
    t->w = (signed char *)get_memory((unsigned)rows * (unsigned)cols);
    for (r = 0; r < rows; r++)
        if (fread(&t->scale[r], 4, 1, fp) != 1 ||
            fread(t->w + (unsigned)r * cols, 1, cols, fp) != (unsigned)cols)
            damaged();
    tables_done++;
    if (on_load)
        on_load(tables_done, tables_total);
}

static float *load_gains(FILE *fp)
{
    float *g = (float *)get_memory((unsigned)D * sizeof(float));
    if (fread(g, 4, D, fp) != (unsigned)D)
        damaged();
    return g;
}

static int read16(FILE *fp)
{
    int lo = fgetc(fp), hi = fgetc(fp);
    if (hi == EOF)
        damaged();
    return lo | (hi << 8);
}

static void load_model(const char *name)
{
    FILE *fp;
    char magic[4];
    int l;

    if (sizeof(char *) != 4 && sizeof(int) == 2)
        fail("please compile with the Large memory model.");
    if (sizeof(float) != 4)
        fail("this compiler has an unusual float size.");
    fp = fopen(name, "rb");
    if (fp == NULL)
        fail("cannot open TINY.BIN. Keep it in the same directory.");
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "TNY2", 4) != 0)
        fail("TINY.BIN is not a version 2 network file.");
    D = read16(fp); L = read16(fp); H = read16(fp); HID = read16(fp);
    if (D < 8 || D > MAXD || L < 1 || L > MAXL || H < 1 || D % H != 0 || HID < 8 || HID > MAXHID)
        fail("TINY.BIN has a network size this program cannot handle.");
    HD = D / H;
    tables_done = 0;
    tables_total = 2 + 6 * L;
    att_div = (float)sqrt((double)HD);
    load_table(fp, &emb, VOCAB, D);
    load_table(fp, &pos, CTX, D);
    for (l = 0; l < L; l++) {
        g1[l] = load_gains(fp);
        load_table(fp, &wq[l], D, D);
        load_table(fp, &wk[l], D, D);
        load_table(fp, &wv[l], D, D);
        load_table(fp, &wo[l], D, D);
        g2[l] = load_gains(fp);
        load_table(fp, &f1[l], HID, D);
        load_table(fp, &f2[l], D, HID);
    }
    gf = load_gains(fp);
    if (fgetc(fp) != EOF)
        fail("TINY.BIN has the wrong size.");
    fclose(fp);
}

/* ---- the network ------------------------------------------------- */

/* Memory of earlier letters: for every layer and position, the "key" and
   "value" lists of D numbers, kept as bytes with one scale per list. */
static signed char *kc[MAXL], *vc[MAXL];
static float *ks[MAXL], *vs[MAXL];

/* Work buffers (static, so they do not use the small DOS stack). */
static float bx[MAXD], bh[MAXD], bq[MAXD], bt[MAXD], ba[MAXD], bm[MAXHID], bs[CTX];
static float layer_act[MAXL];               /* how busy each layer was */

static void alloc_memory(void)
{
    int l;
    for (l = 0; l < L; l++) {
        kc[l] = (signed char *)get_memory((unsigned)CTX * D);
        vc[l] = (signed char *)get_memory((unsigned)CTX * D);
        ks[l] = (float *)get_memory(CTX * sizeof(float));
        vs[l] = (float *)get_memory(CTX * sizeof(float));
    }
}

/* y = table * x */
static void matvec(const Table *t, const float *x, float *y)
{
    int r, c, cols = t->cols;
    const signed char *w = t->w;
    float s;
    for (r = 0; r < t->rows; r++) {
        s = 0.0f;
        for (c = 0; c < cols; c++)
            s += w[c] * x[c];
        y[r] = s * t->scale[r];
        w += cols;
    }
}

/* y = x scaled to a standard size, times the gains g */
static void norm(const float *x, const float *g, float *y)
{
    int i;
    float s = 0.0f;
    for (i = 0; i < D; i++)
        s += x[i] * x[i];
    s = (float)(1.0 / sqrt(s / D + 1e-5));
    for (i = 0; i < D; i++)
        y[i] = x[i] * s * g[i];
}

/* e to the power x, quietly 0 for very negative x (no underflow messages) */
static float safe_exp(float x)
{
    if (x < -60.0f)
        return 0.0f;
    return (float)exp(x);
}

/* keep a list of D numbers as bytes, returning the scale */
static float store(signed char *dst, const float *src)
{
    int i;
    float big = 0.0f, v, k;
    for (i = 0; i < D; i++) {
        v = src[i] < 0 ? -src[i] : src[i];
        if (v > big) big = v;
    }
    if (big < 1e-20f)
        big = 1e-20f;
    k = 127.0f / big;
    for (i = 0; i < D; i++) {
        v = src[i] * k;
        dst[i] = (signed char)(v < 0 ? v - 0.5f : v + 0.5f);
    }
    return big / 127.0f;
}

/* Feed letter number `tok` at position `p` (0, 1, 2, ...).
   Fills score[0..VOCAB-1]: how well each letter fits as the next one. */
static void step(int tok, int p, float *score)
{
    int i, j, l, h;
    const signed char *e = emb.w + tok * D;
    const signed char *q = pos.w + p * D;
    const signed char *kp, *vp;
    float s, big, sum, es = emb.scale[tok], ps = pos.scale[p];

    for (i = 0; i < D; i++)
        bx[i] = e[i] * es + q[i] * ps;

    for (l = 0; l < L; l++) {
        /* look back at the earlier letters */
        norm(bx, g1[l], bh);
        matvec(&wk[l], bh, bt);
        ks[l][p] = store(kc[l] + p * D, bt);
        matvec(&wv[l], bh, bt);
        vs[l][p] = store(vc[l] + p * D, bt);
        matvec(&wq[l], bh, bq);
        for (h = 0; h < H; h++) {
            big = -1e30f;
            for (j = 0; j <= p; j++) {
                kp = kc[l] + j * D + h * HD;
                s = 0.0f;
                for (i = 0; i < HD; i++)
                    s += bq[h * HD + i] * kp[i];
                s = s * ks[l][j] / att_div;
                bs[j] = s;
                if (s > big) big = s;
            }
            sum = 0.0f;
            for (j = 0; j <= p; j++) {
                bs[j] = safe_exp(bs[j] - big);
                sum += bs[j];
            }
            for (i = 0; i < HD; i++)
                ba[h * HD + i] = 0.0f;
            for (j = 0; j <= p; j++) {
                if (bs[j] == 0.0f)
                    continue;
                vp = vc[l] + j * D + h * HD;
                s = bs[j] * vs[l][j] / sum;
                for (i = 0; i < HD; i++)
                    ba[h * HD + i] += s * vp[i];
            }
        }
        matvec(&wo[l], ba, bt);
        for (i = 0; i < D; i++)
            bx[i] += bt[i];

        /* think about it */
        norm(bx, g2[l], bh);
        matvec(&f1[l], bh, bm);
        s = 0.0f;
        for (i = 0; i < HID; i++) {
            if (bm[i] < 0.0f) bm[i] = 0.0f;
            else s += bm[i];
        }
        layer_act[l] = s / HID;                /* for the activity display */
        matvec(&f2[l], bm, bt);
        for (i = 0; i < D; i++)
            bx[i] += bt[i];
    }
    norm(bx, gf, bh);
    matvec(&emb, bh, score);
}

/* ---- choosing letters -------------------------------------------- */

#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC CLK_TCK
#endif

#define MAXQ 72                 /* longest question, in letters */

static float score[VOCAB];
static int creativity = 5;      /* 0 = always the surest answer, 9 = wild */
static float last_conf;         /* how sure the network was of its letter */

static int chosen(int tok)
{
    int i;
    float sum = 0.0f;
    for (i = 0; i < VOCAB; i++)
        sum += safe_exp(score[i] - score[tok]);
    last_conf = 1.0f / sum;
    return tok;
}

/* Pick the next letter. With creativity 0 it is always the best one.
   Otherwise choose at random among the letters that fit nearly as well,
   so the same question gets differently worded answers. */
static int pick(void)
{
    int i, best = 0;
    float t, sum = 0.0f, r;

    for (i = 1; i < VOCAB; i++)
        if (score[i] > score[best]) best = i;
    if (creativity <= 0)
        return chosen(best);
    t = creativity / 5.0f;
    for (i = 0; i < VOCAB; i++) {
        bs[i] = safe_exp((score[i] - score[best]) / t);
        if (bs[i] < 0.15f) bs[i] = 0.0f;      /* never pick unlikely letters */
        sum += bs[i];
    }
    r = sum * (float)(rand() % 10000) / 10000.0f;
    for (i = 0; i < VOCAB; i++) {
        r -= bs[i];
        if (r < 0.0f && bs[i] > 0.0f) return chosen(i);
    }
    return chosen(best);
}

/* Answer question q into out[]. The two callbacks (may be NULL) are told
   about progress through the question and about every new letter.
   Returns the number of letters; *tenths gets the time taken. */
static int think(const char *q, char *out, long *tenths,
                 void (*on_step)(int i, int n), void (*on_letter)(int c))
{
    int p = 0, n = (int)strlen(q), tok, letters = 0;
    clock_t t0 = clock();

    for (; q[p]; p++) {
        if (on_step) on_step(p, n);
        step(q[p] - 30, p, score);
    }
    step(1, p++, score);                       /* end of the question */
    for (; p < CTX; p++) {
        tok = pick();
        if (tok <= 1) break;                   /* end of the answer */
        out[letters++] = (char)(tok + 30);
        if (on_letter) on_letter(tok + 30);
        step(tok, p, score);
    }
    out[letters] = 0;
    *tenths = (long)((double)(clock() - t0) * 10.0 / CLOCKS_PER_SEC);
    return letters;
}

/* Clean up a typed line: printable letters only, no trailing spaces,
   and CAPS LOCK text turned into lower case (the network knows it better). */
static int clean(const char *line, char *q)
{
    int i, n = 0, lower = 0, upper = 0;
    for (i = 0; line[i] && n < MAXQ; i++)
        if (line[i] >= 32 && line[i] <= 126)
            q[n++] = line[i];
    while (n > 0 && q[n - 1] == ' ')
        n--;
    q[n] = 0;
    for (i = 0; i < n; i++) {
        if (q[i] >= 'a' && q[i] <= 'z') lower++;
        if (q[i] >= 'A' && q[i] <= 'Z') upper++;
    }
    if (upper > 0 && lower == 0)
        for (i = 0; i < n; i++)
            if (q[i] >= 'A' && q[i] <= 'Z') q[i] += 'a' - 'A';
    return n;
}

static int is_word(const char *q, const char *w)
{
    int i;
    for (i = 0; q[i] && w[i]; i++)
        if ((q[i] | 32) != w[i]) return 0;
    return q[i] == 0 && w[i] == 0;
}

/* ---- plain text mode ---------------------------------------------- */

static void plain_letter(int c)
{
    putchar(c);
    fflush(stdout);
}

static int plain_main(void)
{
    static char line[256], q[MAXQ + 1], out[CTX + 1];
    int letters;
    long tenths;

    printf("TINY 2 - a neural network that answers questions.\n");
    printf("Loading TINY.BIN ...\n");
    load_model("TINY.BIN");
    alloc_memory();
    printf("Ready: %d layers, %d wide. Creativity %d (TINY 0 to TINY 9).\n", L, D, creativity);
    printf("Ask about countries, animals, space, the body, nature, science, food,\n");
    printf("sums, numbers, spelling, words, shapes, history, music or computers.\n");
    printf("Type QUIT to stop.\n");
    for (;;) {
        printf("\n> ");
        fflush(stdout);
        if (fgets(line, sizeof(line), stdin) == NULL)
            break;
        if (clean(line, q) == 0)
            continue;
        if (is_word(q, "quit") || is_word(q, "exit"))
            break;
        printf("  ");
        letters = think(q, out, &tenths, NULL, plain_letter);
        printf("\n   [%d letters in %ld.%ld seconds]\n", letters, tenths / 10, tenths % 10);
    }
    printf("Goodbye!\n");
    return 0;
}

#ifdef HAVE_UI
/* ==================================================================
   The DOS screen: Turbo C conio for text, the PC speaker for sound,
   and on VGA a reprogrammed neon colour palette.
   (On a POSIX system conio_ncurses.h plays the part of conio; there
   is no sound then, and the palette is set with ncurses.)
   ================================================================== */
#ifdef __TURBOC__
#include <conio.h>
#include <dos.h>
#include <io.h>
#else
#include "conio_ncurses.h"
#endif

/* The size of the screen: 80x25 in DOS; on a POSIX system the size of the terminal, so a phone with a 72x18 terminal gets a layout of its own.
   layout() computes these once the screen exists. */
#ifdef __TURBOC__
#define SCR_COLS() 80
#define SCR_ROWS() 25
#define con_gain(pc) ((void)0)           /* the PC speaker has one volume */
#else
#define SCR_COLS() con_cols()
#define SCR_ROWS() con_rows()
#endif
#define MIN_COLS 60
#define MIN_ROWS 14
#define MON_W    22             /* width of the neural monitor, frame included */

static int scr_w = 80, scr_h = 25;
static int chat_l = 3, chat_t = 3, chat_r = 56, chat_b = 20, chatw = 54;   /* chat window, inside its frame */
static int mon_x = 61;                                                      /* neural monitor panel          */
static int body_b = 21;                                                     /* last row of the two big frames */
static int in_y = 23, in_frame = 1, in_x = 5, in_w = 54;                    /* the line for the question     */
static int mon_full = 1;                                                    /* all rows of the monitor, or the compact one */
static int mr_head, mr_l0, mr_nl, mr_conf, mr_meter, mr_last, mr_heat, mr_speed, mr_speedv, mr_net, mr_netv, mr_mode, mr_modev;

/* colours (the neon palette changes how they look on VGA) */
#define C_FRAME   11            /* electric cyan    */
#define C_FRAME2  3             /* dark cyan        */
#define C_TITLE   0x1B          /* cyan on blue     */
#define C_TITLE2  0x1F          /* white on blue    */
#define C_PANEL   0x3F          /* white on dark cyan: panel names */
#define C_YOU     13            /* neon pink        */
#define C_QTEXT   14            /* amber            */
#define C_TINY    11
#define C_ANSWER  10            /* neon mint        */
#define C_GLOW    15            /* newest letter    */
#define C_SPEAK   0x2F          /* word being spoken */
#define C_DIM     8
#define C_TEXT    7
#define C_BRIGHT  15
#define C_BAR     0x30          /* status bar       */
#define C_BARKEY  0x3F
#define C_BARON   0x3E
#define C_HOT     12
#define C_ON      10

#ifdef __TURBOC__
#define KEY_NAME1 "F1"
#define KEY_NAME2 "F2"
#define KEY_NAME3 "F3"
#define KEY_NAME4 "F4"
#else
#define KEY_NAME1 "^A"
#define KEY_NAME2 "^S"
#define KEY_NAME3 "^V"
#define KEY_NAME4 "^T"
#endif

static int mono, vga, neon = 1, sound_on = 1, voice_on = 1, demo, dump, dumpall;
static int chat_x = 1, chat_y = 1;              /* cursor inside chat window */
static unsigned cursor_shape = 0x0607;
static char last_q[MAXQ + 1];
static FILE *sndlog;
static float act_max[MAXL];
static int heat[18], heat_n;
static long speed10;                            /* letters per 10 seconds */

static void layout(void)
{
    int rows, l;
    scr_w = SCR_COLS();
    scr_h = SCR_ROWS();
    in_frame = scr_h >= 24;                      /* a framed line for the question only when there is room */
    body_b = in_frame ? scr_h - 4 : scr_h - 2;
    in_y = in_frame ? scr_h - 2 : scr_h - 1;
    in_x = in_frame ? 5 : 4;
    in_w = in_frame ? scr_w - 26 : scr_w - 6;
    chat_l = 3;
    chat_t = 3;
    chat_r = scr_w - MON_W - 2;
    chat_b = body_b - 1;
    chatw = chat_r - chat_l + 1;
    mon_x = scr_w - MON_W + 3;
    rows = chat_b - chat_t + 1;
    mon_full = rows >= 18;
    l = L < 6 ? L : 6;
    if (!mon_full && l > 4) l = 4;
    mr_nl = l;
    mr_head = 3;
    mr_l0 = 4;
    if (mon_full) {
        mr_conf = 10; mr_meter = 11; mr_last = 12; mr_heat = 13; mr_speed = 14; mr_speedv = 15;
        mr_net = 16; mr_netv = 17; mr_mode = 19; mr_modev = 20;
    } else {                                     /* the layers, then the rest without the empty rows */
        mr_conf = mr_l0 + l; mr_meter = mr_conf + 1; mr_last = mr_meter + 1; mr_heat = mr_last + 1;
        mr_speed = mr_heat + 1; mr_speedv = mr_speed + 1;
        mr_net = mr_netv = mr_mode = mr_modev = 0;
    }
}

static void color(int a)
{
    if (mono) {
        int fg = a & 15, bg = (a >> 4) & 7;
        if (bg) a = 0x70;
        else a = fg == 8 ? 7 : fg >= 9 ? 15 : 7;
    }
    textattr(a);
}

#ifdef __TURBOC__
static void cursor(int on)
{
    union REGS r;
    if (!on) {
        r.h.ah = 3; r.h.bh = 0;
        int86(0x10, &r, &r);
        if (r.x.cx != 0x2000) cursor_shape = r.x.cx;
        r.x.cx = 0x2000;
    } else
        r.x.cx = cursor_shape;
    r.h.ah = 1;
    int86(0x10, &r, &r);
}

/* ---- neon palette (VGA only) ---- */

static int has_vga(void)
{
    union REGS r;
    r.x.ax = 0x1A00;
    int86(0x10, &r, &r);
    return r.h.al == 0x1A;
}

static void set_dac(int index, int red, int green, int blue)
{
    union REGS r;
    r.x.ax = 0x1010;
    r.x.bx = index;
    r.h.dh = (unsigned char)red;
    r.h.ch = (unsigned char)green;
    r.h.cl = (unsigned char)blue;
    int86(0x10, &r, &r);
}

static void neon_palette(void)
{
    /* text colour 0..15 -> VGA colour register, and the new colour (0..63) */
    static const unsigned char reg[16] = { 0, 1, 2, 3, 4, 5, 20, 7, 56, 57, 58, 59, 60, 61, 62, 63 };
    static const unsigned char rgb[16][3] = {
        { 1, 1, 7 },    { 5, 7, 22 },   { 0, 26, 18 },  { 0, 20, 30 },
        { 40, 4, 12 },  { 28, 6, 40 },  { 42, 26, 4 },  { 40, 45, 54 },
        { 13, 16, 28 }, { 18, 34, 63 }, { 18, 63, 42 }, { 10, 57, 63 },
        { 63, 22, 34 }, { 63, 18, 58 }, { 63, 54, 14 }, { 60, 62, 63 }
    };
    int i;
    union REGS r;
    for (i = 0; i < 16; i++)
        set_dac(reg[i], rgb[i][0], rgb[i][1], rgb[i][2]);
    r.x.ax = 0x1003;                            /* bright backgrounds, no blink */
    r.h.bl = 0;
    int86(0x10, &r, &r);
}
#else
static void cursor(int on)
{
    con_cursor(on);
}

/* "VGA" here means: the terminal can change its colours (xterm with 256 colours can) */
static int has_vga(void)
{
    return con_can_palette();
}

static void neon_palette(void)
{
    static const unsigned char rgb[16][3] = {
        { 1, 1, 7 },    { 5, 7, 22 },   { 0, 26, 18 },  { 0, 20, 30 },
        { 40, 4, 12 },  { 28, 6, 40 },  { 42, 26, 4 },  { 40, 45, 54 },
        { 13, 16, 28 }, { 18, 34, 63 }, { 18, 63, 42 }, { 10, 57, 63 },
        { 63, 22, 34 }, { 63, 18, 58 }, { 63, 54, 14 }, { 60, 62, 63 }
    };
    int i;
    for (i = 0; i < 16; i++)
        con_set_palette(i, rgb[i][0], rgb[i][1], rgb[i][2]);
}
#endif

/* ---- sound ---- */

static void tone(unsigned hz, unsigned ms)      /* hz 0 = silence */
{
    if (sndlog) fprintf(sndlog, "%u %u\n", hz, ms);
    if (!sound_on) return;
    if (hz) sound(hz);
    delay(ms);
    nosound();
}

static void quiet(unsigned ms)
{
    if (sndlog) fprintf(sndlog, "0 %u\n", ms);
    delay(ms);
}

static void sweep(unsigned from, unsigned to, unsigned ms)
{
    unsigned t;
    for (t = 0; t < ms; t += 3)
        tone((unsigned)(from + ((long)to - from) * t / ms), 3);
}

static void snd_boot(void)
{
    static unsigned notes[] = { 392, 523, 659, 784, 1047, 1319 };
    int i;
    for (i = 0; i < 6; i++) tone(notes[i], 55);
    tone(1568, 140);
}

static void snd_ready(void)  { tone(1319, 40); tone(1760, 70); }
static void snd_click(void)  { tone(1800, 2); }
static void snd_error(void)  { tone(140, 120); }
static void snd_bye(void)
{
    sweep(1600, 200, 360);
}

/* ---- robot voice ----
   Not real speech: each letter becomes a little sound with the rhythm of
   speech. Vowels buzz between two formant-like pitches, s/f/h hiss,
   p/t/k click, m/n hum.  The pitch slowly falls through a sentence. */

static void buzz(unsigned f1, unsigned f2, unsigned ms, unsigned shift)
{
    unsigned t;
    f1 = (unsigned)((long)f1 * shift / 100);
    f2 = (unsigned)((long)f2 * shift / 100);
    if (sndlog) fprintf(sndlog, "B%u %u %u\n", f1, f2, ms);
    if (!sound_on) { delay(ms); return; }
    for (t = 0; t < ms; t += 4) {
        sound(f1); delay(2);
        sound(f2); delay(2);
    }
    nosound();
}

static void hiss(unsigned ms)
{
    unsigned t;
    if (sndlog) fprintf(sndlog, "H %u\n", ms);
    if (!sound_on) { delay(ms); return; }
    for (t = 0; t < ms; t++) {
        sound(2500 + rand() % 4500);
        delay(1);
    }
    nosound();
}

/* How long the voice takes: 100 is the speed of the DOS version. On the N900 it is a little slower, so the words can be followed. */
#ifdef __TURBOC__
#define VOICE_STRETCH 100
#else
#define VOICE_STRETCH 118
#endif
#define VS(ms) ((unsigned)((ms) * VOICE_STRETCH / 100))

static void say_letter(int c, unsigned shift)
{
    if (c >= 'A' && c <= 'Z') c += 32;
    switch (c) {
    case 'a': buzz(730, 1090, VS(64), shift); break;
    case 'e': buzz(530, 1840, VS(56), shift); break;
    case 'i': case 'y': buzz(300, 2290, VS(52), shift); break;
    case 'o': buzz(570, 840, VS(64), shift); break;
    case 'u': buzz(320, 870, VS(56), shift); break;
    case 's': case 'z': case 'c': case 'x': hiss(VS(46)); break;
    case 'f': case 'h': case 'v': case 'j': hiss(VS(30)); break;
    case 'p': case 't': case 'k': case 'q':
        tone(3200, 2); quiet(VS(18)); break;
    case 'b': case 'd': case 'g':
        buzz(180, 260, VS(12), shift); quiet(VS(10)); break;
    case 'm': case 'n': buzz(210, 300, VS(40), shift); break;
    case 'l': case 'r': case 'w': buzz(400, 1100, VS(36), shift); break;
    case ' ': quiet(VS(45)); break;
    case ',': case '-': quiet(VS(110)); break;
    case '.': case '!': case '?': case ':': quiet(VS(200)); break;
    default:
        if (c >= '0' && c <= '9')
            buzz(420 + (c - '0') * 60, 900 + (c - '0') * 90, VS(70), shift);
        else
            quiet(VS(20));
    }
}

/* ---- drawing helpers ---- */

static void full(void)
{
    window(1, 1, scr_w, scr_h);
}

static void hline(int x, int y, int n, int ch)
{
    gotoxy(x, y);
    while (n-- > 0) putch(ch);
}

static void put_at(int x, int y, int attr, const char *s)
{
    gotoxy(x, y);
    color(attr);
    cputs(s);
}

static void box(int x1, int y1, int x2, int y2, int attr, const char *title)
{
    int y;
    color(attr);
    gotoxy(x1, y1); putch(201); hline(x1 + 1, y1, x2 - x1 - 1, 205); putch(187);
    for (y = y1 + 1; y < y2; y++) {
        gotoxy(x1, y); putch(186);
        gotoxy(x2, y); putch(186);
    }
    gotoxy(x1, y2); putch(200); hline(x1 + 1, y2, x2 - x1 - 1, 205);
    gotoxy(x2, y2); putch(188);
    if (title) {
        put_at(x1 + 2, y1, attr, "\xB5");
        put_at(x1 + 3, y1, C_PANEL, title);
        color(attr);
        putch(198);
    }
}

/* a bar of `w` cells, `v` from 0 to 1000 */
static void meter(int x, int y, int w, int v, int attr)
{
    int i, n = (int)((long)v * w * 2 / 1000);
    gotoxy(x, y);
    for (i = 0; i < w; i++) {
        if (n >= 2 * i + 2) { color(attr); putch(219); }
        else if (n == 2 * i + 1) { color(attr); putch(221); }
        else { color(C_DIM); putch(250); }
    }
}

static char status_right[16];

static void status_bar(const char *right)
{
    char buf[20];
    int room;
    full();
    color(C_BAR);
    hline(1, scr_h, scr_w - 1, ' ');            /* not the last column: writing it would scroll the screen */
    gotoxy(1, scr_h);
    cputs(" ");
    color(C_BARKEY); cputs(KEY_NAME1); color(C_BAR); cputs(" Help  ");
    color(C_BARKEY); cputs(KEY_NAME2); color(C_BAR); cputs(" Sound ");
    color(C_BARON); cputs(sound_on ? "ON " : "OFF"); color(C_BAR); cputs("  ");
    color(C_BARKEY); cputs(KEY_NAME3); color(C_BAR); cputs(" Voice ");
    color(C_BARON); cputs(voice_on ? "ON " : "OFF"); color(C_BAR); cputs("  ");
    color(C_BARKEY); cputs(KEY_NAME4); color(C_BAR); cputs(" Creativity ");
    color(C_BARON); putch('0' + creativity); color(C_BAR); cputs("  ");
    color(C_BARKEY); cputs("Esc"); color(C_BAR); cputs(" Quit ");
    if (right) strncpy(status_right, right, 15);
    status_right[15] = 0;
    room = scr_w - 1 - (wherex() - 1);           /* what is left of the bar, up to the column before the last */
    if (room > 15) room = 15;
    if (room > 0) {
        sprintf(buf, "%-*.*s", room, room, status_right);
        cputs(buf);
    }
}

static void title_bar(void)
{
    struct time t;
    char buf[12];
    full();
    color(C_TITLE);
    hline(1, 1, scr_w, ' ');
    put_at(2, 1, C_TITLE, "\x11\x10");
    put_at(5, 1, C_TITLE2, "T I N Y");
    if (scr_w >= 80)
        put_at(13, 1, C_TITLE, "\xFA  NEURAL INTERFACE 2.1  \xFA  367,000 WEIGHTS");
    else
        put_at(13, 1, C_TITLE, "\xFA  NEURAL INTERFACE 2.1");
    gettime(&t);
    sprintf(buf, "%02d:%02d", t.ti_hour, t.ti_min);
    put_at(scr_w - 18, 1, 0x1A, "\xFE");
    put_at(scr_w - 16, 1, C_TITLE, "ONLINE");
    put_at(scr_w - 7, 1, C_TITLE2, buf);
}

/* ---- the neural monitor (right panel) ---- */

static void monitor_static(void)
{
    int l;
    char buf[8];
    put_at(mon_x, mr_head, C_DIM, "LAYER ACTIVITY");
    for (l = 0; l < mr_nl; l++) {
        sprintf(buf, "L%d", l + 1);
        put_at(mon_x, mr_l0 + l, C_FRAME, buf);
        meter(mon_x + 3, mr_l0 + l, 15, 0, C_ANSWER);
    }
    put_at(mon_x, mr_conf, C_DIM, "CONFIDENCE");
    meter(mon_x, mr_meter, 13, 0, C_ANSWER);
    put_at(mon_x, mr_last, C_DIM, "LAST LETTERS");
    put_at(mon_x, mr_speed, C_DIM, "SPEED");
    if (mr_net) {
        put_at(mon_x, mr_net, C_DIM, "NETWORK");
        sprintf(buf, "%d", L);
        put_at(mon_x, mr_netv, C_TEXT, buf); cputs(" layers \xFA ");
        sprintf(buf, "%d", D); cputs(buf); cputs(" wide");
        put_at(mon_x, mr_mode, C_DIM, "MODE");
    }
}

static void monitor_live(void)
{
    int l, v, i;
    char buf[20];
    full();
    for (l = 0; l < mr_nl; l++) {
        if (layer_act[l] > act_max[l]) act_max[l] = layer_act[l];
        v = act_max[l] > 0 ? (int)(1000.0f * layer_act[l] / act_max[l]) : 0;
        meter(mon_x + 3, mr_l0 + l, 15, v, l % 2 ? C_FRAME : C_ANSWER);
        act_max[l] *= 0.995f;
    }
    v = (int)(last_conf * 1000.0f);
    meter(mon_x, mr_meter, 13, v, v > 800 ? C_ANSWER : v > 500 ? C_QTEXT : C_HOT);
    sprintf(buf, " %3d%%", v / 10);
    put_at(mon_x + 13, mr_meter, C_BRIGHT, buf);
    gotoxy(mon_x, mr_heat);
    for (i = 0; i < 18; i++) {
        if (i >= heat_n) { color(C_DIM); putch(250); continue; }
        v = heat[i];
        color(v > 800 ? C_ANSWER : v > 500 ? C_QTEXT : C_HOT);
        putch(v > 900 ? 219 : v > 700 ? 178 : v > 450 ? 177 : 176);
    }
    sprintf(buf, "%ld.%ld letters/s  ", speed10 / 10, speed10 % 10);
    put_at(mon_x, mr_speedv, C_TEXT, buf);
    if (mr_modev) {
        sprintf(buf, "creativity %d     ", creativity);
        put_at(mon_x, mr_modev, C_TEXT, buf);
    }
}

/* ---- frame ---- */

static void frame(void)
{
    int cf = scr_w - MON_W;                      /* the right edge of the chat frame */
    layout();                                    /* the number of layers is known now */
    full();
    color(C_TEXT);
    clrscr();
    title_bar();
    box(1, 2, cf, body_b, C_FRAME, " CONVERSATION ");
    box(cf + 1, 2, scr_w, body_b, C_FRAME2, " NEURAL MONITOR ");
    if (in_frame) {
        box(1, scr_h - 3, scr_w, scr_h - 1, C_FRAME2, " ASK TINY ");
        put_at(3, in_y, C_YOU, "\x10");
        put_at(cf + 1, scr_h - 3, C_FRAME2, "\xD1");
        put_at(cf + 1, scr_h - 1, C_FRAME2, "\xCF");
        put_at(cf + 1, in_y, C_FRAME2, "\xB3");
        put_at(cf + 4, in_y, C_DIM, "\x18 last  " KEY_NAME1 " help");
    } else
        put_at(2, in_y, C_YOU, "\x10");
    monitor_static();
    monitor_live();
    status_bar(NULL);
}

/* ---- chat window ---- */

static void chat_win(void)
{
    window(chat_l, chat_t, chat_r, chat_b);
    gotoxy(chat_x, chat_y);
}

static void chat_save(void)
{
    chat_x = wherex();
    chat_y = wherey();
    full();
}

static void chat_nl(void)
{
    cputs("\r\n");
}

/* make sure `lines` free lines are left below the cursor (no scrolling
   while an answer is printed, so its screen position stays known) */
static void chat_room(int lines)
{
    int h = chat_b - chat_t + 1, y = wherey(), x = wherex();
    while (y > h - lines + 1 && y > 1) {
        gotoxy(1, h);
        cputs("\n");                              /* scrolls the window up */
        y--;
    }
    gotoxy(x, y);
}

static void chat_text(int attr, const char *s, int indent)
{
    int i;
    color(attr);
    for (i = 0; s[i]; i++) {
        if (wherex() >= chatw) {
            chat_nl();
            gotoxy(indent, wherey());
        }
        putch(s[i]);
    }
}

static void dump_screen(const char *name)          /* for testing */
{
    char *cells = (char *)malloc(scr_w * scr_h * 2);
    FILE *fp;
    if (cells == NULL) return;
    gettext(1, 1, scr_w, scr_h, cells);
    fp = fopen(name, "wb");
    if (fp) { fwrite(cells, 1, scr_w * scr_h * 2, fp); fclose(fp); }
    free(cells);
}

/* ---- boot screen ---- */

static const char *logo[5] = {
    "#####  ###  #   #  #   #",
    "  #     #   ##  #   # # ",
    "  #     #   # # #    #  ",
    "  #     #   #  ##    #  ",
    "  #    ###  #   #    #  "
};
static int logo_col[5] = { 11, 11, 9, 13, 13 };
static int boot_y;
static int ox;                                   /* the boot screen is drawn for 80 columns; ox moves it to the middle of a narrower or wider one */
static int boot_load, boot_bar_y;
#define LOGO_X (16 + ox)
#define LOGO_Y 2

static int wait_or_key(unsigned ms)
{
    if (dump) return 0;
    delay(ms);
    if (kbhit()) { getch(); return 1; }
    return 0;
}

/* falling green letters, like a film computer */
static void rain(void)
{
    int col[256], spd[256], f, x, y;
    int cols = scr_w < 256 ? scr_w : 256;
    for (x = 0; x < cols; x++) { col[x] = -(rand() % scr_h); spd[x] = 1 + rand() % 2; }
    for (f = 0; f < 46; f++) {
        for (x = 0; x < cols - 1; x += 2) {
            y = col[x];
            if (y >= 1 && y <= scr_h) {
                gotoxy(x + 1, y);
                color(f < 40 ? C_ANSWER : C_DIM);
                putch(33 + rand() % 90);
            }
            if (y + 1 >= 1 && y + 1 <= scr_h) {
                gotoxy(x + 1, y + 1);
                color(C_BRIGHT);
                putch(33 + rand() % 90);
            }
            if (y - 7 >= 1 && y - 7 <= scr_h) {
                gotoxy(x + 1, y - 7);
                putch(' ');
            }
            col[x] += spd[x];
            if (col[x] > scr_h + 7) col[x] = -(rand() % 10);
        }
        if (f % 4 == 0) tone(3000 + rand() % 3000, 2);
        if (wait_or_key(28)) break;
    }
    color(C_TEXT);
    clrscr();
}

/* the logo appears out of random symbols, then a light sweeps over it */
static void logo_in(void)
{
    static unsigned char done[5][24];
    int i, k, y, x, n = 0, total = 0;
    for (y = 0; y < 5; y++)
        for (x = 0; x < 24; x++) {
            done[y][x] = 0;
            if (logo[y][x] == '#') total++;
        }
    for (k = 0; n < total && k < 400; k++) {
        for (i = 0; i < 5; i++) {               /* settle five cells per frame */
            y = rand() % 5; x = rand() % 24;
            if (logo[y][x] != '#' || done[y][x]) continue;
            done[y][x] = 1; n++;
            /* shadow, one cell down and right */
            if (!(y < 4 && x < 23 && logo[y + 1][x + 1] == '#')) {
                gotoxy(LOGO_X + 2 * x + 2, LOGO_Y + y + 1);
                color(1);
                putch(177); putch(177);
            }
            gotoxy(LOGO_X + 2 * x, LOGO_Y + y);
            color(logo_col[y]);
            putch(219); putch(219);
        }
        for (i = 0; i < 6; i++) {               /* flicker in some noise */
            y = rand() % 5; x = rand() % 24;
            if (logo[y][x] != '#' || done[y][x]) continue;
            gotoxy(LOGO_X + 2 * x, LOGO_Y + y);
            color(C_DIM);
            putch(176 + rand() % 3); putch(176 + rand() % 3);
        }
        if (k % 3 == 0) tone(200 + n * 12, 3);
        if (!dump) delay(12);
    }
    for (y = 0; y < 5; y++)                     /* finish anything left */
        for (x = 0; x < 24; x++)
            if (logo[y][x] == '#') {
                gotoxy(LOGO_X + 2 * x, LOGO_Y + y);
                color(logo_col[y]);
                putch(219); putch(219);
            }
    for (x = 0; x < 26 && !dump; x++) {         /* light sweep */
        for (y = 0; y < 5; y++) {
            if (x < 24 && logo[y][x] == '#') {
                gotoxy(LOGO_X + 2 * x, LOGO_Y + y);
                color(C_BRIGHT); putch(219); putch(219);
            }
            if (x >= 2 && logo[y][x - 2] == '#') {
                gotoxy(LOGO_X + 2 * (x - 2), LOGO_Y + y);
                color(logo_col[y]); putch(219); putch(219);
            }
        }
        delay(14);
    }
}

static void boot_bar(int done, int total)
{
    int i, n = (int)(36L * done / total);
    char buf[24];
    full();
    gotoxy(22 + ox, boot_bar_y);
    color(C_FRAME);
    putch(222);
    for (i = 0; i < 36; i++) {
        color(i < n ? (i * 3 < n ? C_FRAME : C_ANSWER) : C_DIM);
        putch(i < n ? 219 : 250);
    }
    color(C_FRAME);
    putch(221);
    sprintf(buf, " %3d%%", (int)(100L * done / total));
    color(C_BRIGHT);
    cputs(buf);
    if (done == 2 || (done > 2 && (done - 2) % 6 == 0)) {
        if (done == 2) sprintf(buf, "letters + places");
        else sprintf(buf, "layer %d of %d", (done - 2) / 6, L);
        if (boot_y <= scr_h - 2) {               /* a short screen has room for fewer lines */
            gotoxy(26 + ox, boot_y);
            color(C_YOU); cputs("\x10 ");
            color(C_TEXT); cputs(buf);
            cputs(" ");
            i = wherex();
            color(C_DIM); hline(i, boot_y, 50 + ox - i, 250);
            color(C_ANSWER); cputs(" ONLINE");
        }
        boot_y++;
        tone(880 + 110 * (boot_y % 6), 12);
    }
}

static void boot(void)
{
    int x, rule = scr_w < 62 ? scr_w - 2 : 60;
    ox = (scr_w - 80) / 2;
    boot_load = scr_h >= 20 ? 13 : 11;           /* a short screen drops the rule and some empty rows */
    boot_bar_y = boot_load + 2;
    color(C_TEXT);
    clrscr();
    cursor(0);
    rain();
    logo_in();
    put_at(24 + ox, 8, C_BRIGHT, "N E U R A L    I N T E R F A C E");
    put_at(29 + ox, 9, C_DIM, "version 2.1");
    if (scr_h >= 20) {
        color(C_FRAME2);
        hline((scr_w - rule) / 2 + 1, 11, rule, 196);
    }
    put_at(22 + ox, boot_load, C_FRAME, "LOADING NEURAL WEIGHTS FROM TINY.BIN");
    boot_y = boot_bar_y + 2;
    on_load = boot_bar;
    load_model("TINY.BIN");
    alloc_memory();
    on_load = NULL;
    snd_boot();
    for (x = 0; x < 6; x++) {
        put_at(30 + ox, boot_y + 1 < scr_h ? boot_y + 1 : scr_h - 1, x % 2 ? C_BRIGHT : C_ANSWER, "\x10 SYSTEM  READY \x11");
        if (!dump) delay(120);
    }
    if (dump) dump_screen("BOOT.DMP");
}

/* ---- help ---- */

static void show_help(void)
{
    static const char *h[] = {
        "Ask a question and press Enter.",
        "Try: capital of peru \xFA is a whale a fish",
        "     what is 7 times 8 \xFA spell elephant",
        "     which countries border estonia",
        "     how do you say thank you in spanish",
        KEY_NAME2 " sound \xFA " KEY_NAME3 " robot voice \xFA " KEY_NAME4 " creativity",
        "\x18 last question \xFA Esc quit \xFA CLS clears",
        NULL
    };
    int i;
    chat_win();
    if (wherex() > 1) chat_nl();
    chat_room(chat_b - chat_t + 1 < 9 ? chat_b - chat_t + 1 : 9);
    color(C_PANEL); cputs(" HELP "); chat_nl();
    for (i = 0; h[i]; i++) {
        color(C_TEXT); cputs(" "); cputs(h[i]); chat_nl();
    }
    chat_save();
}

/* ---- asking ---- */

static int ans_x0, ans_y0, ans_col;
static int spin, scr_prev;

/* where letter i of the answer is on the screen */
static void ans_pos(int i, int *sx, int *sy)
{
    int per = chatw - ans_x0;                    /* letters per line */
    *sx = chat_l + ans_x0 - 1 + i % per;
    *sy = chat_t + ans_y0 - 1 + i / per;
}

/* a scanner sweeps back and forth while TINY reads the question */
static void ui_step(int i, int n)
{
    int k, head, sx, sy, d;
    full();
    head = spin % 18;
    if (head >= 9) head = 17 - head;
    ans_pos(0, &sx, &sy);
    gotoxy(sx, sy);
    for (k = 0; k < 10; k++) {
        d = head - k; if (d < 0) d = -d;
        color(d == 0 ? C_BRIGHT : d == 1 ? C_YOU : d == 2 ? 5 : C_DIM);
        putch(d == 0 ? 219 : d == 1 ? 178 : d == 2 ? 177 : 250);
    }
    color(C_DIM);
    cprintf(" reading %d%%", (int)(100L * (i + 1) / n));
    spin++;
    monitor_live();
    if (i % 3 == 0) tone(2400 + (rand() % 8) * 150, 3);
}

static void ui_letter(int c)
{
    int x, y;
    full();
    if (ans_col == 0) {                          /* erase the scanner */
        ans_pos(0, &x, &y);
        gotoxy(x, y);
        color(C_TEXT);
        cputs("                       ");
    } else {                                     /* last letter stops glowing */
        ans_pos(ans_col - 1, &x, &y);
        gotoxy(x, y);
        color(C_ANSWER);
        putch(scr_prev);
    }
    ans_pos(ans_col, &x, &y);
    if (c != ' ') {                              /* decode flicker */
        gotoxy(x, y);
        color(C_FRAME);
        putch(33 + rand() % 90);
        if (!dump) delay(10);
    }
    gotoxy(x, y);
    color(C_GLOW);
    putch(c);
    scr_prev = c;
    ans_col++;
    if (heat_n < 18)
        heat[heat_n++] = (int)(last_conf * 1000.0f);
    else {
        int k;
        for (k = 0; k < 17; k++) heat[k] = heat[k + 1];
        heat[17] = (int)(last_conf * 1000.0f);
    }
    monitor_live();
    if (!voice_on) tone(900 + (c % 16) * 70, 6);
}

static void speak(const char *a)
{
    int i, j, k, sx, sy, n = (int)strlen(a);
    unsigned shift;
    full();
    for (i = 0; i < n; i = j) {
        for (j = i; j < n && a[j] != ' '; j++) ;   /* one word: i..j-1 */
        for (; j < n && a[j] == ' '; j++) ;
        color(C_SPEAK);
        for (k = i; k < j && a[k] != ' '; k++) {
            ans_pos(k, &sx, &sy);
            gotoxy(sx, sy);
            putch(a[k]);
        }
        shift = (unsigned)(115 - 30L * i / (n + 1));
        for (k = i; k < j; k++) {
            if (kbhit()) { shift = 0; break; }
            say_letter(a[k], shift);
        }
        color(C_ANSWER);
        for (k = i; k < j && a[k] != ' '; k++) {
            ans_pos(k, &sx, &sy);
            gotoxy(sx, sy);
            putch(a[k]);
        }
        if (shift == 0) break;                   /* a key stops the voice */
    }
}

static void ask(const char *q)
{
    static char out[CTX + 1];
    char info[40];
    long tenths;
    int letters, sx, sy;

    chat_win();
    if (wherex() > 1) chat_nl();
    if (wherey() > 1) chat_nl();                 /* empty line between turns */
    chat_room(6);
    color(C_YOU); cputs(" YOU \xAF ");
    chat_text(C_QTEXT, q, 8);
    chat_nl();
    color(C_TINY); cputs("TINY \xAF ");
    ans_x0 = wherex();
    ans_y0 = wherey();
    ans_col = 0;
    heat_n = 0;
    chat_save();
    cursor(0);
    status_bar("thinking...");
    letters = think(q, out, &tenths, ui_step, ui_letter);
    if (letters == 0) { ui_letter('.'); out[0] = '.'; out[1] = 0; letters = 1; }
    ans_pos(ans_col - 1, &sx, &sy);              /* last letter stops glowing */
    gotoxy(sx, sy); color(C_ANSWER); putch(scr_prev);
    speed10 = tenths > 0 ? 100L * letters / tenths : 0;
    sprintf(info, "%d ltr %ld.%lds", letters, tenths / 10, tenths % 10);
    status_bar(info);
    monitor_live();
    if (sound_on && voice_on) {
        tone(1200, 8);
        speak(out);
    } else
        snd_ready();
    chat_x = 1;
    chat_y = sy - chat_t + 2;
    chat_win();
    chat_save();
    title_bar();
    cursor(1);
}

/* ---- typing ---- */

#define K_ESC   27
#define K_ENTER 13
#define K_BS    8
#define K_F1    (256 + 59)
#define K_F2    (256 + 60)
#define K_F3    (256 + 61)
#define K_F4    (256 + 62)
#define K_UP    (256 + 72)

static int key(void)
{
    int c = getch();
    if (c == 0) c = 256 + getch();
    return c;
}

static void draw_input(const char *buf, int n)
{
    int w = in_w;                                /* visible part of the line */
    int shown = n > w ? w : n;
    full();
    gotoxy(in_x, in_y);
    color(C_BRIGHT);
    cputs(n > w ? buf + n - w : buf);
    color(C_TEXT);
    hline(in_x + shown, in_y, w + 1 - shown, ' ');
    gotoxy(in_x + shown, in_y);
}

/* read a line; returns -1 for Esc */
static int read_line(char *buf)
{
    int n = 0, c;
    buf[0] = 0;
    draw_input(buf, 0);
    for (;;) {
        c = key();
        if (c == K_ENTER) return n;
        if (c == K_ESC) return -1;
        if (c == K_BS) {
            if (n > 0) { buf[--n] = 0; snd_click(); }
        } else if (c == K_UP) {
            strcpy(buf, last_q);
            n = (int)strlen(buf);
        } else if (c == K_F1) {
            show_help();
        } else if (c == K_F2) {
            sound_on = !sound_on;
            snd_click();
        } else if (c == K_F3) {
            voice_on = !voice_on;
            snd_click();
        } else if (c == K_F4) {
            creativity = (creativity + 1) % 10;
            snd_click();
            monitor_live();
        } else if (c >= 32 && c <= 126) {
            if (n < MAXQ) { buf[n++] = (char)c; buf[n] = 0; snd_click(); }
            else snd_error();
        }
        status_bar(NULL);
        draw_input(buf, n);
    }
}

/* ---- demo mode ---- */

static const char *demo_q[] = {
    "what is the capital of estonia", "is a whale a fish", "what is 7 times 8",
    "which countries border estonia", "spell computer", "what is the symbol for gold",
    "how do you say thank you in spanish", "what is a group of lions called",
    "what comes after 199", "what time is it", "tell me a joke",
    "name the continents", NULL
};

static void demo_type(const char *q)
{
    char buf[MAXQ + 1];
    int n;
    for (n = 0; q[n]; n++) {
        buf[n] = q[n];
        buf[n + 1] = 0;
        draw_input(buf, n + 1);
        snd_click();
        if (!dump) delay(45 + rand() % 50);
    }
    if (!dump) delay(250);
    draw_input("", 0);
}

static void restore_screen(void)
{
    nosound();
    full();
    if (vga && neon) textmode(C80);             /* resets the colours */
    textattr(7);
    clrscr();
    cursor(1);
#ifndef __TURBOC__
    con_end();                                  /* back to the shell: the screen, the colours */
#endif
}

static int ui_main(void)
{
    static char line[MAXQ + 1], q[MAXQ + 1];
    struct text_info ti;
    int i, n;

    layout();
    if (scr_w < MIN_COLS || scr_h < MIN_ROWS) {
        restore_screen();
        printf("TINY: the terminal is %dx%d; the full-screen mode needs at least %dx%d.\n"
               "      Make the font smaller, or use the plain mode:  tiny /P\n", scr_w, scr_h, MIN_COLS, MIN_ROWS);
        return 1;
    }
    gettextinfo(&ti);
    if (ti.currmode == MONO || ti.currmode == BW80 || ti.currmode == BW40) mono = 1;
    if (ti.currmode != MONO && ti.currmode != C80 && ti.currmode != BW80)
        textmode(mono ? BW80 : C80);
    vga = !mono && has_vga();
    if (vga && neon) neon_palette();
    on_fail = restore_screen;
    if (dump) sndlog = fopen("SOUND.LOG", "w");
    con_gain(30);                                /* the boot rain, the logo and the melody are loud next to the voice: they play at 30 % */
    boot();
    if (!demo && !dump) {
        put_at(29 + ox, boot_y + 2 < scr_h ? boot_y + 2 : scr_h, C_DIM, "press any key to begin");
        while (!kbhit()) ;
        if (key() == K_ESC) { restore_screen(); return 0; }
    }
    sweep(300, 1400, 120);
    con_gain(100);
    frame();
    cursor(1);
    chat_win();
    color(C_DIM);
    cputs("Hello! I am TINY, a neural network in your computer.");
    chat_nl();
    cputs("Ask about countries, animals, space, words or sums.");
    chat_nl();
    chat_save();
    snd_ready();

    if (demo || dump) {
        for (i = 0; demo_q[i]; i++) {
            if (kbhit()) { key(); break; }
            demo_type(demo_q[i]);
            ask(demo_q[i]);
            if (dump && !dumpall && i == 3) break;
            if (!dump) delay(1500);
        }
        if (dump) {
            dump_screen("SCREEN.DMP");
            if (sndlog) fclose(sndlog);
            restore_screen();
            return 0;
        }
    }
    for (;;) {
        n = read_line(line);
        if (n < 0) break;
        draw_input("", 0);
        if (clean(line, q) == 0) continue;
        if (is_word(q, "quit") || is_word(q, "exit")) break;
        if (is_word(q, "cls")) {
            frame();
            chat_x = chat_y = 1;
            continue;
        }
        if (is_word(q, "help")) { show_help(); continue; }
        strcpy(last_q, q);
        ask(q);
    }
    status_bar("goodbye");
    snd_bye();
    restore_screen();
    printf("TINY: goodbye!\n");
    return 0;
}
#endif /* HAVE_UI */

int main(int argc, char **argv)
{
    int i, plain = 0;

    for (i = 1; i < argc; i++) {
        if (argv[i][0] >= '0' && argv[i][0] <= '9')
            creativity = argv[i][0] - '0';
        else if (is_word(argv[i], "/p") || is_word(argv[i], "-p"))
            plain = 1;
#ifdef HAVE_UI
        else if (is_word(argv[i], "/demo") || is_word(argv[i], "-demo"))
            demo = 1;
        else if (is_word(argv[i], "/dump"))
            dump = 1;
        else if (is_word(argv[i], "/dumpall"))
            dump = dumpall = 1;
        else if (is_word(argv[i], "/mono") || is_word(argv[i], "-mono"))
            mono = 1;
        else if (is_word(argv[i], "/quiet") || is_word(argv[i], "-quiet"))
            sound_on = 0;
        else if (is_word(argv[i], "/classic") || is_word(argv[i], "-classic"))
            neon = 0;
#endif
    }
    srand((unsigned)time(NULL));
#ifdef HAVE_UI
    if (!plain && isatty(0) && isatty(1))
        return ui_main();
#endif
    return plain_main();
}
