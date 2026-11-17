/* conio_ncurses.c - the part of Turbo C's <conio.h> that TINY.C uses, on top of ncurses (wide characters).
   Written for the Nokia N900 port (Maemo 5, terminal osso-xterm), but it is plain POSIX + ncurses.

   TINY.C keeps its 80x25 DOS screen. This header keeps a virtual screen of 80x25 cells (a CP437 character and a DOS colour attribute
   each) and a text window, as conio does, and draws the changed cells with ncurses. Sound is not played (the PC speaker is
   not on the phone): sound() and nosound() do nothing, delay() still waits, so the animation keeps its timing.

   Keys: DOS function keys come out as 0 followed by the scan code, as getch() does in DOS. A phone keyboard has no F1..F4,
   so Ctrl-A, Ctrl-S, Ctrl-V and Ctrl-T give the same codes (help, sound, voice, creativity). */

#define _GNU_SOURCE 1
#define _XOPEN_SOURCE_EXTENDED 1


#if defined(__has_include)
#if __has_include(<ncursesw/curses.h>)
#include <ncursesw/curses.h>   /* the wide-character declarations (Debian, the Maemo SDK) */
#else
#include <curses.h>
#endif
#else
#include <curses.h>
#endif
/* curses.h is included only in this file: its macros (box, hline, getch ...) would clash with the names in TINY.C */
#include <locale.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CONIO_IMPL 1
#include "conio_ncurses.h"

/* the screen is as big as the terminal (at most SCR_MAXW x SCR_MAXH cells) */
#define SCR_MAXW 200
#define SCR_MAXH 80

typedef struct { unsigned char ch, attr; } ConCell;

static int SCR_W = 80, SCR_H = 25;
static ConCell con_scr[SCR_MAXH][SCR_MAXW];           /* what the program has drawn            */
static unsigned char con_shown_ch[SCR_MAXH][SCR_MAXW];  /* what is on the real screen            */
static unsigned char con_shown_at[SCR_MAXH][SCR_MAXW];
static int con_started, con_dirty, con_color;
static int con_wl = 1, con_wt = 1, con_wr = 80, con_wb = 25;   /* the text window (1-based)  */
static int con_x = 1, con_y = 1;                                      /* cursor inside the window   */
static unsigned char con_attr = 7;
static int con_cursor_on = 1;
static unsigned char con_cmap[16];

/* CP437 -> Unicode */
static const unsigned short con_cp437[256] = {
    0x0020, 0x263A, 0x263B, 0x2665, 0x2666, 0x2663, 0x2660, 0x2022,
    0x25D8, 0x25CB, 0x25D9, 0x2642, 0x2640, 0x266A, 0x266B, 0x263C,
    0x25BA, 0x25C4, 0x2195, 0x203C, 0x00B6, 0x00A7, 0x25AC, 0x21A8,
    0x2191, 0x2193, 0x2192, 0x2190, 0x221F, 0x2194, 0x25B2, 0x25BC,
    0x0020, 0x0021, 0x0022, 0x0023, 0x0024, 0x0025, 0x0026, 0x0027,
    0x0028, 0x0029, 0x002A, 0x002B, 0x002C, 0x002D, 0x002E, 0x002F,
    0x0030, 0x0031, 0x0032, 0x0033, 0x0034, 0x0035, 0x0036, 0x0037,
    0x0038, 0x0039, 0x003A, 0x003B, 0x003C, 0x003D, 0x003E, 0x003F,
    0x0040, 0x0041, 0x0042, 0x0043, 0x0044, 0x0045, 0x0046, 0x0047,
    0x0048, 0x0049, 0x004A, 0x004B, 0x004C, 0x004D, 0x004E, 0x004F,
    0x0050, 0x0051, 0x0052, 0x0053, 0x0054, 0x0055, 0x0056, 0x0057,
    0x0058, 0x0059, 0x005A, 0x005B, 0x005C, 0x005D, 0x005E, 0x005F,
    0x0060, 0x0061, 0x0062, 0x0063, 0x0064, 0x0065, 0x0066, 0x0067,
    0x0068, 0x0069, 0x006A, 0x006B, 0x006C, 0x006D, 0x006E, 0x006F,
    0x0070, 0x0071, 0x0072, 0x0073, 0x0074, 0x0075, 0x0076, 0x0077,
    0x0078, 0x0079, 0x007A, 0x007B, 0x007C, 0x007D, 0x007E, 0x2302,
    0x00C7, 0x00FC, 0x00E9, 0x00E2, 0x00E4, 0x00E0, 0x00E5, 0x00E7,
    0x00EA, 0x00EB, 0x00E8, 0x00EF, 0x00EE, 0x00EC, 0x00C4, 0x00C5,
    0x00C9, 0x00E6, 0x00C6, 0x00F4, 0x00F6, 0x00F2, 0x00FB, 0x00F9,
    0x00FF, 0x00D6, 0x00DC, 0x00A2, 0x00A3, 0x00A5, 0x20A7, 0x0192,
    0x00E1, 0x00ED, 0x00F3, 0x00FA, 0x00F1, 0x00D1, 0x00AA, 0x00BA,
    0x00BF, 0x2310, 0x00AC, 0x00BD, 0x00BC, 0x00A1, 0x00AB, 0x00BB,
    0x2591, 0x2592, 0x2593, 0x2502, 0x2524, 0x2561, 0x2562, 0x2556,
    0x2555, 0x2563, 0x2551, 0x2557, 0x255D, 0x255C, 0x255B, 0x2510,
    0x2514, 0x2534, 0x252C, 0x251C, 0x2500, 0x253C, 0x255E, 0x255F,
    0x255A, 0x2554, 0x2569, 0x2566, 0x2560, 0x2550, 0x256C, 0x2567,
    0x2568, 0x2564, 0x2565, 0x2559, 0x2558, 0x2552, 0x2553, 0x256B,
    0x256A, 0x2518, 0x250C, 0x2588, 0x2584, 0x258C, 0x2590, 0x2580,
    0x03B1, 0x00DF, 0x0393, 0x03C0, 0x03A3, 0x03C3, 0x00B5, 0x03C4,
    0x03A6, 0x0398, 0x03A9, 0x03B4, 0x221E, 0x03C6, 0x03B5, 0x2229,
    0x2261, 0x00B1, 0x2265, 0x2264, 0x2320, 0x2321, 0x00F7, 0x2248,
    0x00B0, 0x2219, 0x00B7, 0x221A, 0x207F, 0x00B2, 0x25A0, 0x00A0,
};

void con_end(void)
{
    if (con_started) {
        endwin();
        con_started = 0;
    }
}

/* the colour of a DOS attribute: foreground 0..15, background 0..7 */
static void con_pair_init(void)
{
    int f, b;
    static const unsigned char ansi[16] = { 0, 4, 2, 6, 1, 5, 3, 7, 8, 12, 10, 14, 9, 13, 11, 15 };   /* CGA order -> ANSI order */
    for (f = 0; f < 16; f++)
        con_cmap[f] = (COLORS >= 16 && can_change_color()) ? f : (COLORS >= 16 ? ansi[f] : ansi[f] & 7);
    if (COLOR_PAIRS >= 128 + 1)
        for (b = 0; b < 8; b++)
            for (f = 0; f < 16; f++)
                init_pair(1 + b * 16 + f, con_cmap[f], con_cmap[b]);
    else
        for (b = 0; b < 8; b++)
            for (f = 0; f < 8; f++)
                init_pair(1 + b * 16 + f, con_cmap[f] & 7, con_cmap[b] & 7);
}

static void con_start(void)
{
    if (con_started)
        return;
    setlocale(LC_ALL, "");
    initscr();
    SCR_W = COLS < SCR_MAXW ? COLS : SCR_MAXW;
    SCR_H = LINES < SCR_MAXH ? LINES : SCR_MAXH;
    con_wl = 1; con_wt = 1; con_wr = SCR_W; con_wb = SCR_H;
    cbreak();
    noecho();
    nonl();
    keypad(stdscr, TRUE);
    set_escdelay(40);
    nodelay(stdscr, TRUE);
    if (has_colors()) {
        start_color();
        con_color = 1;
        con_pair_init();
    }
    curs_set(1);
    erase();
    memset(con_scr, 0, sizeof con_scr);
    memset(con_shown_ch, 0, sizeof con_shown_ch);
    memset(con_shown_at, 0, sizeof con_shown_at);
    con_started = 1;
    atexit(con_end);
}

/* set the colour n (0..15 as in a DOS attribute) to r, g, b given from 0 to 63 like the VGA palette; returns 0 when the terminal cannot */
int con_set_palette(int n, int r, int g, int b)
{
    con_start();
    if (!con_color || !can_change_color() || COLORS < 16)
        return 0;
    init_color(n, r * 1000 / 63, g * 1000 / 63, b * 1000 / 63);
    return 1;
}

int con_can_palette(void)
{
    con_start();
    return con_color && can_change_color() && COLORS >= 16;
}

void con_cursor(int on)
{
    con_start();
    con_cursor_on = on;
    curs_set(on);
}

static void con_cell_draw(int y, int x)
{
    ConCell *c = &con_scr[y][x];
    unsigned char a = c->attr;
    cchar_t cc;
    wchar_t w[2];
    attr_t at = 0;
    int fg = a & 15, bg = (a >> 4) & 7;
    w[0] = c->ch ? con_cp437[c->ch] : L' ';
    w[1] = 0;
    if (con_color) {
        int pair = 1 + bg * 16 + fg;
        if (COLOR_PAIRS < 129) {
            pair = 1 + bg * 16 + (fg & 7);
            if (fg >= 8) at |= A_BOLD;
        }
        setcchar(&cc, w, at, (short)pair, NULL);
    } else {
        if (bg) at |= A_REVERSE;
        else if (fg >= 9) at |= A_BOLD;
        else if (fg == 8) at |= A_DIM;
        setcchar(&cc, w, at, 0, NULL);
    }
    mvadd_wch(y, x, &cc);
}

/* draw what changed since the last time, and put the cursor where the program has it */
static void con_flush(void)
{
    int x, y;
    if (!con_started)
        return;
    if (con_dirty) {
        for (y = 0; y < SCR_H; y++)
            for (x = 0; x < SCR_W; x++)
                if (con_scr[y][x].ch != con_shown_ch[y][x] || con_scr[y][x].attr != con_shown_at[y][x]) {
                    con_cell_draw(y, x);
                    con_shown_ch[y][x] = con_scr[y][x].ch;
                    con_shown_at[y][x] = con_scr[y][x].attr;
                }
        con_dirty = 0;
    }
    move(con_wt - 1 + con_y - 1, con_wl - 1 + con_x - 1);
    refresh();
}

static void con_scroll(void)
{
    int y, x;
    for (y = con_wt; y < con_wb; y++)
        for (x = con_wl; x <= con_wr; x++)
            con_scr[y - 1][x - 1] = con_scr[y][x - 1];
    for (x = con_wl; x <= con_wr; x++) {
        con_scr[con_wb - 1][x - 1].ch = ' ';
        con_scr[con_wb - 1][x - 1].attr = con_attr;
    }
    con_dirty = 1;
}

int con_cols(void) { con_start(); return SCR_W; }
int con_rows(void) { con_start(); return SCR_H; }

void conio_window(int l, int t, int r, int b)
{
    con_start();
    con_wl = l; con_wt = t; con_wr = r; con_wb = b;
    con_x = con_y = 1;
}

void conio_gotoxy(int x, int y)
{
    int w = con_wr - con_wl + 1, h = con_wb - con_wt + 1;
    con_x = x < 1 ? 1 : x > w ? w : x;
    con_y = y < 1 ? 1 : y > h ? h : y;
}

int conio_wherex(void) { return con_x; }
int conio_wherey(void) { return con_y; }
void conio_textattr(int a) { con_start(); con_attr = (unsigned char)a; }

void conio_clrscr(void)
{
    int x, y;
    con_start();
    for (y = con_wt; y <= con_wb; y++)
        for (x = con_wl; x <= con_wr; x++) {
            con_scr[y - 1][x - 1].ch = ' ';
            con_scr[y - 1][x - 1].attr = con_attr;
        }
    con_x = con_y = 1;
    con_dirty = 1;
}

int conio_putch(int c)
{
    int w = con_wr - con_wl + 1, h = con_wb - con_wt + 1;
    con_start();
    c &= 0xFF;
    if (c == '\r') { con_x = 1; return c; }
    if (c == '\n') {
        if (con_y >= h) con_scroll(); else con_y++;
        return c;
    }
    if (c == '\b') { if (con_x > 1) con_x--; return c; }
    con_scr[con_wt - 1 + con_y - 1][con_wl - 1 + con_x - 1].ch = (unsigned char)c;
    con_scr[con_wt - 1 + con_y - 1][con_wl - 1 + con_x - 1].attr = con_attr;
    con_dirty = 1;
    if (++con_x > w) {
        con_x = 1;
        if (con_y >= h) con_scroll(); else con_y++;
    }
    return c;
}

int conio_cputs(const char *s)
{
    while (*s)
        conio_putch((unsigned char)*s++);
    return 0;
}

int conio_cprintf(const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    conio_cputs(buf);
    return (int)strlen(buf);
}

/* the cells of a part of the screen, a character and an attribute byte each (as gettext does in DOS) */
int conio_gettext(int l, int t, int r, int b, void *dest)
{
    unsigned char *d = (unsigned char *)dest;
    int x, y;
    for (y = t; y <= b; y++)
        for (x = l; x <= r; x++) {
            *d++ = con_scr[y - 1][x - 1].ch;
            *d++ = con_scr[y - 1][x - 1].attr;
        }
    return 1;
}

void conio_gettextinfo(struct text_info *ti)
{
    con_start();
    ti->currmode = con_color ? C80 : BW80;
}

void conio_textmode(int mode)
{
    (void)mode;
    con_start();
}

void conio_gettime(struct time *t)
{
    time_t n = time(NULL);
    struct tm *tm = localtime(&n);
    t->ti_hour = (unsigned char)tm->tm_hour;
    t->ti_min = (unsigned char)tm->tm_min;
    t->ti_sec = (unsigned char)tm->tm_sec;
    t->ti_hund = 0;
}

/* ---- sound ----
   TINY plays tones: sound(hz), then delay(ms), then nosound(), in real time (the PC speaker, a square wave). Here sound() and nosound() only
   remember the frequency, and delay(ms) puts ms milliseconds of that square wave into a pipe to `paplay` (the PulseAudio player of the N900) and waits
   as long as it lasts, so the picture stays in step with the voice.

   Why paplay and not the library: the sound policy of the N900 puts every program it does not know into the group "othermedia" and limits
   the volume of its streams to about 30 % (-30 dB), however loud the phone is set. A stream that asks for its volume itself when it connects
   (paplay does, 100 %) keeps it; a stream made with the simple API of libpulse does not, and it was inaudible but for the clicks.
   When paplay is missing, or TINY_SILENT is set in the environment, delay() just waits. */

#include <signal.h>
#include <sys/time.h>

#define AUDIO_RATE 16000
#define AUDIO_LEAD_MS 60
#define AUDIO_AMP  8000          /* of 32767 (the stream plays at 100 %, so this is already loud on the phone); TINY_VOLUME=1..100 in the environment scales it */

static FILE *audio;                      /* the pipe to paplay */
static int audio_state;                  /* 0 not tried, 1 open, -1 not available */
static unsigned sound_hz;
static double lead_end, t_open;          /* when the sound already written will have been played; when the stream was opened (ms, wall clock) */
static unsigned long st_tone_ms, st_quiet_ms, st_calls, st_low;
static double sound_phase;
static float gain = 1.0f;                /* con_gain(): a share of the volume, for the sounds that are louder than the voice */

/* TINY_AUDIO_LOG=/path writes what happens to the sound (for testing; the screen has no room for messages) */
static void audio_log(const char *fmt, ...)
{
    const char *name = getenv("TINY_AUDIO_LOG");
    FILE *fp;
    va_list ap;
    if (!name) return;
    fp = fopen(name, "a");
    if (!fp) return;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fputc('\n', fp);
    fclose(fp);
}

static void audio_flush(void);

static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void audio_close(void)
{
    static int closing;
    if (closing) return;
    closing = 1;
    audio_flush();
    audio_log("played: %lu ms of tone, %lu ms of silence, %lu writes (%lu of them shorter than 5 ms), %.0f ms since the start", st_tone_ms, st_quiet_ms, st_calls, st_low, now_ms() - t_open);
    if (audio_state == 1 && audio)
        pclose(audio);                           /* waits until paplay has played the rest */
    audio = NULL;
    audio_state = -1;
}

static void audio_open(void)
{
    /* a WAV header for a stream of unknown length: paplay reads the sound from its standard input */
    static const unsigned char wav[44] = {
        'R', 'I', 'F', 'F', 0xFF, 0xFF, 0xFF, 0x7F, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 16, 0, 0, 0,
        1, 0, 1, 0, (AUDIO_RATE & 255), (AUDIO_RATE >> 8), 0, 0, ((AUDIO_RATE * 2) & 255), ((AUDIO_RATE * 2) >> 8), 0, 0, 2, 0, 16, 0,
        'd', 'a', 't', 'a', 0xFF, 0xFF, 0xFF, 0x7F
    };
    audio_state = -1;
    if (getenv("TINY_SILENT")) {
        audio_log("TINY_SILENT is set");
        return;
    }
    if (access("/usr/bin/paplay", X_OK) != 0) {
        audio_log("no /usr/bin/paplay");
        return;
    }
    /* the X display would go into the properties of the stream and the policy would take it for the stream of a window */
    unsetenv("DISPLAY");
    signal(SIGPIPE, SIG_IGN);
    audio = popen("exec /usr/bin/paplay --client-name=TINY --stream-name='robot voice' --volume=65536 2>/dev/null", "w");
    if (!audio) {
        audio_log("popen failed");
        return;
    }
    if (fwrite(wav, 1, sizeof wav, audio) != sizeof wav) {
        audio_log("cannot write to paplay");
        pclose(audio);
        audio = NULL;
        return;
    }
    audio_state = 1;
    t_open = now_ms();
    lead_end = 0;
    audio_log("sound opened through paplay, %d Hz", AUDIO_RATE);
    atexit(audio_close);
}

/* The samples wait in a small buffer and go out in pieces of about 30 ms: the voice asks for 2 ms of sound at a time (a vowel is a tone that
   changes every 2 ms), and a write for each of them would leave gaps. When the player has run dry, or has been idle, 50 ms of silence
   go first, so there is a lead again and the stream does not stop between two pieces. */
#define AUDIO_CHUNK (AUDIO_RATE * 30 / 1000)
static short abuf[AUDIO_CHUNK + AUDIO_RATE / 10];
static unsigned an;
static float lp;                         /* the low-pass filter that takes the edge off the square wave */

static void audio_flush(void)
{
    double t;
    short silence[AUDIO_RATE * 50 / 1000];
    if (audio_state != 1 || an == 0)
        return;
    t = now_ms();
    if (t > lead_end) {                          /* run dry, or idle: rebuild the lead */
        memset(silence, 0, sizeof silence);
        if (fwrite(silence, 1, sizeof silence, audio) != sizeof silence) { audio_log("write to paplay failed"); audio_close(); return; }
        lead_end = t + 50.0;
    }
    if (getenv("TINY_AUDIO_DUMP")) {               /* for testing: the samples as they go to the player (raw s16le, mono, 16 kHz); not in /tmp, it is only 1 MB */
        FILE *df = fopen(getenv("TINY_AUDIO_DUMP"), "ab");
        if (df) { if (fwrite(abuf, sizeof abuf[0], an, df) != an) { /* test output only */ } fclose(df); }
    }
    if (fwrite(abuf, sizeof abuf[0], an, audio) != an || fflush(audio) != 0) {
        audio_log("write to paplay failed");
        audio_close();
        return;
    }
    lead_end += an * 1000.0 / AUDIO_RATE;
    an = 0;
    /* keep the program at most AUDIO_LEAD_MS ahead of the sound: the delays of TINY, and so the highlight of the spoken words, must take their real time */
    t = lead_end - now_ms() - AUDIO_LEAD_MS;
    if (t > 1.0)
        usleep((useconds_t)(t * 1000.0));
}

/* `ms` milliseconds of the current tone (or silence) into the buffer; the buffer goes out when it is full, and that waits for the sound card */
static void audio_play(unsigned ms)
{
    unsigned n = ms * AUDIO_RATE / 1000, i;
    double step = sound_hz ? (double)sound_hz / AUDIO_RATE : 0.0;
    static float amp;
    if (amp == 0.0f) {
        const char *v = getenv("TINY_VOLUME");
        int pc = v ? atoi(v) : 100;
        amp = (float)AUDIO_AMP * (float)(pc < 1 ? 1 : pc > 100 ? 100 : pc) / 100.0f;
    }
    if (sound_hz) st_tone_ms += ms; else st_quiet_ms += ms;
    st_calls++;
    if (ms < 5) st_low++;
    for (i = 0; i < n; i++) {
        float x = 0.0f;
        if (step > 0.0) {
            sound_phase += step;
            if (sound_phase >= 1.0) sound_phase -= 1.0;
            x = (sound_phase < 0.5 ? amp : -amp) * gain;
        }
        lp += 0.85f * (x - lp);      /* a mild low-pass: 0.45 cut the voice (formants above 1 kHz, hiss, clicks) almost out */
        abuf[an++] = (short)lp;
        if (an >= AUDIO_CHUNK)
            audio_flush();
        if (audio_state != 1)
            return;
    }
}

/* While TINY waits for a key the stream would run dry; the sound server (on the N900 its policy for the "x-maemo" streams) then lets a stream go quiet
   for good, and the voice came out silent after a pause of some seconds. A little silence now and then keeps the stream playing. */
static void audio_idle(void)
{
    unsigned save;
    if (audio_state != 1 || lead_end - now_ms() >= 40.0)
        return;
    save = sound_hz;
    sound_hz = 0;
    audio_play(30);
    sound_hz = save;
}

void con_gain(int percent)
{
    gain = (percent < 0 ? 0 : percent > 100 ? 100 : percent) / 100.0f;
}

void conio_sound(unsigned hz)
{
    if (hz && getenv("TINY_AUDIO_TRACE") && audio_state == 1) audio_log("snd %u @%.0f", hz, now_ms() - t_open);
    sound_hz = hz;
    if (hz && audio_state == 0)
        audio_open();
}

void conio_nosound(void)
{
    sound_hz = 0;
}

void conio_delay(unsigned ms)
{
    if (audio_state == 1) {
        if (ms >= 5)
            con_flush();
        audio_play(ms);
        return;
    }
    if (ms < 5) { usleep(ms * 1000u); return; }
    con_flush();
    napms((int)ms);
}

/* The keys wait in a small queue: a DOS key is a character code, or 0 followed by a scan code. */
static int con_q[4], con_qn;

static void con_push(int a, int b)
{
    con_q[con_qn++] = a;
    if (b >= 0) con_q[con_qn++] = b;
}

/* read one key from ncurses (if there is one) into the queue */
static void con_poll(void)
{
    int c = wgetch(stdscr);
    if (c != ERR) audio_log("key %d", c);
    if (con_qn > 2) return;
    switch (c) {
    case ERR: return;
    case 10: case 13: case KEY_ENTER: con_push(13, -1); return;
    case 127: case KEY_BACKSPACE: con_push(8, -1); return;
    case 1:  con_push(0, 59); return;          /* Ctrl-A = F1 help       */
    case 19: con_push(0, 60); return;          /* Ctrl-S = F2 sound      */
    case 22: con_push(0, 61); return;          /* Ctrl-V = F3 voice      */
    case 20: con_push(0, 62); return;          /* Ctrl-T = F4 creativity */
    case KEY_F(1): con_push(0, 59); return;
    case KEY_F(2): con_push(0, 60); return;
    case KEY_F(3): con_push(0, 61); return;
    case KEY_F(4): con_push(0, 62); return;
    case KEY_UP: con_push(0, 72); return;
    case KEY_DOWN: con_push(0, 80); return;
    case KEY_LEFT: con_push(0, 75); return;
    case KEY_RIGHT: con_push(0, 77); return;
    default:
        if (c >= KEY_MIN) con_push(0, 1);
        else con_push(c & 0xFF, -1);
    }
}

int conio_kbhit(void)
{
    con_start();
    if (con_qn == 0) {
        con_flush();
        con_poll();
        if (con_qn == 0)
            audio_idle();
    }
    return con_qn > 0;
}

int conio_getch(void)
{
    int c, i;
    con_start();
    audio_flush();
    while (con_qn == 0) {
        con_flush();
        con_poll();
        if (con_qn == 0) {
            audio_idle();
            if (audio_state != 1) napms(15);     /* with sound, the silence written above takes the time */
            else napms(5);
        }
    }
    c = con_q[0];
    for (i = 1; i < con_qn; i++) con_q[i - 1] = con_q[i];
    con_qn--;
    return c;
}


