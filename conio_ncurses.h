/* conio_ncurses.h - the DOS functions of <conio.h> that TINY.C uses, implemented in conio_ncurses.c on top of ncurses.
   TINY.C includes this header on a POSIX system (compile with -DCONIO_NCURSES, link conio_ncurses.c with -lncursesw).
   The implementation is in a separate file because <curses.h> defines macros (box, hline, getch ...) that clash with TINY.C. */

#ifndef CONIO_NCURSES_H
#define CONIO_NCURSES_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>           /* isatty */

#define MONO 7
#define BW40 0
#define BW80 2
#define C40  1
#define C80  3

struct text_info { unsigned char currmode; };
struct time { unsigned char ti_min, ti_hour, ti_hund, ti_sec; };

void conio_window(int l, int t, int r, int b);
void conio_gotoxy(int x, int y);
int  conio_wherex(void);
int  conio_wherey(void);
void conio_textattr(int a);
void conio_clrscr(void);
int  conio_putch(int c);
int  conio_cputs(const char *s);
int  conio_cprintf(const char *fmt, ...);
int  conio_gettext(int l, int t, int r, int b, void *dest);
void conio_gettextinfo(struct text_info *ti);
void conio_textmode(int mode);
void conio_gettime(struct time *t);
void conio_sound(unsigned hz);
void conio_nosound(void);
void conio_delay(unsigned ms);
int  conio_kbhit(void);
int  conio_getch(void);

/* what TINY.C needs besides conio: the cursor, and the palette of the terminal */
void con_end(void);
void con_cursor(int on);
int  con_can_palette(void);
void con_gain(int percent);   /* the share of the volume (0..100) of the following sounds */
int  con_cols(void);      /* the size of the terminal */
int  con_rows(void);
int  con_set_palette(int n, int r, int g, int b);

#ifndef CONIO_IMPL      /* not in conio_ncurses.c: curses.h has macros of its own there */
#define window    conio_window
#define gotoxy    conio_gotoxy
#define wherex    conio_wherex
#define wherey    conio_wherey
#define textattr  conio_textattr
#define clrscr    conio_clrscr
#define putch     conio_putch
#define cputs     conio_cputs
#define cprintf   conio_cprintf
#define gettext   conio_gettext
#define gettextinfo conio_gettextinfo
#define textmode  conio_textmode
#define gettime   conio_gettime
#define sound     conio_sound
#define nosound   conio_nosound
#define delay     conio_delay
#define kbhit     conio_kbhit
#define getch     conio_getch
#define random(n) (rand() % (n))
#endif

#endif
