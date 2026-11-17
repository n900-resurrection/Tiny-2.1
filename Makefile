# TINY 2.1 on a POSIX system (the N900 port): the full-screen interface runs on ncurses (conio_ncurses.c), the network is unchanged.
# make          build `tiny` with the full-screen interface (needs ncurses with wide characters: libncursesw)
# make plain    build `tiny-plain`, the text-only version (no ncurses needed)
# make test     ask a few questions in the plain mode
# Cross-compiling: make CC=/path/to/arm-gcc NCURSES_LIBS="-Wl,-Bstatic -lncursesw -Wl,-Bdynamic"
CC ?= gcc
CFLAGS ?= -O2
NCURSES_LIBS ?= -lncursesw
# the wide-character curses.h (Debian: ncursesw/curses.h is found by itself; give -I.../include/ncursesw when your system keeps it apart)

all: tiny

tiny: TINY.C conio_ncurses.c conio_ncurses.h
	$(CC) $(CFLAGS) -Wall -DCONIO_NCURSES -x c TINY.C conio_ncurses.c $(NCURSES_LIBS) -lm -ldl -o $@

plain: tiny-plain

tiny-plain: TINY.C
	$(CC) $(CFLAGS) -Wall -x c TINY.C -lm -o $@

test: tiny-plain
	printf 'what is the capital of france\nwhat is 7 plus 5\nquit\n' | ./tiny-plain /P

clean:
	rm -f tiny tiny-plain

.PHONY: all plain test clean
