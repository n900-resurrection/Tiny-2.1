# TINY on the Nokia N900 (branch `n900`)

This branch builds TINY 2.1 on a POSIX system, in particular for the Nokia N900 (Maemo 5, ARM, glibc 2.5), with the full-screen interface:

- `conio_ncurses.c` / `conio_ncurses.h` provide the DOS `conio` functions that `TINY.C` uses (screen, colours, keys) on top of ncurses (a virtual 80x25 screen, drawn with wide characters).
- `TINY.C` has small changes: the interface is built when `HAVE_UI` is defined (Turbo C, or `-DCONIO_NCURSES`), the palette is set with ncurses, and the function keys F1..F4
  are also on Ctrl-A, Ctrl-S, Ctrl-V and Ctrl-T (a phone keyboard has no F-keys).
- There is no sound: the PC speaker is not on the phone. The robot voice runs silently with the same timing.
- The terminal needs at least 80 x 25 characters; with a smaller one `tiny` says so, and `tiny /P` is the plain text mode.
- The terminal should have 256 colours (`TERM=xterm-256color`) for the neon palette; with fewer, the nearest standard colours are used.

Build: `make` (see the top of the `Makefile`). Packages for the N900: <https://n900-resurrection.github.io>.
