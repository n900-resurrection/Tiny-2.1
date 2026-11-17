# TINY on the Nokia N900 (branch `n900`)

This branch builds TINY 2.1 on a POSIX system, in particular for the Nokia N900 (Maemo 5, ARM, glibc 2.5), with the full-screen interface:

- `conio_ncurses.c` / `conio_ncurses.h` provide the DOS `conio` functions that `TINY.C` uses (screen, colours, keys) on top of ncurses (a virtual 80x25 screen, drawn with wide characters).
- `TINY.C` has small changes: the interface is built when `HAVE_UI` is defined (Turbo C, or `-DCONIO_NCURSES`), the palette is set with ncurses, and the function keys F1..F4
  are also on Ctrl-A, Ctrl-S, Ctrl-V and Ctrl-T (a phone keyboard has no F-keys).
- **Sound:** the PC speaker is not on the phone, so the tones of TINY (the keys, the robot voice) are played as a square wave through PulseAudio (`libpulse-simple`, loaded at run time; no headers needed). `delay()` writes the tone to the sound server and so keeps the voice and the picture in step. `TINY_SILENT=1` or a missing PulseAudio gives the old silent behaviour.
- **The layout follows the size of the terminal** (`layout()` in `TINY.C`; the DOS version is still 80 x 25): the frames, the chat window, the neural monitor (the compact one when there are fewer than 18 rows),
  the line for the question (framed only from 24 rows), the status bar and the boot screen are computed from the columns and rows. It was tried at 72x16 and 72x18 (the N900 terminal with its normal font, windowed and full screen) and at 80x25.
  The terminal needs at least 60 x 14 characters; with a smaller one `tiny` says so, and `tiny /P` is the plain text mode.
- The terminal should have 256 colours (`TERM=xterm-256color`) for the neon palette; with fewer, the nearest standard colours are used.

Build: `make` (see the top of the `Makefile`). Packages for the N900: <https://n900-resurrection.github.io>.
