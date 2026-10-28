#!/usr/bin/env python3
"""tail.py - follow a growing log file like `tail -F` (shared by the Cowrie and Zeek readers).

  * Yields complete lines only: a line still being written (no trailing newline) is kept
    in a buffer until its end arrives.
  * Survives rotation: when the path points to a new file (different inode), the rest of
    the old file is read first, then the new file is followed from its beginning.
  * Survives truncation (file got shorter): reading restarts at the beginning.
  * from_start=False (default) skips what is already in the file, so only new events count.
"""
import os
import time
from datetime import datetime


def ts(v):
    """Event time -> epoch seconds; identical to ts() in scripts/10_build_lab_v2.py."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


class Tail:
    def __init__(self, path, from_start=False, poll=0.2):
        self.path, self.poll = path, poll
        self.f, self.ino, self.buf = None, None, ""
        self.rotations = 0
        self._open(seek_end=not from_start)

    def _open(self, seek_end):
        try:
            f = open(self.path, "r", encoding="utf-8", errors="replace")
        except FileNotFoundError:
            self.f, self.ino = None, None          # not there yet: retry on the next poll
            return
        if self.f:
            self.f.close()
        self.f, self.ino, self.buf = f, os.fstat(f.fileno()).st_ino, ""
        if seek_end:
            self.f.seek(0, os.SEEK_END)

    def _rotated_or_truncated(self):
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return False                            # rotation in progress: keep the old file
        if st.st_ino != self.ino:
            return "rotated"
        if st.st_size < self.f.tell():
            return "truncated"
        return False

    def lines(self, stop=None):
        """Generator of complete lines (without the newline). stop: threading.Event or None."""
        while stop is None or not stop.is_set():
            if self.f is None:
                self._open(seek_end=False)
                if self.f is None:
                    time.sleep(self.poll)
                    continue
            chunk = self.f.readline()
            if chunk:
                self.buf += chunk
                if self.buf.endswith("\n"):
                    line, self.buf = self.buf[:-1], ""
                    yield line
                continue
            change = self._rotated_or_truncated()
            if change == "rotated":
                self.rotations += 1
                self._open(seek_end=False)          # old file fully read above -> new from start
            elif change == "truncated":
                self.f.seek(0)
                self.buf = ""
            else:
                time.sleep(self.poll)
