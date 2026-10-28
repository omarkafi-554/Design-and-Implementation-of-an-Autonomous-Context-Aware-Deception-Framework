#!/usr/bin/env python3
"""ingest_zeek.py - live reader of Zeek's conn.log (Algorithm 2.9, event source 2).

This Zeek writes its logs in the default TSV format: comment lines start with '#', the
'#fields' line names the columns, values are separated by TAB, '-' means empty.
The column names are taken from the '#fields' line of the file itself (read again after
every hourly rotation), so the reader does not depend on a fixed column order.
Zeek's configuration is not changed.

Every connection record becomes one event:  ("ZEEK_CONN", epoch_seconds, record_dict)
Filtering (watched ports, REJ / S0) is done later by probe_watcher.py (Week 4).

Demo (Ctrl+C to stop):
    python3 orchestrator/ingest_zeek.py
"""
import argparse
import queue
import threading

from tail import Tail

DEFAULT_PATH = "/opt/zeek/spool/zeek/conn.log"
KEEP = ("ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
        "service", "duration", "conn_state", "history")
INTS = ("id.orig_p", "id.resp_p")


class ZeekConnReader:
    def __init__(self, q, path=DEFAULT_PATH, from_start=False):
        self.q, self.path = q, path
        self.fields, self.sep = None, "\t"
        self.n_events, self.n_bad = 0, 0
        self._read_header()                           # learn the columns of the current file
        self.tail = Tail(path, from_start=from_start)  # then follow only new records

    def _read_header(self):
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.startswith("#"):
                        break
                    self.parse(line.rstrip("\n"))
        except FileNotFoundError:
            pass

    def parse(self, line):
        if line.startswith("#"):
            if line.startswith("#separator"):
                val = line.split(" ", 1)[1].strip()
                self.sep = bytes(val, "ascii").decode("unicode_escape")
            elif line.startswith("#fields"):
                self.fields = line.split(self.sep)[1:]
            return None
        if not line.strip() or self.fields is None:
            return None
        vals = line.split(self.sep)
        if len(vals) != len(self.fields):
            self.n_bad += 1
            return None
        rec = dict(zip(self.fields, vals))
        out = {k: (None if rec.get(k) in ("-", "(empty)", None) else rec[k]) for k in KEEP}
        try:
            out["ts"] = float(out["ts"])
            for k in INTS:
                out[k] = int(out[k]) if out[k] is not None else None
        except (TypeError, ValueError):
            self.n_bad += 1
            return None
        return ("ZEEK_CONN", out["ts"], out)

    def run(self, stop=None):
        for line in self.tail.lines(stop):
            item = self.parse(line)
            if item:
                self.q.put(item)
                self.n_events += 1

    def start(self, stop):
        t = threading.Thread(target=self.run, args=(stop,), name="zeek-reader", daemon=True)
        t.start()
        return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--from-start", action="store_true")
    a = ap.parse_args()
    q, stop = queue.Queue(), threading.Event()
    r = ZeekConnReader(q, a.path, a.from_start)
    r.start(stop)
    print(f"[+] following {a.path} (Ctrl+C to stop)")
    try:
        while True:
            kind, t, c = q.get()
            print(f"{t:.3f}  {c['id.orig_h']}:{c['id.orig_p']} -> {c['id.resp_h']}:{c['id.resp_p']}"
                  f"  {c['proto']}  {c['conn_state']}")
    except KeyboardInterrupt:
        stop.set()
        print(f"\n[+] events {r.n_events}, malformed {r.n_bad}, rotations {r.tail.rotations}")
