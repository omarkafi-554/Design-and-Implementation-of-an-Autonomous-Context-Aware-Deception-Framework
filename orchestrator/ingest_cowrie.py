#!/usr/bin/env python3
"""ingest_cowrie.py - live reader of Cowrie's JSON log (Algorithm 2.9, event source 1).

Every new line of cowrie.json becomes one event on the orchestrator queue:
    ("COWRIE_EVENT", epoch_seconds, event_dict)
Only the event types the orchestrator needs are passed on; a malformed line is counted and
skipped, it never stops the reader.

Demo (prints events as they arrive; Ctrl+C to stop):
    python3 orchestrator/ingest_cowrie.py
    python3 orchestrator/ingest_cowrie.py --from-start      # replay the current file first
"""
import argparse
import json
import os
import queue
import threading

from tail import Tail, ts

DEFAULT_PATH = os.path.expanduser("~/cowrie/var/log/cowrie/cowrie.json")
WANTED = {"cowrie.session.connect", "cowrie.login.success", "cowrie.login.failed",
          "cowrie.command.input", "cowrie.session.closed", "cowrie.log.closed",
          "cowrie.client.version", "cowrie.direct-tcpip.request"}


class CowrieReader:
    def __init__(self, q, path=DEFAULT_PATH, from_start=False):
        self.q, self.path = q, path
        self.tail = Tail(path, from_start=from_start)
        self.n_events, self.n_bad, self.n_skipped = 0, 0, 0

    def parse(self, line):
        """One JSON line -> event tuple, or None (empty, malformed or not needed)."""
        if not line.strip():
            return None
        try:
            ev = json.loads(line)
            if ev.get("eventid") not in WANTED or not ev.get("session"):
                self.n_skipped += 1
                return None
            return ("COWRIE_EVENT", ts(ev["timestamp"]), ev)
        except (ValueError, KeyError, TypeError):
            self.n_bad += 1
            return None

    def run(self, stop=None):
        for line in self.tail.lines(stop):
            item = self.parse(line)
            if item:
                self.q.put(item)
                self.n_events += 1

    def start(self, stop):
        t = threading.Thread(target=self.run, args=(stop,), name="cowrie-reader", daemon=True)
        t.start()
        return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--from-start", action="store_true")
    a = ap.parse_args()
    q, stop = queue.Queue(), threading.Event()
    r = CowrieReader(q, a.path, a.from_start)
    r.start(stop)
    print(f"[+] following {a.path} (Ctrl+C to stop)")
    try:
        while True:
            kind, t, ev = q.get()
            extra = ev.get("input") or ev.get("username") or ev.get("src_ip") or ""
            print(f"{t:.3f}  {ev['session']}  {ev['eventid']:28s} {extra}")
    except KeyboardInterrupt:
        stop.set()
        print(f"\n[+] events {r.n_events}, malformed {r.n_bad}, other types {r.n_skipped}, "
              f"rotations {r.tail.rotations}")
