#!/usr/bin/env python3
"""orchestrator.py - Algorithm 2.9 ORCHESTRATOR-MAIN-LOOP (Week 2 version).

One queue, two readers (Cowrie, Zeek), one main loop:
  COWRIE_EVENT -> update the session context -> classify once the session reaches input
                  line k -> log the verdict and the decision in the state database.
  ZEEK_CONN    -> counted only (probe handling arrives in Week 4).
Week 2 action is LOG_ONLY: nothing is deployed or redirected yet.

Run:  python3 orchestrator/orchestrator.py           (Ctrl+C or SIGTERM to stop)
"""
import argparse
import os
import queue
import signal
import sys
import threading
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_classify import Classifier, fv2  # noqa: E402  (fv2 = the frozen feature code)
from state_db import StateDB  # noqa: E402

ACTION = "LOG_ONLY"                                   # Week 2: decide and record only


class Session:
    """Context ctx of one SSH session (orchestrator_design.md, section 3)."""
    __slots__ = ("sid", "src_ip", "username", "login_ts", "started", "cmds", "verdict", "closed")

    def __init__(self, sid, t):
        self.sid, self.started = sid, t
        self.src_ip = self.username = self.login_ts = self.verdict = None
        self.cmds, self.closed = [], False


class Orchestrator:
    def __init__(self, classifier, db, out=print):
        self.clf, self.db, self.out = classifier, db, out
        self.ctx = {}                                 # session id -> Session
        self.stats = {"cowrie_events": 0, "zeek_conns": 0, "decisions": 0, "sessions_closed": 0}

    # ---------------------------------------------------------------- events
    def handle(self, item):
        kind, t, ev = item
        if kind == "COWRIE_EVENT":
            self.stats["cowrie_events"] += 1
            self.on_cowrie(t, ev)
        elif kind == "ZEEK_CONN":
            self.stats["zeek_conns"] += 1          # Week 4: probe_watcher.py

    def on_cowrie(self, t, ev):
        sid, eid = ev["session"], ev["eventid"]
        s = self.ctx.get(sid)
        if s is None:                               # also for sessions already open at start-up
            s = self.ctx[sid] = Session(sid, t)
        s.src_ip = s.src_ip or ev.get("src_ip")
        if eid == "cowrie.login.success" and s.login_ts is None:
            s.username, s.login_ts = ev.get("username"), t
        elif eid == "cowrie.command.input":
            s.cmds.append((t, ev.get("input")))
            if s.verdict is None:
                self.try_classify(s, t)
        elif eid == "cowrie.session.closed":
            self.close(s, t)

    # ---------------------------------------------------------------- decision
    def try_classify(self, s, t):
        t0 = time.perf_counter()
        r = self.clf.classify(s.cmds, s.login_ts)
        if r["reason"].startswith("pending:"):
            return                                  # not at line k yet: no decision
        latency_ms = (time.perf_counter() - t0) * 1000.0
        s.verdict = r["verdict"]                    # one verdict per session, never revised
        reason = {"verdict": r["verdict"], "p_human": r["p_human"], "model": r["model"],
                  "n_lines": r["n_lines"], "why": r["reason"],
                  "lag_s": round(time.time() - t, 3)}  # Cowrie log time -> decision
        self.db.log_decision("COWRIE_EVENT", ACTION, s.src_ip, s.sid, reason, round(latency_ms, 3),
                             ts=t, session={"src_ip": s.src_ip, "username": s.username,
                                            "started": s.started, "n_lines": r["n_lines"],
                                            "verdict": r["verdict"], "p_human": r["p_human"],
                                            "model_name": r["model"]})
        self.stats["decisions"] += 1
        self.out(f"[decision] {s.sid} {s.src_ip} user={s.username} -> {r['verdict']} "
                 f"(p={r['p_human']}, {r['reason']}, {latency_ms:.1f} ms, lag {reason['lag_s']} s)")

    def close(self, s, t):
        lines, _, _ = fv2.input_lines(s.cmds)
        self.db.upsert_session(s.sid, src_ip=s.src_ip, username=s.username, started=s.started,
                               ended=t, n_lines=len(lines))
        self.stats["sessions_closed"] += 1
        self.out(f"[closed]   {s.sid} {s.src_ip} lines={len(lines)} verdict={s.verdict}")
        del self.ctx[s.sid]

    # ---------------------------------------------------------------- loop
    def run(self, q, stop):
        while not stop.is_set():
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.handle(item)
            except Exception as e:                  # one bad event never stops the loop
                self.out(f"[error] {type(e).__name__}: {e} on {item[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    paths = cfg.get("paths", {})
    from ingest_cowrie import CowrieReader, DEFAULT_PATH as COWRIE
    from ingest_zeek import ZeekConnReader, DEFAULT_PATH as ZEEK

    clf = Classifier(a.config)
    db = StateDB(os.path.expanduser(paths.get("db", os.path.join(HERE, "acdf.db"))))
    q, stop = queue.Queue(), threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    CowrieReader(q, os.path.expanduser(paths.get("cowrie_json", COWRIE))).start(stop)
    ZeekConnReader(q, paths.get("zeek_conn", ZEEK)).start(stop)
    d = clf.describe()
    print(f"[+] orchestrator up: model {list(d['models'])}, k = {d['k']}, theta = {d['theta']}, "
          f"action = {ACTION}, db = {db.path}")
    orch = Orchestrator(clf, db)
    orch.run(q, stop)
    print(f"\n[+] stopped: {orch.stats}, open sessions {len(orch.ctx)}")
    db.close()


if __name__ == "__main__":
    main()
