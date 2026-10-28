#!/usr/bin/env python3
"""test_orchestrator.py - main-loop logic with a stub classifier and a temporary database.
(The real classifier is checked by test_parity.py; here only the orchestration is tested.)
Run:  python3 orchestrator/tests/test_orchestrator.py   (or with pytest)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator import Orchestrator  # noqa: E402
from state_db import StateDB  # noqa: E402


class StubClassifier:
    """Same interface as live_classify.Classifier: pending below k, then a fixed verdict."""
    def __init__(self, k=3, verdict="Bot"):
        self.k, self.verdict, self.calls = k, verdict, 0

    def classify(self, cmds, login_ts=None, tty_records=None):
        self.calls += 1
        n = len([c for c in cmds if (c[1] or "").strip()])
        if n < self.k:
            return {"verdict": "Uncertain", "p_human": None, "model": None, "n_lines": n,
                    "reason": f"pending: {n}/{self.k} input lines"}
        return {"verdict": self.verdict, "p_human": 0.0 if self.verdict == "Bot" else 1.0,
                "model": "STUB", "n_lines": n, "reason": f"STUB at line {self.k}"}


def setup(verdict="Bot"):
    db = StateDB(os.path.join(tempfile.mkdtemp(prefix="acdf_orch_"), "t.db"))
    clf = StubClassifier(verdict=verdict)
    return Orchestrator(clf, db, out=lambda *_: None), clf, db


def ev(eid, sid="s1", **kw):
    return {"eventid": eid, "session": sid, "src_ip": "192.168.71.129", **kw}


def feed(o, events):
    for t, e in events:
        o.handle(("COWRIE_EVENT", t, e))


def session(sid="s1", cmds=("whoami", "uname -a", "ls", "exit")):
    out = [(100.0, ev("cowrie.session.connect", sid)),
           (101.0, ev("cowrie.login.success", sid, username="root"))]
    out += [(102.0 + i, ev("cowrie.command.input", sid, input=c)) for i, c in enumerate(cmds)]
    out.append((110.0, ev("cowrie.session.closed", sid)))
    return out


def test_decision_exactly_at_line_k():
    o, clf, db = setup()
    feed(o, session()[:4])                          # connect, login, 2 commands -> no decision
    assert db.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    feed(o, session()[4:5])                         # 3rd command -> decision
    rows = db.db.execute("SELECT ts, action, session_id FROM decisions").fetchall()
    assert [tuple(r) for r in rows] == [(104.0, "LOG_ONLY", "s1")]


def test_one_verdict_per_session_and_close():
    o, clf, db = setup("Human")
    feed(o, session())
    assert db.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert clf.calls == 3                           # stops classifying after the verdict
    s = dict(db.db.execute("SELECT * FROM sessions WHERE session_id='s1'").fetchone())
    assert (s["verdict"], s["username"], s["started"], s["ended"], s["n_lines"]) == \
        ("Human", "root", 100.0, 110.0, 4)
    assert o.ctx == {} and o.stats["sessions_closed"] == 1


def test_short_session_has_no_verdict():
    o, clf, db = setup()
    feed(o, session(cmds=("ls", "")))               # 1 real line, 1 empty Enter
    assert db.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
    s = dict(db.db.execute("SELECT * FROM sessions WHERE session_id='s1'").fetchone())
    assert s["verdict"] is None and s["n_lines"] == 1


def test_two_interleaved_sessions():
    o, clf, db = setup()
    a, b = session("A"), session("B")
    feed(o, sorted(a + b, key=lambda x: x[0]))
    got = {r[0] for r in db.db.execute("SELECT session_id FROM decisions")}
    assert got == {"A", "B"} and o.ctx == {}


def test_zeek_counted_and_bad_event_does_not_stop():
    o, clf, db = setup()
    o.handle(("ZEEK_CONN", 1.0, {"id.resp_p": 3306, "conn_state": "REJ"}))
    assert o.stats["zeek_conns"] == 1
    import queue
    import threading
    q, stop = queue.Queue(), threading.Event()
    q.put(("COWRIE_EVENT", 1.0, {"eventid": "cowrie.command.input"}))   # no 'session' key
    for t, e in session():
        q.put(("COWRIE_EVENT", t, e))
    threading.Timer(1.0, stop.set).start()
    o.run(q, stop)
    assert db.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"[+] {len(tests)} tests passed")
