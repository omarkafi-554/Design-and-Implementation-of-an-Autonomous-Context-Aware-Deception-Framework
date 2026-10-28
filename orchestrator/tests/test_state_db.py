#!/usr/bin/env python3
"""test_state_db.py - STATE-DB checks on a temporary database (the real acdf.db is untouched).
Run:  python3 orchestrator/tests/test_state_db.py   (or with pytest)"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from state_db import StateDB  # noqa: E402


def fresh_db():
    d = tempfile.mkdtemp(prefix="acdf_test_")
    return StateDB(os.path.join(d, "test.db"))


def test_wal_and_foreign_keys():
    db = fresh_db()
    assert db.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert db.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_three_decisions_replay_in_order():
    db = fresh_db()
    db.log_decision("COWRIE_EVENT", "KEEP_IN_COWRIE", "10.0.0.5", "s1", {"why": "bot"}, 3.1, ts=102,
                    session={"verdict": "Bot", "p_human": 0.0, "model_name": "A3", "n_lines": 3})
    db.log_decision("PROBE", "DEPLOY", "10.0.0.6", None, {"port": 3306}, 1.0, ts=101)
    db.log_decision("COWRIE_EVENT", "DEPLOY", "10.0.0.7", "s2", {"why": "human"}, 2.0, ts=103,
                    session={"verdict": "Human", "p_human": 1.0, "model_name": "A3", "n_lines": 3})
    r = db.replay(100, 200)["decisions"]
    assert [x["ts"] for x in r] == [101, 102, 103]
    assert [x["verdict"] for x in r] == [None, "Bot", "Human"]


def test_decision_and_session_are_atomic():
    db = fresh_db()
    try:   # action = None violates NOT NULL -> the session upsert must be rolled back too
        db.log_decision("COWRIE_EVENT", None, "10.0.0.8", "s9", session={"verdict": "Human"})
        raise AssertionError("insert should have failed")
    except sqlite3.IntegrityError:
        pass
    assert db.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    assert db.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_unknown_session_is_rejected():
    db = fresh_db()
    try:
        db.log_decision("COWRIE_EVENT", "LOG_ONLY", "10.0.0.9", "no-such-session")
        raise AssertionError("foreign key not enforced")
    except sqlite3.IntegrityError:
        pass


def test_decoy_and_rule_lifecycle():
    db = fresh_db()
    db.add_decoy("d1", "sql", {"hostname": "db01"}, "c0ffee", "172.30.0.2", 3306, "10.0.0.6", started=110)
    db.add_rule("d1", "10.0.0.6", 3306, "172.30.0.2:3306", "acdf:d1", added=111)
    assert len(db.open_rules()) == 1
    db.close_rule("acdf:d1", removed=150)
    db.close_decoy("d1", stopped=151)
    assert db.open_rules() == []
    rep = db.replay(100, 200)
    assert rep["decoys"][0]["status"] == "retired" and rep["rules"][0]["removed"] == 150


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"[+] {len(tests)} tests passed")
