#!/usr/bin/env python3
"""state_db.py - Algorithm 2.17 STATE-DB.

Auditable, replayable history of every decision of the orchestrator.
  * WAL journal (readers never block the writer) and foreign keys ON.
  * log_decision() writes the session row and the decision row in ONE transaction:
    either both are stored or neither.
  * One lock: the orchestrator's threads may share one StateDB object.
"""
import json
import os
import sqlite3
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "schema.sql")
SESSION_COLS = ("src_ip", "username", "started", "ended", "n_lines", "verdict", "p_human",
                "model_name", "features_json")


def _j(x):
    return None if x is None else json.dumps(x, sort_keys=True, default=str)


class StateDB:
    def __init__(self, path=os.path.join(HERE, "acdf.db")):
        self.path = path
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(open(SCHEMA).read())

    def close(self):
        self.db.close()

    # ---------------------------------------------------------------- internal
    def _tx(self, fn):
        """Run fn(cursor) inside BEGIN ... COMMIT; ROLLBACK on any error."""
        with self.lock:
            cur = self.db.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                out = fn(cur)
                cur.execute("COMMIT")
                return out
            except Exception:
                cur.execute("ROLLBACK")
                raise

    @staticmethod
    def _upsert_session(cur, session_id, fields):
        fields = {k: v for k, v in fields.items() if k in SESSION_COLS}
        if "features_json" in fields and not isinstance(fields["features_json"], (str, type(None))):
            fields["features_json"] = _j(fields["features_json"])
        cur.execute("INSERT OR IGNORE INTO sessions(session_id) VALUES (?)", (session_id,))
        if fields:
            sets = ", ".join(f"{k} = ?" for k in fields)
            cur.execute(f"UPDATE sessions SET {sets} WHERE session_id = ?",
                        (*fields.values(), session_id))

    # ---------------------------------------------------------------- sessions
    def upsert_session(self, session_id, **fields):
        self._tx(lambda c: self._upsert_session(c, session_id, fields))

    def close_session(self, session_id, end=None):
        self.upsert_session(session_id, ended=end if end is not None else time.time())

    # ---------------------------------------------------------------- decisions
    def log_decision(self, event_type, action, src_ip=None, session_id=None, reason=None,
                     latency_ms=None, ts=None, session=None):
        """session: optional dict of session fields (verdict, p_human, ...) stored in the
        same transaction as the decision."""
        ts = ts if ts is not None else time.time()

        def fn(cur):
            if session_id is not None and session:
                self._upsert_session(cur, session_id, session)
            cur.execute("INSERT INTO decisions(ts, event_type, src_ip, session_id, action, "
                        "reason_json, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (ts, event_type, src_ip, session_id, action, _j(reason), latency_ms))
            return cur.lastrowid
        return self._tx(fn)

    # ---------------------------------------------------------------- decoys and rules
    def add_decoy(self, decoy_id, svc, profile=None, container_id=None, addr=None, port=None,
                  attacker_ip=None, started=None, status="running"):
        self._tx(lambda c: c.execute(
            "INSERT INTO decoys(decoy_id, svc, profile_json, container_id, addr, port, "
            "attacker_ip, started, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (decoy_id, svc, _j(profile), container_id, addr, port, attacker_ip,
             started if started is not None else time.time(), status)))

    def close_decoy(self, decoy_id, status="retired", stopped=None):
        self._tx(lambda c: c.execute(
            "UPDATE decoys SET stopped = ?, status = ? WHERE decoy_id = ?",
            (stopped if stopped is not None else time.time(), status, decoy_id)))

    def add_rule(self, decoy_id, src_ip, dport, target, comment, added=None):
        return self._tx(lambda c: c.execute(
            "INSERT INTO rules(decoy_id, src_ip, dport, target, comment, added) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (decoy_id, src_ip, dport, target, comment,
             added if added is not None else time.time())).lastrowid)

    def close_rule(self, comment, removed=None):
        self._tx(lambda c: c.execute(
            "UPDATE rules SET removed = ? WHERE comment = ? AND removed IS NULL",
            (removed if removed is not None else time.time(), comment)))

    def open_rules(self):
        with self.lock:
            return [dict(r) for r in self.db.execute("SELECT * FROM rules WHERE removed IS NULL")]

    def log_scan(self, services, ts=None):
        return self._tx(lambda c: c.execute(
            "INSERT INTO scans(ts, services_json) VALUES (?, ?)",
            (ts if ts is not None else time.time(), _j(services))).lastrowid)

    # ---------------------------------------------------------------- replay
    def replay(self, t1, t2):
        """Every decision between t1 and t2 in time order, with its session verdict, and the
        decoys and rules created in that window."""
        with self.lock:
            dec = [dict(r) for r in self.db.execute(
                "SELECT d.*, s.verdict, s.p_human, s.model_name, s.username "
                "FROM decisions d LEFT JOIN sessions s ON s.session_id = d.session_id "
                "WHERE d.ts BETWEEN ? AND ? ORDER BY d.ts, d.id", (t1, t2))]
            dco = [dict(r) for r in self.db.execute(
                "SELECT * FROM decoys WHERE started BETWEEN ? AND ? ORDER BY started", (t1, t2))]
            rul = [dict(r) for r in self.db.execute(
                "SELECT * FROM rules WHERE added BETWEEN ? AND ? ORDER BY added", (t1, t2))]
        return {"decisions": dec, "decoys": dco, "rules": rul}


if __name__ == "__main__":
    db = StateDB()
    with db.lock:
        n = {t: db.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
             for t in ("sessions", "decisions", "decoys", "rules", "scans")}
    print(db.path, db.db.execute("PRAGMA journal_mode").fetchone()[0], n)
