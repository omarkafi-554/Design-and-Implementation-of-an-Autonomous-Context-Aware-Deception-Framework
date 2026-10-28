#!/usr/bin/env python3
"""check_sessions.py - read-only check of one participant's Cowrie sessions (operator QA).

For every session of the given username: start time (UTC), session id, input lines
(counted with the frozen fv2 rules), commands (the lines without a final 'exit'), TTY status,
whether Cowrie has ended the session (ended=yes/no), and PASS when commands >= 5 and the
session's own TTY log exists. Run it on the Ubuntu VM after each session ends (after 'exit').
tty=no_log_closed with ended=yes means Cowrie ended the session without linking its TTY log
(seen after the 180 s idle auto-logout).
It reads the logs only; it never writes or deletes anything.

Usage:  python3 scripts/check_sessions.py p11
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fv2  # noqa: E402


def ts(v):
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--cowrie", default=os.path.expanduser("~/cowrie/var/log/cowrie"))
    ap.add_argument("--tty", default=os.path.expanduser("~/cowrie/var/lib/cowrie/tty"))
    a = ap.parse_args()
    S = {}
    for f in sorted(glob.glob(os.path.join(a.cowrie, "cowrie.json*"))):
        for line in open(f, errors="replace"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid, e = ev.get("session"), ev.get("eventid")
            if not sid:
                continue
            s = S.setdefault(sid, {"user": None, "start": None, "cmds": [], "closed": None,
                                   "ended": False})
            t = ts(ev.get("timestamp") or ev.get("ts"))
            if e == "cowrie.login.success" and s["user"] is None:
                s["user"], s["start"] = ev.get("username"), t
            elif e == "cowrie.command.input":
                s["cmds"].append((t, ev.get("input")))
            elif e == "cowrie.log.closed":
                s["closed"] = ev
            elif e == "cowrie.session.closed":
                s["ended"] = True
    rows = sorted(((sid, s) for sid, s in S.items() if s["user"] == a.user),
                  key=lambda x: x[1]["start"])
    for sid, s in rows:
        lines, _, _ = fv2.input_lines(s["cmds"])
        n_cmd = len(lines) - (1 if lines and lines[-1][1].strip() == "exit" else 0)
        _, status = fv2.tty_path_status(s["closed"], a.tty)
        when = datetime.fromtimestamp(s["start"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        verdict = "PASS" if n_cmd >= 5 and status == "ok" else "CHECK"
        ended = "yes" if s["ended"] else "no"
        print(f"{when}  session={sid}  lines={len(lines):>3}  commands={n_cmd:>3}  "
              f"tty={status:<14} ended={ended:<3}  {verdict}")
    print(f"{len(rows)} session(s) for {a.user}")


if __name__ == "__main__":
    main()
