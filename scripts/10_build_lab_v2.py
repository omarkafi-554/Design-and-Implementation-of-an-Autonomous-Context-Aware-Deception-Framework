#!/usr/bin/env python3
"""10_build_lab_v2.py - lab dataset v2: provenance labels + prefix features at k = 3, 4, 5.

Labels (provenance = login username, independent of every feature)
    Human : p<number>            group = the participant (p1, p2, ...)
    Bot   : bot<anything>        group = the bot username (one username per bot class)
    other : everything else      excluded (attack tools, tests such as root)
Split
    'fresh'   participants listed in --fresh (held out until the model is frozen)
    'redteam' bot usernames listed in --redteam (adversarial test only, never trained on)
    'dev'     everyone else

Writes
    data/processed/lab_v2_sessions.csv   one row per labelled session
    data/processed/lab_v2_features.csv   one row per (session, k) that reached line k
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fv2  # noqa: E402


def ts(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


def label_of(user):
    u = user or ""
    if re.fullmatch(r"p\d+", u):
        return "Human"
    if re.fullmatch(r"bot[\w-]*", u):
        return "Bot"
    return None


def load(cowrie_dir):
    S = {}
    bad = 0
    for f in sorted(glob.glob(os.path.join(cowrie_dir, "cowrie.json*"))):
        for line in open(f, errors="replace"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            sid = ev.get("session")
            if sid is None:
                continue
            s = S.setdefault(sid, {"user": None, "login": None, "cmds": [], "closed": None})
            e = ev.get("eventid")
            t = ts(ev.get("timestamp") or ev.get("ts"))
            if e == "cowrie.login.success" and s["login"] is None:
                s["user"], s["login"] = ev.get("username"), t
            elif e == "cowrie.command.input":
                s["cmds"].append((t, ev.get("input")))
            elif e == "cowrie.log.closed":
                s["closed"] = ev
    return S, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cowrie", default=os.path.expanduser("~/cowrie/var/log/cowrie"))
    ap.add_argument("--tty", default=os.path.expanduser("~/cowrie/var/lib/cowrie/tty"))
    ap.add_argument("--fresh", default="", help="comma-separated held-out participants")
    ap.add_argument("--redteam", default="bot_mimic", help="comma-separated red-team bots")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    fresh = {x.strip() for x in a.fresh.split(",") if x.strip()}
    redteam = {x.strip() for x in a.redteam.split(",") if x.strip()}
    os.makedirs(a.out, exist_ok=True)

    S, bad = load(a.cowrie)
    srows, frows = [], []
    for sid, s in S.items():
        lab = label_of(s["user"])
        if lab is None or not s["cmds"]:
            continue
        lines, derived, empty = fv2.input_lines(s["cmds"])
        path, status = fv2.tty_path_status(s["closed"], a.tty)
        recs = fv2.read_tty(path) if path else None
        srows.append({"session": sid, "user": s["user"], "label": lab,
                      "is_human": int(lab == "Human"), "group": s["user"],
                      "split": ("fresh" if s["user"] in fresh else
                                "redteam" if s["user"] in redteam else "dev"),
                      "login_ts": s["login"], "n_events": len(s["cmds"]),
                      "n_lines": len(lines), "n_derived": derived, "n_empty": empty,
                      "tty_status": status, "mode": fv2.tty_mode(recs)})
        for k in fv2.K_CANDIDATES:
            jf = fv2.json_prefix_features(lines, s["login"], k)
            if jf is None:
                continue
            tf = fv2.tty_prefix_features(recs, jf["t_decision"], k)
            frows.append({"session": sid, "k": k, "is_human": int(lab == "Human"),
                          "group": s["user"], "split": srows[-1]["split"], **jf, **tf})
    sdf, fdf = pd.DataFrame(srows), pd.DataFrame(frows)
    sdf.to_csv(f"{a.out}/lab_v2_sessions.csv", index=False)
    fdf.to_csv(f"{a.out}/lab_v2_features.csv", index=False)

    print(f"[+] malformed lines skipped: {bad}")
    print(f"[+] labelled sessions with commands: {len(sdf)}  "
          f"(Human {int(sdf.is_human.sum())}, Bot {int((sdf.is_human == 0).sum())})")
    print(sdf.groupby(["label", "group"]).agg(
        sessions=("session", "size"), lines_median=("n_lines", "median"),
        derived=("n_derived", "sum"), tty_ok=("tty_status", lambda x: int((x == "ok").sum())),
        split=("split", "first")).to_string())
    print("\nTTY status:", sdf.tty_status.value_counts().to_dict())
    for k in fv2.K_CANDIDATES:
        g = fdf[fdf.k == k]
        print(f"k={k}: reached by {len(g)} sessions (Human {int(g.is_human.sum())}, "
              f"Bot {int((g.is_human == 0).sum())}); with TTY input: "
              f"{int((g.tty_input_writes > 0).sum())}")
    print(f"[+] {a.out}/lab_v2_sessions.csv  {a.out}/lab_v2_features.csv")


if __name__ == "__main__":
    main()
