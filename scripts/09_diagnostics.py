#!/usr/bin/env python3
"""09_diagnostics.py - measure two Cowrie logging artefacts before any re-training.

A) Command substitution: Cowrie <= 2.6 logs every $(...) / `...` inside a line as an
   extra cowrie.command.input event (HoneyPotShell.run_subshell_command -> lineReceived).
B) TTY de-duplication: Cowrie names a TTY log by the SHA-256 of its INPUT bytes and
   deletes the log of any later session whose input is identical (log.closed duplicate=true).
Read-only: nothing in data/ or the Cowrie folders is modified.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

_NORM = re.compile(r"[\s'\"\\]")


def norm(text):
    return _NORM.sub("", text)


def ts(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


def collapse_subshell(cmds, max_gap=1.0):
    """cmds: [(t, text)] sorted by t. An event is a substitution artefact when the current
    top-level line contains $( or a backtick, the event text (no whitespace/quotes) is a
    substring of that line, and it arrives <= max_gap s after the previous event."""
    lines, derived, top, top_sub, prev = [], 0, "", False, None
    for t, text in cmds:
        n = norm(text)
        if top_sub and prev is not None and t - prev <= max_gap and n and n in top:
            derived += 1
        else:
            lines.append((t, text))
            top, top_sub = n, ("$(" in text or "`" in text)
        prev = t
    return lines, derived


def seq_hash(lines):
    seq = "\n".join(" ".join(x.split()) for _, x in lines)
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


def bins(series):
    edges = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 8, "4-8"),
             (9, 9, "9"), (10, 10**9, ">=10")]
    return {lbl: int(((series >= lo) & (series <= hi)).sum()) for lo, hi, lbl in edges}


def user_class(u):
    u = u or ""
    if re.fullmatch(r"p\d+", u):
        return "human"
    if re.fullmatch(r"bot\d+", u):
        return "bot"
    return "other"


# --------------------------------------------------------------------------- lab
def lab(json_dir, tty_dir):
    S = defaultdict(lambda: {"user": None, "cmds": [], "closed": []})
    bad = 0
    for f in sorted(glob.glob(os.path.join(json_dir, "cowrie.json*"))):
        for line in open(f, errors="replace"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            s = S[ev.get("session")]
            e = ev.get("eventid")
            if e in ("cowrie.login.success", "cowrie.login.failed") and s["user"] is None:
                s["user"] = ev.get("username")
            elif e == "cowrie.command.input":
                s["cmds"].append((ts(ev.get("timestamp") or ev.get("ts")),
                                  (ev.get("input") or "").strip()))
            elif e == "cowrie.log.closed":
                s["closed"].append(ev)
    out = {"malformed_lines": bad, "by_class": {}, "derived_examples": [], "bot_commands": None}
    rows = []
    for sid, s in S.items():
        cls = user_class(s["user"])
        if cls == "other" and not s["cmds"]:
            continue
        cmds = sorted(s["cmds"], key=lambda x: x[0])
        nonempty = [c for c in cmds if c[1]]
        lines, derived = collapse_subshell(nonempty)
        lc = s["closed"][-1] if s["closed"] else {}
        ttyfile = os.path.join(tty_dir, os.path.basename(lc.get("ttylog", "") or "x"))
        rows.append({"session": sid, "user": s["user"], "class": cls,
                     "events": len(cmds), "n_empty": len(cmds) - len(nonempty),
                     "lines": len(lines), "derived": derived,
                     "log_closed": int(bool(lc)), "duplicate": int(bool(lc.get("duplicate"))),
                     "shasum": lc.get("shasum"), "tty_file_exists": int(os.path.exists(ttyfile)),
                     "start": cmds[0][0] if cmds else None})
        if derived and len(out["derived_examples"]) < 10:
            out["derived_examples"].append({"session": sid, "class": cls, "commands": [c for _, c in cmds]})
        if cls == "bot" and out["bot_commands"] is None and cmds:
            out["bot_commands"] = [c for _, c in cmds]
    df = pd.DataFrame(rows)
    for cls, g in df.groupby("class"):
        out["by_class"][cls] = {
            "sessions": len(g), "with_commands": int((g.events > 0).sum()),
            "log_closed": int(g.log_closed.sum()), "duplicate_true": int(g.duplicate.sum()),
            "unique_shasums": int(g.shasum.dropna().nunique()),
            "tty_file_exists": int(g.tty_file_exists.sum()),
            "sessions_with_derived": int((g.derived > 0).sum()),
            "derived_events": int(g.derived.sum()), "empty_inputs": int(g["n_empty"].sum())}
    b = df[df["class"] == "bot"].sort_values("start")
    if len(b):
        last = b.shasum.dropna()
        out["bot_shasum_counts"] = Counter(last).most_common(5)
        out["bot_sessions_in_order"] = b[["session", "user", "duplicate", "shasum"]].astype(str).values.tolist()
    return out


# ---------------------------------------------------------------------- external
def flatten(ev, prefix=""):
    flat = {}
    for k, v in ev.items():
        if isinstance(v, dict):
            flat.update(flatten(v, f"{prefix}{k}_"))
        else:
            flat[f"{prefix}{k}"] = v
    return flat


def external(csv_path, logs, kappa):
    s = pd.read_csv(csv_path, usecols=["session", "src_ip", "config_group", "seq_hash",
                                        "proxy_label", "command_count"])
    keep = set(s.session)
    cmds, login, day, asn = defaultdict(list), {}, {}, defaultdict(set)
    asn_key, keys_seen = None, None
    for f in logs:
        d = os.path.basename(f).split(".")[0]
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                is_cmd = '"cowrie.command.input"' in line
                if not is_cmd and '"cowrie.login.success"' not in line:
                    continue
                ev = json.loads(line)
                sid = ev.get("session")
                if sid not in keep:
                    continue
                day.setdefault(sid, d)
                t = ts(ev.get("ts") or ev.get("timestamp"))
                if is_cmd:
                    cmds[sid].append((t, (ev.get("input") or "").strip()))
                else:
                    login.setdefault(sid, t)
                    if keys_seen is None:
                        keys_seen = sorted(flatten(ev).keys())
                        cands = [k for k in keys_seen if "asn" in k.lower() and "src" in k.lower()
                                 and ("number" in k.lower() or k.lower().endswith("asn"))]
                        asn_key = cands[0] if cands else None
                    if asn_key:
                        v = flatten(ev).get(asn_key)
                        if v is not None:
                            asn[ev.get("src_ip")].add(v)
        print(f"[+] scanned {f}", file=sys.stderr)

    rows = []
    for sid in s.session:
        c = sorted(cmds.get(sid, []), key=lambda x: x[0])
        nonempty = [x for x in c if x[1]]
        lines, derived = collapse_subshell(nonempty)
        lt = login.get(sid)
        rows.append({"session": sid, "day": day.get(sid), "events": len(c),
                     "n_empty": len(c) - len(nonempty), "lines": len(lines), "derived": derived,
                     "seq_hash_v2": seq_hash(lines) if lines else None,
                     "login_ts": lt,
                     "first_cmd_latency": round(lines[0][0] - lt, 4) if (lines and lt) else None,
                     "first_line": lines[0][1][:120] if lines else ""})
    v = s.merge(pd.DataFrame(rows), on="session")
    ips = v.groupby("seq_hash_v2").src_ip.nunique()
    v["seq_ip_count_v2"] = v.seq_hash_v2.map(ips)
    v["proxy_label_v2"] = np.where((v.seq_ip_count_v2 >= kappa) & (v.lines >= 1),
                                   "campaign_bot", "unattributed")
    camp = v[v.proxy_label_v2 == "campaign_bot"]
    old_primary = v[(v.proxy_label == "campaign_bot") & (v.command_count >= 2)]

    out = {
        "sessions": len(v), "sessions_without_events": int((v.events == 0).sum()),
        "sessions_with_derived": int((v.derived > 0).sum()),
        "derived_events": int(v.derived.sum()), "empty_inputs": int(v["n_empty"].sum()),
        "events_hist_old": bins(v.events), "lines_hist_v2": bins(v.lines),
        "old_primary_by_seq": {h: {"sessions": len(g), "lines_v2": bins(g.lines)}
                                for h, g in old_primary.groupby("seq_hash")},
        "campaign_v2": {"all_ge1": len(camp), "ge2": int((camp.lines >= 2).sum()),
                        "ge3": int((camp.lines >= 3).sum()),
                        "sequences_ge2": int(camp[camp.lines >= 2].seq_hash_v2.nunique())},
        "login_ts_available": round(float(v.login_ts.notna().mean()), 4),
        "first_cmd_latency_median_campaign": float(camp.first_cmd_latency.median()),
        "pseudonyms": int(v.src_ip.nunique()),
        "pseudonyms_on_ge2_days": int((v.groupby("src_ip").day.nunique() >= 2).sum()),
        "login_event_keys": keys_seen, "asn_key_used": asn_key,
        "pseudonyms_with_asn": len(asn),
        "pseudonyms_with_gt1_asn": sum(1 for x in asn.values() if len(x) > 1),
    }
    top = (camp[camp.lines >= 2].groupby("seq_hash_v2")
           .agg(sessions=("session", "size"), ips=("src_ip", "nunique"),
                lines=("lines", "first"), first_line=("first_line", "first"))
           .sort_values("sessions", ascending=False))
    out["campaign_v2_ge2_sequences"] = top.reset_index().to_dict("records")
    return out, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-json", default=os.path.expanduser("~/cowrie/var/log/cowrie"))
    ap.add_argument("--lab-tty", default=os.path.expanduser("~/cowrie/var/lib/cowrie/tty"))
    ap.add_argument("--ext-csv", default="reports/external/external_sessions.csv")
    ap.add_argument("--ext-logs", default="data/external/cow160/logs_by_day/*.json.gz")
    ap.add_argument("--kappa", type=int, default=5)
    ap.add_argument("--out", default="reports/diagnostics")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    res = {"lab": lab(a.lab_json, a.lab_tty)}
    print("\n=== LAB ===")
    print(json.dumps({k: res["lab"][k] for k in ("malformed_lines", "by_class")}, indent=1))
    print("bot sessions in time order [session, user, duplicate, shasum]:")
    for r in res["lab"].get("bot_sessions_in_order", []):
        print("  ", r[0], r[1], "dup=" + r[2], r[3][:12])
    print("one bot session's commands:", res["lab"]["bot_commands"])
    for ex in res["lab"]["derived_examples"]:
        print("derived example:", ex["class"], ex["commands"][:4])

    logs = sorted(glob.glob(a.ext_logs))
    ext, v = external(a.ext_csv, logs, a.kappa)
    res["external"] = ext
    v.drop(columns=["first_line"]).to_csv(f"{a.out}/external_lines_v2.csv", index=False)
    print("\n=== EXTERNAL ===")
    for k in ("sessions", "sessions_without_events", "sessions_with_derived", "derived_events",
              "empty_inputs", "events_hist_old", "lines_hist_v2", "campaign_v2",
              "login_ts_available", "first_cmd_latency_median_campaign", "pseudonyms",
              "pseudonyms_on_ge2_days", "asn_key_used", "pseudonyms_with_asn",
              "pseudonyms_with_gt1_asn"):
        print(f"{k}: {ext[k]}")
    print("old primary sequences -> lines after collapse:")
    for h, d in ext["old_primary_by_seq"].items():
        print(f"  {h}  n={d['sessions']:>6}  {d['lines_v2']}")
    print("campaign sequences with >= 2 lines after collapse:")
    for r in ext["campaign_v2_ge2_sequences"][:20]:
        print(f"  {r['seq_hash_v2']}  n={r['sessions']:>6}  ips={r['ips']:>5}  "
              f"lines={r['lines']:>3}  {r['first_line'][:70]}")
    print("login event keys:", ext["login_event_keys"])
    json.dump(res, open(f"{a.out}/diagnostics.json", "w"), indent=1, default=str)
    print(f"\n[+] {a.out}/diagnostics.json  and  {a.out}/external_lines_v2.csv")


if __name__ == "__main__":
    main()
