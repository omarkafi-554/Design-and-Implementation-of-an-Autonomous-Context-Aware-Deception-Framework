#!/usr/bin/env python3
"""09b_template_labels.py - exploration days only: how many real bot sessions reach the
decision point under the exact-sequence and the template-sequence campaign labels.

Both labels use content only (never a classifier feature):
  exact    : identical whitespace-normalised line sequence from >= kappa pseudonyms
  template : identical sequence after fv2.template_line() from >= kappa pseudonyms
Family (cluster unit) = template of the first input line (fv2.family_hash).

Reads   reports/external/external_sessions.csv + the 5 exploration day files
Writes  reports/diagnostics/template_sessions.csv    one row per session (hashes, labels)
        reports/diagnostics/template_sequences.csv   every template campaign sequence that
                                                     reaches line 3, for manual inspection
        reports/diagnostics/template_summary.json
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fv2  # noqa: E402


def ts(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-csv", default="reports/external/external_sessions.csv")
    ap.add_argument("--ext-logs", default="data/external/cow160/logs_by_day/*.json.gz")
    ap.add_argument("--kappa", type=int, default=5)
    ap.add_argument("--out", default="reports/diagnostics")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    s = pd.read_csv(a.ext_csv, usecols=["session", "src_ip", "config_group"])
    keep = set(s.session)
    cmds, day = defaultdict(list), {}
    for f in sorted(glob.glob(a.ext_logs)):
        d = os.path.basename(f).split(".")[0]
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"cowrie.command.input"' not in line:
                    continue
                ev = json.loads(line)
                sid = ev.get("session")
                if sid in keep:
                    day.setdefault(sid, d)
                    cmds[sid].append((ts(ev.get("ts") or ev.get("timestamp")), ev.get("input")))
        print(f"[+] scanned {f}", file=sys.stderr)

    rows, texts, arte = [], {}, Counter()
    for sid in s.session:
        lines, det, _ = fv2.input_lines(cmds.get(sid, []), detail=True)
        arte.update(det)
        if not lines:
            continue
        th = fv2.sequence_hash(lines, template=True)
        rows.append({"session": sid, "day": day.get(sid), "n_lines": len(lines),
                     "exact_hash": fv2.sequence_hash(lines), "template_hash": th,
                     "family_hash": fv2.family_hash(lines)})
        if th not in texts:
            texts[th] = [fv2.template_line(x) for _, x in lines]
    v = s.merge(pd.DataFrame(rows), on="session")
    for lab in ("exact", "template"):
        n_ip = v.groupby(f"{lab}_hash").src_ip.nunique()
        v[f"{lab}_campaign"] = (v[f"{lab}_hash"].map(n_ip) >= a.kappa).astype(int)
    v.to_csv(f"{a.out}/template_sessions.csv", index=False)

    print(f"\nartefact events removed: {dict(arte)}")
    print(f"sessions with >= 1 input line: {len(v):,}")
    print(f"{'label':9s} {'campaign':>9s} | {'k':>2s} {'sessions':>9s} {'sequences>=5':>13s} {'families>=5':>12s}")
    summary = {"artefacts_removed": dict(arte)}
    for lab in ("exact", "template"):
        c = v[v[f"{lab}_campaign"] == 1]
        summary[lab] = {"campaign_sessions": int(len(c))}
        for k in (3, 4, 5):
            r = c[c.n_lines >= k]
            per_s = r.groupby(f"{lab}_hash").size()
            per_f = r.groupby("family_hash").size()
            summary[lab][f"k{k}"] = {"sessions": int(len(r)), "sequences_ge5": int((per_s >= 5).sum()),
                                     "families_ge5": int((per_f >= 5).sum())}
            print(f"{lab:9s} {len(c):>9,} | {k:>2d} {len(r):>9,} {int((per_s >= 5).sum()):>13d} "
                  f"{int((per_f >= 5).sum()):>12d}")
    un = v[(v.template_campaign == 0) & (v.n_lines >= 3)]
    print(f"unattributed after templating with >= 3 lines: {len(un):,} sessions")

    t = v[(v.template_campaign == 1) & (v.n_lines >= 3)]
    seq = (t.groupby("template_hash")
           .agg(sessions=("session", "size"), ips=("src_ip", "nunique"),
                exact_variants=("exact_hash", "nunique"), lines=("n_lines", "max"),
                days=("day", "nunique"), groups=("config_group", "nunique"),
                family_hash=("family_hash", "first"))
           .sort_values("sessions", ascending=False).reset_index())
    seq["template_text"] = seq.template_hash.map(lambda h: " || ".join(texts[h]))
    seq.to_csv(f"{a.out}/template_sequences.csv", index=False)
    fam = (seq.groupby("family_hash")
           .agg(sequences=("template_hash", "size"), sessions=("sessions", "sum"),
                first_line=("template_text", lambda x: x.iloc[0].split(" || ")[0][:90]))
           .sort_values("sessions", ascending=False))
    print(f"\ntemplate campaign families reaching line 3: {len(fam)}")
    for h, r in fam.iterrows():
        print(f"  {h} sequences={r.sequences:>3} sessions={r.sessions:>6,}  {r.first_line}")
    json.dump(summary, open(f"{a.out}/template_summary.json", "w"), indent=1)
    print(f"\n[+] {a.out}/template_sessions.csv  template_sequences.csv  template_summary.json")


if __name__ == "__main__":
    main()
