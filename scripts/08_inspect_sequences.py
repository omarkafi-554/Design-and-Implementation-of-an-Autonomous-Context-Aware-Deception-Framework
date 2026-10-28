#!/usr/bin/env python3
"""08_inspect_sequences.py - print the command sequences behind the primary
campaign_bot population so the proxy labels can be checked by hand."""
import glob, gzip, hashlib, json, sys
from collections import defaultdict
from datetime import datetime
import pandas as pd

LOGS = sorted(glob.glob("data/external/cow160/logs_by_day/*.json.gz"))
CSV = "reports/external/external_sessions.csv"
OUT = "reports/external/sequence_inspection.json"
PER_SEQ = 5          # example sessions kept per sequence


def ts(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


s = pd.read_csv(CSV)
prim = s[(s.proxy_label == "campaign_bot") & (s.command_count >= 2)]
summary = prim.groupby("seq_hash").agg(
    sessions=("session", "size"), distinct_ips=("src_ip", "nunique"),
    commands=("command_count", "first"), icd_mean_median=("icd_mean", "median"),
    duration_median=("session_duration", "median"),
    groups=("config_group", lambda x: sorted(int(v) for v in x.unique())),
).sort_values("sessions", ascending=False)
print(f"primary population: {len(prim):,} sessions, {len(summary)} sequences\n")

sample = {}
for h, g in prim.groupby("seq_hash"):
    for sid in g.sort_values("session").session.head(PER_SEQ):
        sample[sid] = h
cmds = defaultdict(list)
for f in LOGS:
    with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"cowrie.command.input"' not in line:
                continue
            ev = json.loads(line)
            sid = ev.get("session")
            if sid in sample:
                cmds[sid].append((ts(ev.get("ts") or ev.get("timestamp")),
                                  (ev.get("input") or "").strip()))
    print(f"[+] scanned {f}", file=sys.stderr)

report = {}
for h, row in summary.iterrows():
    sids = [sid for sid, hh in sample.items() if hh == h]
    entry = {k: (v if isinstance(v, list) else float(v)) for k, v in row.items()}
    entry["examples"] = []
    for sid in sids:
        c = sorted(cmds[sid], key=lambda x: x[0])
        seq = "\n".join(" ".join(x.split()) for _, x in c)
        entry["examples"].append({
            "session": sid,
            "hash_matches": hashlib.sha1(seq.encode()).hexdigest()[:16] == h,
            "gaps_s": [round(b[0] - a[0], 3) for a, b in zip(c, c[1:])],
            "commands": [x for _, x in c]})
    report[h] = entry
    print("=" * 78)
    print(f"seq {h} | sessions {int(row.sessions):,} | distinct IPs {int(row.distinct_ips):,} | "
          f"commands {int(row.commands)} | median μICD {row.icd_mean_median:.3f} s | "
          f"groups {row.groups}")
    ok = sum(e["hash_matches"] for e in entry["examples"])
    print(f"hash re-check: {ok}/{len(sids)} example sessions match")
    for i, cmd in enumerate(entry["examples"][0]["commands"], 1):
        extra = f" …[+{len(cmd) - 220} chars]" if len(cmd) > 220 else ""
        print(f"  {i:>2}. {cmd[:220]}{extra}")
    for e in entry["examples"]:
        print(f"  gaps (s) {e['session']}: {e['gaps_s']}")
json.dump(report, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"\n[+] full detail -> {OUT}")
