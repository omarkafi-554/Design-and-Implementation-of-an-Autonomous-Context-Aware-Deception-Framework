#!/usr/bin/env python3
"""12_external_v2.py - pre-registered external test of the frozen Model A on COW160x4 days.

Two stages, so that the manual check of the proxy labels is blind to the model:
  inspect : quality checks, input lines, campaign labels. Writes every campaign sequence
            that reaches line k with >= 5 sessions to sequences_for_inspection.txt.
            No prediction is made.
  score   : the same pipeline, then Model A at its frozen k on the campaign sessions.
            Sequences listed in --exclude (judged human-operated at the inspect stage,
            one '<hash> <reason>' per line) are removed and reported.

Metrics (Human = positive; every scored session is a bot, so only FP and TN exist)
  conditional FPR = predicted Human / campaign sessions that reach line k
  operational FPR = predicted Human / all campaign sessions with >= 1 input line
  macro FPR       = mean per-family FPR over families with >= 5 sessions at line k
                    (family = template of the first input line; per-sequence also reported)
  95 % CIs        = cluster bootstrap over families (B = 2000, seed 42)
  split           = families seen on the exploration days vs new families
Acceptance (P3): conditional FPR <= 10 % and family macro FPR <= 10 %.

Usage
  python3 scripts/12_external_v2.py inspect --days "data/external/cow160/confirm/*.json.gz"
  python3 scripts/12_external_v2.py score   --days "data/external/cow160/confirm/*.json.gz" \
          --exclude reports/v2/external_confirmatory/exclusions.txt
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fv2  # noqa: E402

SEED, B, MIN_SEQ, LIMIT = 42, 2000, 5, 0.10
KEEP = {"cowrie.session.connect", "cowrie.session.closed", "cowrie.command.input",
        "cowrie.login.success", "cowrie.login.failed", "cowrie.client.size",
        "cowrie.log.closed"}


def ts(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


def events(files, qc):
    for f in files:
        day = os.path.basename(f).split(".")[0]
        with gzip.open(f, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                qc["lines"] += 1
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    qc["malformed_lines"] += 1
                    continue
                if "eventid" not in ev or not ev.get("session"):
                    qc["missing_required"] += 1
                    continue
                yield day, ev
        print(f"[+] read {f}", file=sys.stderr)


def build(files):
    """Quality counters (same definitions as 06_external_validation.py) + one row per
    complete session (connect and closed seen) with at least one input line."""
    qc, seen = Counter(), set()
    connect, closed, login = set(), set(), {}
    S = defaultdict(lambda: {"cmds": [], "ip": None, "group": None, "day": None})
    for day, ev in events(files, qc):
        e, sid = ev["eventid"], ev["session"]
        if e not in KEEP:
            continue
        key = (sid, e, ev.get("ts") or ev.get("timestamp"), ev.get("input"))
        h = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
        if h in seen:
            qc["duplicate_events"] += 1
            continue
        seen.add(h)
        if e == "cowrie.session.connect":
            connect.add(sid)
        elif e == "cowrie.session.closed":
            closed.add(sid)
        elif e == "cowrie.command.input":
            s = S[sid]
            s["cmds"].append((ts(ev.get("ts") or ev.get("timestamp")), ev.get("input")))
            s["ip"] = s["ip"] or ev.get("src_ip")
            s["group"] = s["group"] if s["group"] is not None else ev.get("group")
            s["day"] = s["day"] or day
        elif e == "cowrie.login.success":
            t = ts(ev.get("ts") or ev.get("timestamp"))
            login[sid] = min(login.get(sid, t), t)
    del seen
    complete = connect & closed
    qc["sessions_seen"] = len(connect | closed | set(S))
    qc["sessions_complete"] = len(complete)
    qc["sessions_with_commands"] = len(S)
    qc["cmd_sessions_complete"] = sum(1 for sid in S if sid in complete)
    rows, lines_of = [], {}
    for sid, s in S.items():
        if not s["cmds"] or sid not in complete:
            continue
        lines, det, empty = fv2.input_lines(s["cmds"], detail=True)
        qc["substitution_events_removed"] += det["substitution"]
        qc["passwd_echo_events_removed"] += det["passwd_echo"]
        qc["empty_inputs"] += empty
        if not lines:
            qc["sessions_only_empty_input"] += 1
            continue
        lines_of[sid] = lines
        rows.append({"session": sid, "src_ip": s["ip"], "config_group": s["group"],
                     "day": s["day"], "login_ts": login.get(sid), "n_lines": len(lines),
                     "exact_hash": fv2.sequence_hash(lines),
                     "template_hash": fv2.sequence_hash(lines, template=True),
                     "family_hash": fv2.family_hash(lines)})
    return dict(qc), pd.DataFrame(rows), lines_of


def label(df, how, kappa):
    col = f"{how}_hash"
    n_ip = df.groupby(col).src_ip.nunique()
    df["seq"] = df[col]
    df["family"] = df["family_hash"]
    df["campaign"] = (df[col].map(n_ip) >= kappa).astype(int)
    return df


def cluster_boot(per):
    """per: frame with n (sessions at line k) and fp per cluster."""
    rng = np.random.default_rng(SEED)
    n, fp = per.n.values, per.fp.values
    pooled, macro = [], []
    for _ in range(B):
        i = rng.integers(0, len(n), len(n))
        pooled.append(fp[i].sum() / max(n[i].sum(), 1))
        big = n[i] >= MIN_SEQ
        if big.any():
            macro.append(float(np.mean(fp[i][big] / n[i][big])))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] \
        if v else None  # noqa: E731
    return ci(pooled), ci(macro)


def summarize(d, unit="family"):
    """d: campaign sessions with columns family, seq, reach (0/1), pred (1 = Human)."""
    r = d[d.reach == 1]
    per = r.groupby(unit).agg(n=("pred", "size"), fp=("pred", "sum")).reset_index()
    per["fpr"] = per.fp / per.n
    big, small = per[per.n >= MIN_SEQ], per[per.n < MIN_SEQ]
    ci_pooled, ci_macro = cluster_boot(per) if len(per) else (None, None)
    out = {"unit": unit, "campaign_sessions_ge1_line": int(len(d)), "reached_line_k": int(len(r)),
           "predicted_human_FP": int(r.pred.sum()), "predicted_bot_TN": int((r.pred == 0).sum()),
           "conditional_FPR": round(float(r.pred.mean()), 4) if len(r) else None,
           "conditional_FPR_95CI": ci_pooled,
           "operational_FPR": round(float(r.pred.sum()) / len(d), 4) if len(d) else None,
           "clusters_at_k": int(len(per)), "clusters_ge5": int(len(big)),
           "macro_FPR_ge5": round(float(big.fpr.mean()), 4) if len(big) else None,
           "macro_FPR_95CI": ci_macro,
           "small_clusters_pooled_FPR": round(float(small.fp.sum() / small.n.sum()), 4)
           if len(small) else None}
    return out, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inspect", "score"])
    ap.add_argument("--days", required=True, help="glob of day files (*.json.gz)")
    ap.add_argument("--model", default="models/model_A_v2.joblib")
    ap.add_argument("--label", choices=["template", "exact"], default="template")
    ap.add_argument("--kappa", type=int, default=5)
    ap.add_argument("--seen", default="reports/diagnostics/template_sessions.csv",
                    help="exploration-day sessions (09b output) for the seen/new split")
    ap.add_argument("--exclude", default="", help="file: '<sequence hash> <reason>' per line")
    ap.add_argument("--tag", default="confirmatory")
    a = ap.parse_args()
    out = f"reports/v2/external_{a.tag}"
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(a.days))
    if not files:
        sys.exit("[-] no day files match --days")
    obj = joblib.load(a.model)
    clf, meta = obj["clf"], obj["meta"]
    k, feats = meta["k"], meta["features"]

    qc, df, lines_of = build(files)
    df = label(df, a.label, a.kappa)
    camp = df[df.campaign == 1].copy()
    camp["reach"] = (camp.n_lines >= k).astype(int)
    res = {"stage": a.stage, "tag": a.tag, "files": [os.path.basename(f) for f in files],
           "label": a.label, "kappa": a.kappa, "k": k, "model": meta, "quality": qc,
           "sessions_ge1_line": int(len(df)), "campaign_sessions": int(len(camp))}

    if a.stage == "inspect":
        per = camp[camp.reach == 1].groupby("seq").agg(
            sessions=("session", "size"), ips=("src_ip", "nunique"),
            exact_variants=("exact_hash", "nunique"), lines=("n_lines", "max"),
            family=("family", "first"), first=("session", "first")
        ).sort_values(["family", "sessions"], ascending=[True, False])
        per = per[per.sessions >= MIN_SEQ]
        with open(f"{out}/sequences_for_inspection.txt", "w") as fh:
            for h, r in per.iterrows():
                fh.write(f"=== family {r.family}  sequence {h}  sessions={r.sessions}  "
                         f"ips={r.ips}  exact_variants={r.exact_variants}  lines={r.lines}\n")
                lines = lines_of[r["first"]]
                gaps = [round(b[0] - x[0], 3) for x, b in zip(lines, lines[1:])]
                for i, (_, txt) in enumerate(lines[:20], 1):
                    fh.write(f"  {i:>2}. {txt[:200]}\n")
                fh.write(f"  gaps (s): {gaps[:19]}\n\n")
        res["sequences_for_inspection"] = int(len(per))
        json.dump(res, open(f"{out}/inspect.json", "w"), indent=1, default=str)
        print(json.dumps({x: res[x] for x in ("quality", "sessions_ge1_line", "campaign_sessions",
                                               "sequences_for_inspection")}, indent=1, default=str))
        print(f"[+] {out}/sequences_for_inspection.txt  (no predictions were made)")
        return

    excluded = {}
    if a.exclude:
        for line in open(a.exclude):
            if line.strip() and not line.startswith("#"):
                h, _, why = line.strip().partition(" ")
                excluded[h] = why.strip()
    res["excluded_sequences"] = {h: {"reason": w, "sessions": int((camp.seq == h).sum())}
                                 for h, w in excluded.items()}
    camp = camp[~camp.seq.isin(excluded)].copy()
    camp["pred"] = 0
    reached = camp.reach == 1
    X = []
    for sid, login in zip(camp.loc[reached, "session"], camp.loc[reached, "login_ts"]):
        f = fv2.json_prefix_features(lines_of[sid], None if pd.isna(login) else login, k)
        X.append([f[c] for c in feats])
    X = np.array(X, dtype=float)
    ok = ~np.isnan(X).any(axis=1) if len(X) else np.array([], dtype=bool)
    res["reached_without_login_ts_excluded"] = int((~ok).sum())
    idx = camp.index[reached][ok]
    if len(idx):
        camp.loc[idx, "pred"] = clf.predict(X[ok])
    camp.loc[camp.index[reached][~ok], "reach"] = 0
    overall, per = summarize(camp, "family")
    res["overall"] = overall
    res["per_sequence_secondary"] = summarize(camp, "seq")[0]
    seen_fam, seen_seq = set(), set()
    if a.seen and os.path.exists(a.seen):
        s = pd.read_csv(a.seen)
        e = s[s[f"{a.label}_campaign"] == 1]
        seen_fam, seen_seq = set(e.family_hash), set(e[f"{a.label}_hash"])
    camp["seen_family"] = camp.family.isin(seen_fam).astype(int)
    camp["seen_sequence"] = camp.seq.isin(seen_seq).astype(int)
    res["seen_families"] = summarize(camp[camp.seen_family == 1])[0]
    res["new_families"] = summarize(camp[camp.seen_family == 0])[0]
    res["new_sequences_in_seen_families"] = summarize(
        camp[(camp.seen_family == 1) & (camp.seen_sequence == 0)], "seq")[0]
    res["by_config_group"] = {f"g{g}": summarize(x)[0]["conditional_FPR"]
                              for g, x in camp.groupby("config_group")}
    p3 = (overall["conditional_FPR"] is not None and overall["conditional_FPR"] <= LIMIT
          and overall["macro_FPR_ge5"] is not None and overall["macro_FPR_ge5"] <= LIMIT)
    res["P3_pass"] = bool(p3)
    per["seen"] = per.family.isin(seen_fam).astype(int)
    per.sort_values("n", ascending=False).to_csv(f"{out}/per_family.csv", index=False)
    summarize(camp, "seq")[1].sort_values("n", ascending=False).to_csv(
        f"{out}/per_sequence.csv", index=False)
    camp[["session", "day", "config_group", "family", "seq", "n_lines", "reach", "pred",
          "seen_family", "seen_sequence"]].to_csv(f"{out}/campaign_sessions.csv", index=False)
    json.dump(res, open(f"{out}/score.json", "w"), indent=1, default=str)
    print(json.dumps({x: res[x] for x in ("quality", "overall", "per_sequence_secondary",
                                           "seen_families", "new_families", "by_config_group",
                                           "excluded_sequences", "P3_pass")},
                     indent=1, default=str))
    print(f"[+] {out}/score.json  per_family.csv  per_sequence.csv  campaign_sessions.csv")


if __name__ == "__main__":
    main()
