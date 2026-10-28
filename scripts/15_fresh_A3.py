#!/usr/bin/env python3
"""15_fresh_A3.py - Amendment 1, criterion C1: frozen Model A3 on the fresh participants.

Population = the same fresh sessions that 11_eval_v2.py fresh scores for Model A
(split == "fresh" in lab_v2_features.csv at A3's k). A3's features are rebuilt from the raw
Cowrie log exactly as in 13_external_human.py fit: fv2.input_lines -> json_prefix_features(lines,
None, k). Nothing is refitted. Also reports A3 on the red-team sessions (secondary).

  TPR at decision = predicted Human / fresh sessions that reach line k
  TPR operational = predicted Human / fresh sessions with >= 1 input line
  95 % CI         = bootstrap over participants (B = 2000, seed 42, percentile)
  C1: TPR at decision >= 0.80

Usage (after 10_build_lab_v2.py --fresh p16,p17,p18,p19,p20):
  python3 scripts/15_fresh_A3.py
"""
import argparse
import importlib.util
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fv2  # noqa: E402

SEED, B, C1_LIMIT = 42, 2000, 0.80
A3 = ["log_gap_median", "gap_cv", "log_max_rate"]


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(HERE, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def boot_ci(groups, yp):
    rng = np.random.default_rng(SEED)
    by = {g: yp[groups == g] for g in np.unique(groups)}
    keys = sorted(by)
    vals = []
    for _ in range(B):
        pick = rng.choice(keys, len(keys))
        v = np.concatenate([by[g] for g in pick])
        vals.append(v.mean())
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def predict_split(df, S, clf, k, split):
    d = df[(df.split == split) & (df.k == k)].dropna(subset=fv2.JSON_FEATURES).reset_index(drop=True)
    rows = []
    for sid in d.session:
        lines, _, _ = fv2.input_lines(S[sid]["cmds"])
        f = fv2.json_prefix_features(lines, None, k)
        rows.append([f[c] for c in A3])
    X = np.array(rows, dtype=float).reshape(-1, len(A3))
    for i, c in enumerate(("log_gap_median", "gap_cv")):
        if len(d) and not np.allclose(d[c].values, X[:, i], atol=1e-9):
            sys.exit(f"[-] rebuilt {c} differs from the features CSV - stop")
    yp = clf.predict(X) if len(d) else np.array([], dtype=int)
    return d, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/lab_v2_features.csv")
    ap.add_argument("--sessions", default="data/processed/lab_v2_sessions.csv")
    ap.add_argument("--cowrie", default=os.path.expanduser("~/cowrie/var/log/cowrie"))
    ap.add_argument("--model-a3", dest="model_a3", default="models/model_A3_v2.joblib")
    ap.add_argument("--out", default="reports/v2")
    a = ap.parse_args()
    obj = joblib.load(a.model_a3)
    clf, meta = obj["clf"], obj["meta"]
    k = int(meta["k"])
    df, sess = pd.read_csv(a.features), pd.read_csv(a.sessions)
    S, _ = module("10_build_lab_v2.py").load(a.cowrie)

    d, yp = predict_split(df, S, clf, k, "fresh")
    if len(d) and (d.is_human != 1).any():
        sys.exit("[-] a fresh session is not labelled Human - stop")
    all_fresh = sess[(sess.split == "fresh") & (sess.n_lines >= 1)]
    tpr = round(float(yp.mean()), 4) if len(d) else None
    res = {"criterion": "C1 (PREREG_AMENDMENT_1)", "model": "A3", "k": k, "algo": meta["algo"],
           "fresh_participants": sorted(set(all_fresh.user)),
           "fresh_sessions_total": int(len(all_fresh)), "reached_line_k": int(len(d)),
           "detected_as_human": int(yp.sum()), "TPR_at_decision": tpr,
           "TPR_95CI_participant_bootstrap": boot_ci(d.group.values, yp) if len(d) else None,
           "TPR_operational": round(float(yp.sum()) / len(all_fresh), 4) if len(all_fresh) else None,
           "per_participant": {g: f"{int(x.sum())}/{len(x)}" for g, x in
                               pd.Series(yp, index=d.group.values).groupby(level=0)},
           "C1_pass": bool(tpr is not None and tpr >= C1_LIMIT)}
    rt, yr = predict_split(df, S, clf, k, "redteam")
    if len(rt):
        res["redteam_bot_secondary"] = {"reached_line_k": int(len(rt)), "called_human_FP": int(yr.sum()),
                                        "TNR": round(1 - float(yr.mean()), 4)}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/fresh_A3.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"[+] {a.out}/fresh_A3.json")


if __name__ == "__main__":
    main()
