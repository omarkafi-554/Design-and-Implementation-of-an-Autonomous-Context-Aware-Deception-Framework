#!/usr/bin/env python3
"""14_posthoc_p3.py - POST-HOC (exploratory) diagnosis of the P3 result. Not pre-registered.
It does not change the confirmatory verdict: no model, threshold or label is changed.

It recomputes Model A's features on the scored COW160 campaign sessions (same pipeline as
12_external_v2.py, imported unchanged) and compares them with the lab training data, to show
WHY the frozen model calls these bots Human.

Usage
  python3 scripts/14_posthoc_p3.py --days "data/external/cow160/confirm/logs_by_day/*.json.gz"
Outputs  reports/v2/external_confirmatory/posthoc/
"""
import argparse
import glob
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

spec = importlib.util.spec_from_file_location("ext12", os.path.join(HERE, "12_external_v2.py"))
ext12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext12)


def q(s):
    s = pd.Series(s).dropna()
    if not len(s):
        return "-"
    return f"{s.quantile(.25):.2f} / {s.median():.2f} / {s.quantile(.75):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--model", default="models/model_A_v2.joblib")
    ap.add_argument("--lab", default="data/processed/lab_v2_features.csv")
    ap.add_argument("--exclude", default="reports/v2/external_confirmatory/exclusions.txt")
    a = ap.parse_args()
    out = "reports/v2/external_confirmatory/posthoc"
    os.makedirs(out, exist_ok=True)
    obj = joblib.load(a.model)
    clf, meta = obj["clf"], obj["meta"]
    k, feats = meta["k"], meta["features"]
    rep = {"note": "POST-HOC, exploratory; confirmatory verdict unchanged", "k": k,
           "features": feats, "algo": meta.get("algo")}
    est = clf.steps[-1][1] if hasattr(clf, "steps") else clf
    if hasattr(est, "feature_importances_"):
        rep["feature_importance"] = {f: round(float(v), 3)
                                     for f, v in zip(feats, est.feature_importances_)}

    # lab training rows (exactly what the frozen model was fitted on)
    lab = pd.read_csv(a.lab)
    lab = lab[(lab.split == "dev") & (lab.k == k)].dropna(subset=feats)
    lab["who"] = np.where(lab.is_human == 1, "LAB human", "LAB bot")

    # COW160 campaign sessions that reach line k (same pipeline as 12, unchanged)
    files = sorted(glob.glob(a.days))
    qc, df, lines_of = ext12.build(files)
    df = ext12.label(df, "template", 5)
    excl = set()
    if os.path.exists(a.exclude):
        excl = {ln.split()[0] for ln in open(a.exclude) if ln.strip() and not ln.startswith("#")}
    camp = df[(df.campaign == 1) & (df.n_lines >= k) & (~df.seq.isin(excl))].copy()
    rows = []
    for r in camp.itertuples():
        login = None if pd.isna(r.login_ts) else r.login_ts
        f = fv2.json_prefix_features(lines_of[r.session], login, k)
        rows.append({"session": r.session, "group": f"g{r.config_group}", "family": r.family,
                     "seq": r.seq, **{c: f[c] for c in feats}})
    cow = pd.DataFrame(rows).dropna(subset=feats)
    X = cow[feats].values.astype(float)
    cow["pred_human"] = clf.predict(X)
    cow["p_human"] = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    cow.to_csv(f"{out}/cow_features_at_k.csv", index=False)
    rep["cow_sessions_scored"] = int(len(cow))
    rep["cow_FPR_recomputed"] = round(float(cow.pred_human.mean()), 4)

    # 1) feature distributions: lab human vs lab bot vs COW (by prediction)
    cow["who"] = np.where(cow.pred_human == 1, "COW bot -> called Human", "COW bot -> called Bot")
    allr = pd.concat([lab[["who"] + feats], cow[["who"] + feats]])
    t1 = pd.DataFrame({f: allr.groupby("who")[f].apply(q) for f in feats})
    t1["n"] = allr.groupby("who").size()
    print("\n=== Feature quartiles (Q1 / median / Q3) at line k =", k, "===")
    print(t1.to_string())
    t1.to_csv(f"{out}/feature_quartiles.csv")

    # 2) by honeypot configuration group
    t2 = cow.groupby("group").agg(n=("pred_human", "size"), FPR=("pred_human", "mean"),
                                  **{f"med_{f}": (f, "median") for f in feats})
    print("\n=== By COW160 configuration group ===")
    print(t2.round(3).to_string())
    t2.round(4).to_csv(f"{out}/by_group.csv")

    # 3) family x group
    t3 = cow.groupby(["family", "group"]).agg(n=("pred_human", "size"),
                                              FPR=("pred_human", "mean"),
                                              **{f"med_{f}": (f, "median") for f in feats})
    t3 = t3[t3.n >= 5]
    print("\n=== Family x group (n >= 5) ===")
    print(t3.round(3).to_string())
    t3.round(4).to_csv(f"{out}/by_family_group.csv")

    # 4) lab bot classes at k (what the model learned 'bot' looks like)
    if "group" in lab.columns:
        t4 = lab.groupby(["who", "group"]).agg(n=(feats[0], "size"),
                                               **{f"med_{f}": (f, "median") for f in feats})
        print("\n=== Lab training groups (median features) ===")
        print(t4.round(3).to_string())
        t4.round(4).to_csv(f"{out}/lab_groups.csv")

    json.dump(rep, open(f"{out}/posthoc.json", "w"), indent=1)
    print("\n" + json.dumps(rep, indent=1))
    print(f"[+] {out}/  (post-hoc; nothing frozen was changed)")


if __name__ == "__main__":
    main()
