#!/usr/bin/env python3
"""03_evaluate.py — honest Bot/Human evaluation, labelled by provenance, Leave-One-Group-Out."""
import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

TIMING = ["command_count", "unique_commands", "command_diversity",
          "icd_mean", "icd_std", "session_duration"]
KEYSTROKE = ["has_tty", "single_char_ratio", "keystroke_gap_median", "backspaces"]
FULL = TIMING + KEYSTROKE


def _models():
    return {
        "RandomForest": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42)),
        "DecisionTree": lambda: make_pipeline(StandardScaler(), DecisionTreeClassifier(
            class_weight="balanced", random_state=42)),
        "SVM": lambda: make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=1000)),
        "Baseline_majority": lambda: DummyClassifier(strategy="most_frequent"),
    }


def _logo(df, feats, make):
    X, y, g = df[feats].values, df.is_human.values, df.group.values
    yt, yp, held = [], [], []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make().fit(X[tr], y[tr])
        yt += list(y[te]); yp += list(clf.predict(X[te])); held += list(g[te])
    yt, yp = np.array(yt), np.array(yp)
    hp = pd.DataFrame({"g": held, "e": yt != yp})
    return {
        "mcc": round(matthews_corrcoef(yt, yp), 4),
        "balanced_accuracy": round(balanced_accuracy_score(yt, yp), 4),
        "errors_by_group": {k: int(v) for k, v in hp.groupby("g").e.sum().items() if v},
        "n": int(len(yt)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/labeled.csv")
    ap.add_argument("--out", default="reports/metrics.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df = pd.read_csv(a.inp)
    df = df[(df.labeled) & (df.population == "interactive")].copy()
    for c in FULL:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    out = {"dataset": {
        "interactive_sessions": int(len(df)),
        "human": int(df.is_human.sum()),
        "bot": int((df.is_human == 0).sum()),
        "groups": int(df.group.nunique()),
    }, "results": {}}
    print(f"Interactive sessions: {len(df)} "
          f"(Human={int(df.is_human.sum())}, Bot={int((df.is_human==0).sum())}, "
          f"groups={df.group.nunique()})\n")
    for fname, feats in [("TIMING_only", TIMING), ("FULL", FULL)]:
        out["results"][fname] = {}
        print(f"===== {fname} ({len(feats)} features) =====")
        for mname, make in _models().items():
            r = _logo(df, feats, make)
            out["results"][fname][mname] = r
            print(f"  {mname:18s} MCC={r['mcc']:+.4f}  balAcc={r['balanced_accuracy']:.4f}  "
                  f"errors={r['errors_by_group']}")
        print()
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[+] saved {a.out}")


if __name__ == "__main__":
    main()
