#!/usr/bin/env python3
"""07_external_analysis.py — report figures and the exploratory feature-set comparison
for the COW160x4 external validation (Chapter 2, Sections 2.2.8–2.2.9).

Reads   reports/external/external_sessions.csv  (written by 06_external_validation.py)
        data/processed/labeled.csv
Writes  reports/external/exploratory_ablation.json
        reports/external/fig_ext_command_count.png
        reports/external/fig_ext_icd_ecdf.png
        reports/external/fig_ext_ablation.png

The feature-set comparison is EXPLORATORY: it was not part of the pre-set protocol.
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HUM, BOT, EXT = "#2a78d6", "#eb6834", "#1baf7a"
MODELS = {
    "RF": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42)),
    "DT": lambda: make_pipeline(StandardScaler(), DecisionTreeClassifier(
        class_weight="balanced", random_state=42)),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000)),
}
SETS = {
    "TIMING_6": ["command_count", "unique_commands", "command_diversity",
                 "icd_mean", "icd_std", "session_duration"],
    "no_counts_3": ["icd_mean", "icd_std", "session_duration"],
    "no_counts_keystroke_6": ["icd_mean", "icd_std", "session_duration",
                              "single_char_ratio", "keystroke_gap_median", "backspaces"],
}


def logo_mcc(lab, feats, make):
    X, y, g = lab[feats].values, lab.is_human.values, lab.group.values
    yt, yp = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        yt += list(y[te]); yp += list(make().fit(X[tr], y[tr]).predict(X[te]))
    return round(float(matthews_corrcoef(yt, yp)), 4)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#888888"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="reports/external/external_sessions.csv")
    ap.add_argument("--lab", default="data/processed/labeled.csv")
    ap.add_argument("--out", default="reports/external")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lab = pd.read_csv(a.lab)
    lab = lab[(lab.labeled) & (lab.population == "interactive")].copy()
    for f in set(sum(SETS.values(), [])):
        lab[f] = pd.to_numeric(lab[f], errors="coerce").fillna(0.0)
    s = pd.read_csv(a.ext)
    camp = s[s.proxy_label == "campaign_bot"]
    pops = {"ge2": camp[camp.command_count >= 2], "ge4": camp[camp.command_count >= 4]}

    res = {}
    for sn, feats in SETS.items():
        for mn, make in MODELS.items():
            r = {"logo_mcc": logo_mcc(lab, feats, make)}
            if sn != "no_counts_keystroke_6":       # keystroke features absent externally
                m = make().fit(lab[feats].values, lab.is_human.values)
                for pn, p in pops.items():
                    r[f"false_human_{pn}"] = round(float(m.predict(p[feats].values).mean()), 4)
            res[f"{sn}|{mn}"] = r
            print(sn, mn, r)
    res["_note"] = "Exploratory, post hoc; not part of the pre-set protocol."
    json.dump(res, open(f"{a.out}/exploratory_ablation.json", "w"), indent=1)

    style()
    g2 = pops["ge2"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for v, lbl, col, ls in [(lab[lab.is_human == 1].icd_mean, f"Lab humans (n = {int(lab.is_human.sum())})", HUM, "-"),
                            (lab[lab.is_human == 0].icd_mean, f"Lab bots (n = {int((lab.is_human == 0).sum())})", BOT, "--"),
                            (g2.icd_mean, f"External campaign bots, ≥ 2 commands (n = {len(g2):,})", EXT, "-")]:
        x = np.sort(np.clip(v.values, 1e-4, None)); y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=col, lw=2, ls=ls, label=lbl)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 100); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean inter-command delay μICD (s, log scale)"); ax.set_ylabel("Cumulative share of sessions")
    ax.grid(color="#e6e6e6", lw=0.6); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_ext_icd_ecdf.png", dpi=220); plt.close(fig)

    bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 8, "4–8"), (9, 9, "9"), (10, 10**6, "≥ 10")]
    def shares(cc): return [((cc >= lo) & (cc <= hi)).mean() * 100 for lo, hi, _ in bins]
    groups = [(shares(lab[lab.is_human == 1].command_count), "Lab humans", HUM),
              (shares(lab[lab.is_human == 0].command_count), "Lab bots", BOT),
              (shares(camp.command_count), f"External campaign bots (n = {len(camp):,})", EXT)]
    fig, ax = plt.subplots(figsize=(7.6, 4.0)); x = np.arange(len(bins)); w = 0.26
    for i, (vals, lbl, col) in enumerate(groups):
        bars = ax.bar(x + (i - 1) * (w + 0.02), vals, w, color=col, label=lbl, edgecolor="white")
        for b, v in zip(bars, vals):
            if v >= 5:
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=7.5)
    ax.set_xticks(x, [b[2] for b in bins]); ax.set_xlabel("Commands per session"); ax.set_ylabel("Share of sessions (%)")
    ax.set_ylim(0, 108); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_ext_command_count.png", dpi=220); plt.close(fig)

    names = {"RF": "Random Forest", "DT": "Decision Tree", "SVM": "SVM", "LogReg": "Logistic Reg."}
    with_c = [res[f"TIMING_6|{m}"]["false_human_ge2"] * 100 for m in MODELS]
    without = [res[f"no_counts_3|{m}"]["false_human_ge2"] * 100 for m in MODELS]
    fig, ax = plt.subplots(figsize=(7.2, 4.0)); x = np.arange(len(MODELS)); w = 0.36
    b1 = ax.bar(x - w / 2 - 0.01, with_c, w, color=BOT, label="Timing + command counts (6 features, as specified)", edgecolor="white")
    b2 = ax.bar(x + w / 2 + 0.01, without, w, color=HUM, label="Timing only, no counts (3 features, exploratory)", edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.1f}%", ha="center", fontsize=8)
    ax.axhline(10, color="#555555", ls=":", lw=1.2)
    ax.text(len(MODELS) - 0.45, 11.5, "10 % acceptance limit", ha="right", fontsize=8, color="#555555")
    ax.set_xticks(x, [names[m] for m in MODELS]); ax.set_ylabel("False-Human rate (%)"); ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_ext_ablation.png", dpi=220); plt.close(fig)
    print(f"[+] figures and exploratory_ablation.json -> {a.out}/")


if __name__ == "__main__":
    main()
