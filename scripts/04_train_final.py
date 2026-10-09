#!/usr/bin/env python3
"""04_train_final.py — train final model on all interactive sessions + make report figures."""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import matthews_corrcoef, confusion_matrix

TIMING = ["command_count", "unique_commands", "command_diversity",
          "icd_mean", "icd_std", "session_duration"]
KEYSTROKE = ["has_tty", "single_char_ratio", "keystroke_gap_median", "backspaces"]
FULL = TIMING + KEYSTROKE
NAVY, RED, GREY = "#1C2B5E", "#C0392B", "#95A5A6"


def load(path):
    df = pd.read_csv(path)
    df = df[(df.labeled) & (df.population == "interactive")].copy()
    for c in FULL:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def logo_predict(df, feats, make):
    X, y, g = df[feats].values, df.is_human.values, df.group.values
    yt, yp = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make().fit(X[tr], y[tr])
        yt += list(y[te]); yp += list(clf.predict(X[te]))
    return np.array(yt), np.array(yp)


def fig_icd(df, out):
    plt.figure(figsize=(7, 4))
    h = df[df.is_human == 1].icd_mean
    plt.hist(h[h > 0], bins=20, alpha=0.6, color=NAVY, label="Human (p1-p10)")
    for prof, c in [("bot1", "#27AE60"), ("bot2", "#2980B9"), ("bot3", RED), ("bot4", "#8E44AD")]:
        v = df[df.group == prof].icd_mean
        if len(v):
            plt.scatter(v, np.full(len(v), -0.5), s=40, color=c, label=prof, zorder=3)
    plt.axvline(4.0, ls="--", color="black", lw=1.5)
    plt.text(4.1, plt.ylim()[1] * 0.9, "old 4s threshold", fontsize=9)
    plt.xlabel("Mean inter-command delay (s)"); plt.ylabel("Human sessions")
    plt.title("Timing overlaps: the slow bot (bot3) sits inside the human range")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{out}/fig_icd_distribution.png", dpi=160); plt.close()


def fig_keystroke(df, out):
    plt.figure(figsize=(6, 4))
    for lab, c, name in [(1, NAVY, "Human"), (0, RED, "Bot")]:
        v = df[df.is_human == lab].single_char_ratio
        plt.hist(v, bins=20, alpha=0.6, color=c, label=name)
    plt.xlabel("single_char_ratio (1 = types char-by-char, 0 = pastes whole line)")
    plt.ylabel("Sessions"); plt.title("Keystroke dynamics separate Human from Bot")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/fig_keystroke.png", dpi=160); plt.close()


def fig_importance(df, out):
    m = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42)).fit(df[FULL], df.is_human)
    imp = pd.Series(m.steps[-1][1].feature_importances_, FULL).sort_values()
    plt.figure(figsize=(7, 4))
    colors = [RED if f in KEYSTROKE else NAVY for f in imp.index]
    plt.barh(imp.index, imp.values, color=colors)
    plt.xlabel("Importance"); plt.title("Feature importance (navy = timing, red = keystroke)")
    plt.tight_layout(); plt.savefig(f"{out}/fig_feature_importance.png", dpi=160); plt.close()


def fig_confusion(df, out):
    yt, yp = logo_predict(df, FULL, lambda: make_pipeline(
        StandardScaler(), DecisionTreeClassifier(class_weight="balanced", random_state=42)))
    cm = confusion_matrix(yt, yp)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Bot", "Human"]); plt.yticks([0, 1], ["Bot", "Human"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion matrix (LOGO, full features)")
    plt.tight_layout(); plt.savefig(f"{out}/fig_confusion_matrix.png", dpi=160); plt.close()


def fig_mcc(df, out):
    models = {
        "RandomForest": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42)),
        "DecisionTree": lambda: make_pipeline(StandardScaler(), DecisionTreeClassifier(
            class_weight="balanced", random_state=42)),
        "SVM": lambda: make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=1000)),
    }
    names = list(models)
    tim = [matthews_corrcoef(*logo_predict(df, TIMING, models[n])) for n in names]
    ful = [matthews_corrcoef(*logo_predict(df, FULL, models[n])) for n in names]
    x = np.arange(len(names)); w = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, tim, w, color=GREY, label="Timing only")
    plt.bar(x + w / 2, ful, w, color=NAVY, label="Full (+ keystroke)")
    plt.xticks(x, names); plt.ylabel("MCC"); plt.ylim(0, 1.05)
    plt.title("Timing-only vs full feature set (leave-one-group-out)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/fig_mcc_comparison.png", dpi=160); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="../data/processed/labeled.csv")
    ap.add_argument("--out", default="../reports")
    ap.add_argument("--models", default="../models")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.models, exist_ok=True)
    df = load(a.inp)
    fig_icd(df, a.out); fig_keystroke(df, a.out); fig_importance(df, a.out)
    fig_confusion(df, a.out); fig_mcc(df, a.out)
    final = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42)).fit(df[FULL], df.is_human)
    joblib.dump(final, f"{a.models}/final_model.pkl")
    print(f"[+] 5 figures -> {a.out}/  and final_model.pkl -> {a.models}/")


if __name__ == "__main__":
    main()
