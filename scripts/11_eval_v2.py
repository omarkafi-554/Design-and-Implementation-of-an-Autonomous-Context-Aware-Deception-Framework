#!/usr/bin/env python3
"""11_eval_v2.py - pre-registered lab evaluation, model selection, freezing, fresh-human test.

select
  Population for choosing k: DEV sessions that reach line 5, so k = 3, 4, 5 are compared on
  the SAME sessions (only the number of lines the classifier may use changes).
  Model A = JSON features; Model B = JSON + TTY features (sessions with keyboard input).
  Candidates: k in {3,4,5} x {RF, LogReg, DT, SVM}; also a majority baseline and one fixed
  literature rule per model (no fitting).
  Validation: leave-one-group-out, group = participant (humans) / bot class (bots).
  Metrics (Human = positive): TP FP TN FN, TPR, FPR, specificity, precision, MCC with a
  95 % group-bootstrap CI (human and bot groups resampled separately, B = 2000, seed 42).
  Choice: highest LOGO MCC; ties -> smaller k -> RF, LogReg, DT, SVM.
  Selection bias: the whole choice is repeated inside every outer fold (nested LOGO); the
  nested MCC is the unbiased lab estimate of the procedure.
  The chosen configuration is refitted on ALL dev sessions that reach line k and frozen.
fresh
  Scores the frozen models on the held-out participants (split = fresh) and on the red-team
  bot (split = redteam). Run once, after 'select'; a model is never changed because of
  these results.
"""
import argparse
import hashlib
import json
import os
import sys

import joblib
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fv2  # noqa: E402

SEED, B = 42, 2000
ALGOS = {
    "RF": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED)),
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=1000)),
    "DT": lambda: make_pipeline(StandardScaler(), DecisionTreeClassifier(
        class_weight="balanced", random_state=SEED)),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
}
ORDER = list(ALGOS)
MODELS = {"A": fv2.JSON_FEATURES, "B": fv2.JSON_FEATURES + fv2.TTY_FEATURES}
RULES = {"A": ("Human if max implied typing rate <= 10 chars/s (about 120 WPM)",
               lambda d: (d.log_max_rate <= 1.0).astype(int).values),
         "B": ("Human if single-char ratio >= 0.5",
               lambda d: (d.single_char_ratio >= 0.5).astype(int).values)}
K_MAX = max(fv2.K_CANDIDATES)


def metrics(yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"n": int(len(yt)), "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "TPR": div(tp, tp + fn), "FPR": div(fp, fp + tn), "specificity": div(tn, fp + tn),
            "precision": div(tp, tp + fp), "MCC": round(float(matthews_corrcoef(yt, yp)), 4)}


def group_bootstrap(groups, yt, yp):
    rng = np.random.default_rng(SEED)
    d = pd.DataFrame({"g": groups, "yt": yt, "yp": yp})
    by_g = {g: (x.yt.values, x.yp.values) for g, x in d.groupby("g")}
    hum = sorted(g for g, (y, _) in by_g.items() if y[0] == 1)
    bot = sorted(g for g, (y, _) in by_g.items() if y[0] == 0)
    vals = []
    for _ in range(B):
        gs = list(rng.choice(hum, len(hum))) + list(rng.choice(bot, len(bot)))
        vals.append(matthews_corrcoef(np.concatenate([by_g[g][0] for g in gs]),
                                      np.concatenate([by_g[g][1] for g in gs])))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def logo_predict(d, feats, make):
    X, y, g = d[feats].values, d.is_human.values, d.group.values
    yp = np.zeros(len(d), dtype=int)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        yp[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return yp


def enough_groups(d):
    return d[d.is_human == 1].group.nunique() >= 2 and d[d.is_human == 0].group.nunique() >= 2


def choose(tables, feats):
    """tables: {k: frame}. Returns (k, algo, {(k, algo): MCC}) by LOGO MCC with tie-breaks."""
    scores = {}
    for k, d in tables.items():
        if not enough_groups(d):
            continue
        for algo in ORDER:
            scores[(k, algo)] = matthews_corrcoef(d.is_human.values,
                                                  logo_predict(d, feats, ALGOS[algo]))
    if not scores:
        return None, None, scores
    k, algo = max(scores, key=lambda ka: (round(scores[ka], 10), -ka[0], -ORDER.index(ka[1])))
    return k, algo, scores


def outer_fold(tables, feats, g):
    inner = {k: t[t.group != g].reset_index(drop=True) for k, t in tables.items()}
    k, algo, _ = choose(inner, feats)
    if k is None:
        return None
    test = tables[k][tables[k].group == g]
    clf = ALGOS[algo]().fit(inner[k][feats].values, inner[k].is_human.values)
    return f"k{k}-{algo}", list(test.is_human.values), list(clf.predict(test[feats].values))


def per_group(d, yp):
    x = d.assign(correct=(d.is_human.values == yp).astype(int))
    return {g: f"{int(v.correct.sum())}/{len(v)}" for g, v in x.groupby("group")}


def select(a):
    df = pd.read_csv(a.features)
    sha = hashlib.sha256(open(a.features, "rb").read()).hexdigest()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.models, exist_ok=True)
    dev = df[df.split == "dev"]
    report, table = {"features_csv_sha256": sha, "sklearn": sklearn.__version__}, []
    for model, feats in MODELS.items():
        r = report[model] = {}
        base = dev[dev.k == K_MAX]
        if model == "B":
            base = base[base.tty_input_writes > 0]
        pop = set(base.dropna(subset=feats).session)
        tables = {k: dev[(dev.k == k) & dev.session.isin(pop)].dropna(subset=feats)
                  .reset_index(drop=True) for k in fv2.K_CANDIDATES}
        r["selection_population"] = {"sessions": len(pop),
                                     "human": int(tables[K_MAX].is_human.sum()),
                                     "bot": int((tables[K_MAX].is_human == 0).sum()),
                                     "groups": int(tables[K_MAX].group.nunique())}
        for k, d in tables.items():
            if not enough_groups(d):
                r[f"k{k}"] = {"skipped": "fewer than 2 groups in a class"}
                continue
            e = r[f"k{k}"] = {}
            for name, make in {**ALGOS, "majority": lambda: DummyClassifier(
                    strategy="most_frequent")}.items():
                yp = logo_predict(d, feats, make)
                m = metrics(d.is_human.values, yp)
                m["MCC_95CI"] = group_bootstrap(d.group.values, d.is_human.values, yp)
                m["per_group"] = per_group(d, yp)
                e[name] = m
                table.append({"model": model, "k": k, "algo": name, **{x: m[x] for x in (
                    "n", "TP", "FP", "TN", "FN", "TPR", "FPR", "MCC")}, "MCC_95CI": m["MCC_95CI"]})
            desc, rule = RULES[model]
            yp = rule(d)
            e["rule"] = {"rule": desc, **metrics(d.is_human.values, yp),
                         "MCC_95CI": group_bootstrap(d.group.values, d.is_human.values, yp)}
            table.append({"model": model, "k": k, "algo": "rule", **{x: e["rule"][x] for x in (
                "n", "TP", "FP", "TN", "FN", "TPR", "FPR", "MCC")}, "MCC_95CI": e["rule"]["MCC_95CI"]})
        k, algo, _ = choose(tables, feats)
        if k is None:
            r["chosen"] = None
            continue
        # nested LOGO: repeat the whole choice without the held-out group
        groups = sorted(tables[K_MAX].group.unique())
        folds = Parallel(n_jobs=a.jobs)(delayed(outer_fold)(tables, feats, g) for g in groups)
        yt_all, yp_all, g_all, picks = [], [], [], {}
        for g, res in zip(groups, folds):
            if res is None:
                continue
            pick, yt, yp = res
            picks[g] = pick
            yt_all += yt
            yp_all += yp
            g_all += [g] * len(yt)
        r["nested_logo"] = {**metrics(np.array(yt_all), np.array(yp_all)),
                            "MCC_95CI": group_bootstrap(np.array(g_all), np.array(yt_all),
                                                        np.array(yp_all)),
                            "choice_per_outer_fold": picks}
        # freeze: refit on every dev session that reaches line k
        full = dev[dev.k == k]
        if model == "B":
            full = full[full.tty_input_writes > 0]
        full = full.dropna(subset=feats)
        clf = ALGOS[algo]().fit(full[feats].values, full.is_human.values)
        hd = full[full.is_human == 1].decision_delay
        meta = {"model": model, "k": int(k), "algo": algo, "features": feats,
                "n_train": int(len(full)), "n_train_human": int(full.is_human.sum()),
                "features_csv_sha256": sha, "sklearn": sklearn.__version__,
                "decision": "predict(): Human when P(Human) >= 0.5",
                "human_decision_delay_s": {"median": round(float(hd.median()), 2),
                                           "p75": round(float(hd.quantile(0.75)), 2)}}
        joblib.dump({"clf": clf, "meta": meta}, f"{a.models}/model_{model}_v2.joblib")
        r["chosen"] = meta
    json.dump(report, open(f"{a.out}/selection.json", "w"), indent=1, default=str)
    pd.DataFrame(table).to_csv(f"{a.out}/selection_table.csv", index=False)
    print(pd.DataFrame(table).to_string(index=False))
    for model in MODELS:
        c = report[model].get("chosen")
        n = report[model].get("nested_logo", {})
        print(f"\n[+] Model {model}: chosen {c and (c['k'], c['algo'])}  "
              f"nested LOGO MCC {n.get('MCC')} CI {n.get('MCC_95CI')}  "
              f"TP {n.get('TP')} FP {n.get('FP')} TN {n.get('TN')} FN {n.get('FN')}")
    print(f"[+] {a.out}/selection.json  {a.out}/selection_table.csv  models -> {a.models}/")


def fresh(a):
    df = pd.read_csv(a.features)
    sess = pd.read_csv(a.sessions)
    out = {}
    for model in MODELS:
        path = f"{a.models}/model_{model}_v2.joblib"
        if not os.path.exists(path):
            continue
        obj = joblib.load(path)
        meta, clf = obj["meta"], obj["clf"]
        d = df[(df.split == "fresh") & (df.k == meta["k"])]
        if model == "B":
            d = d[d.tty_input_writes > 0]
        d = d.dropna(subset=meta["features"])
        all_fresh = sess[(sess.split == "fresh") & (sess.n_lines >= 1)]
        yp = clf.predict(d[meta["features"]].values) if len(d) else np.array([], dtype=int)
        out[model] = {"k": meta["k"], "algo": meta["algo"],
                      "fresh_sessions_total": int(len(all_fresh)),
                      "reached_line_k": int(len(d)),
                      "detected_as_human": int(yp.sum()),
                      "TPR_at_decision": round(float(yp.mean()), 4) if len(d) else None,
                      "TPR_operational": round(float(yp.sum()) / len(all_fresh), 4)
                      if len(all_fresh) else None,
                      "per_participant": {g: f"{int(x.sum())}/{len(x)}" for g, x in
                                          pd.Series(yp, index=d.group.values).groupby(level=0)}}
        rt = df[(df.split == "redteam") & (df.k == meta["k"])]
        if model == "B":
            rt = rt[rt.tty_input_writes > 0]
        rt = rt.dropna(subset=meta["features"])
        if len(rt):
            yr = clf.predict(rt[meta["features"]].values)
            out[model]["redteam_bot"] = {"reached_line_k": int(len(rt)),
                                         "called_human_FP": int(yr.sum()),
                                         "TNR": round(1 - float(yr.mean()), 4)}
    json.dump(out, open(f"{a.out}/fresh_humans.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "fresh"])
    ap.add_argument("--features", default="data/processed/lab_v2_features.csv")
    ap.add_argument("--sessions", default="data/processed/lab_v2_sessions.csv")
    ap.add_argument("--out", default="reports/v2")
    ap.add_argument("--models", default="models")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel outer folds (nested LOGO)")
    a = ap.parse_args()
    select(a) if a.mode == "select" else fresh(a)


if __name__ == "__main__":
    main()
