#!/usr/bin/env python3
"""16_ensemble_v2.py - PREREG_AMENDMENT_2: Model E, a hard-voting ensemble on the A3 features.

Why (supervisor, 19 Oct 2026): instead of one decision tree, use a vote of three trained
models, LightGBM among them, in hard mode.

Model E
  Features   the three A3 features (log_gap_median, gap_cv, log_max_rate over lines 2..k),
             rebuilt from the raw Cowrie log exactly as 13_external_human.py fit does, on the
             same k and the same dev sessions as the frozen Model A3.
  Members    LightGBM, LogReg and SVM (RBF): three different model families, so their errors
             are less correlated than those of three tree models. Configurations: members().
  Imbalance  class_weight="balanced" in every member (as in 11_eval_v2.py). SMOTE is only a
             comparison (VOTE_SMOTE), applied inside the training folds.
  Voting     hard: the majority of the three labels (3 voters, 2 classes -> no ties). There is
             no P(Human); the share of member votes for Human (0, 1/3, 2/3, 1) is the score.

Stages
  eval          lab LOGO on the A3 population (group = participant / bot class) for RF, LogReg,
                DT, SVM (11_eval_v2.py configurations), LightGBM, VOTE (= Model E), VOTE_SMOTE,
                the majority class and the rate rule. Metrics, group-bootstrap CI and per-group
                results as in 11_eval_v2.py, plus the pairwise diversity of the three members.
                Checks: the rebuilt rows give back the frozen A3 when refitted (else stop), and
                A3's algorithm reproduces A3's lab MCC (reported). Every configuration is fixed
                in the amendment, so nothing is chosen from these results.
  fit           refit VOTE on the whole A3 population -> models/model_E_v2.joblib
  score-humans  frozen E next to frozen A3: fresh participants and red-team bot (population of
                15_fresh_A3.py) and KYPO (as 13_external_human.py score).
                E2: KYPO human recall >= 0.80.
  score-bots    frozen E next to frozen A3 on COW160x4 days (pipeline, template label and
                exclusions of 12_external_v2.py).
                E1: conditional and family-macro false-human rate <= 10 %.

Usage (from ~/deception-framework)
  python3 scripts/16_ensemble_v2.py eval
  python3 scripts/16_ensemble_v2.py fit
  python3 scripts/16_ensemble_v2.py score-humans
  python3 scripts/16_ensemble_v2.py score-bots --days "data/external/cow160/confirm3/logs_by_day/*.json.gz" \
          --exclude reports/v2/external_confirmatory3/exclusions.txt --tag confirmatory3
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fv2  # noqa: E402

try:
    import lightgbm
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    sys.exit("[-] lightgbm is missing: see the install step of PREREG_AMENDMENT_2 (lightgbm==4.7.0)")

SEED, HUMAN = 42, 1
A3 = ["log_gap_median", "gap_cv", "log_max_rate"]
MEMBERS = ["LightGBM", "LogReg", "SVM"]
E1_LIMIT, E2_LIMIT = 0.10, 0.80
OUT = "reports/v2/ensemble"


def module(name):
    """Import a frozen sibling script (file names start with a digit) without running it."""
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(HERE, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- Model E (fixed in the amendment)
def lgbm(cw="balanced"):
    """Small-data LightGBM: at most 7 leaves, depth 3, >= 10 sessions per leaf."""
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, max_depth=3,
                          min_child_samples=10, reg_lambda=1.0, class_weight=cw,
                          random_state=SEED, deterministic=True, force_row_wise=True,
                          n_jobs=1, verbose=-1)


def members(cw="balanced", scale=True):
    """The three voters. LogReg and SVM are the 11_eval_v2.py configurations."""
    est = [("LightGBM", lgbm(cw)),
           ("LogReg", LogisticRegression(class_weight=cw, max_iter=1000)),
           ("SVM", SVC(class_weight=cw))]
    return [(n, make_pipeline(StandardScaler(), e)) if scale else (n, e) for n, e in est]


def make_vote():
    """Model E."""
    return VotingClassifier(members(), voting="hard")


def make_vote_smote():
    """Comparison only: the same members without class weights; SMOTE inside fit (training
    data only - imblearn's Pipeline never resamples in predict)."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    return Pipeline([("scale", StandardScaler()), ("smote", SMOTE(random_state=SEED)),
                     ("vote", VotingClassifier(members(cw=None, scale=False), voting="hard"))])


def make_lgbm():
    return make_pipeline(StandardScaler(), lgbm())


def vote_share(clf, X):
    """Share of the member votes that say Human: 0, 1/3, 2/3 or 1 (hard voting has no P)."""
    labels = clf.classes_[clf.transform(np.asarray(X, dtype=float))]
    return (labels == HUMAN).mean(axis=1)


def share_counts(s):
    s = np.round(np.asarray(s, dtype=float) * 3).astype(int)
    return {f"{v}/3": int((s == v).sum()) for v in range(4)}


def versions():
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
           "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__}
    try:
        import imblearn
        out["imbalanced-learn"] = imblearn.__version__
    except ImportError:
        out["imbalanced-learn"] = None
    return out


# ---------------------------------------------------------------- lab population (as 13 fit)
def same_as_frozen_a3(d, meta_a3, clf_a3):
    """The fresh-split rebuild of lab_v2_features.csv changes its SHA-256, so identity with the
    data A3 was trained on is checked directly: refitting A3's frozen configuration on the
    rebuilt rows must give the frozen model back (same predictions on the rows and on 5,000
    random points spanning the feature ranges)."""
    m11 = module("11_eval_v2.py")
    X, y = d[A3].values, d.is_human.values
    refit = m11.ALGOS[meta_a3["algo"]]().fit(X, y)
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = 0.1 * (hi - lo) + 1e-6
    cloud = np.random.default_rng(0).uniform(lo - pad, hi + pad, size=(5000, X.shape[1]))
    return bool((refit.predict(X) == clf_a3.predict(X)).all()
                and (refit.predict(cloud) == clf_a3.predict(cloud)).all())


def a3_population(a):
    """Dev sessions at A3's k (the sessions A and A3 were refitted on) with the A3 features
    rebuilt from the raw Cowrie log. Stops when anything differs from the frozen models."""
    m10 = module("10_build_lab_v2.py")
    meta_a = joblib.load(a.model_a)["meta"]
    a3 = joblib.load(a.model_a3)
    meta_a3 = a3["meta"]
    k = int(meta_a3["k"])
    if k != int(meta_a["k"]) or list(meta_a3["features"]) != A3:
        sys.exit("[-] Model A3 does not have the k of Model A and the three A3 features - stop")
    sha = sha256(a.features)
    feat = pd.read_csv(a.features)
    pop = feat[(feat.split == "dev") & (feat.k == k)].dropna(subset=fv2.JSON_FEATURES)
    pop = pop.reset_index(drop=True)
    if (len(pop) != int(meta_a.get("n_train", -1)) or len(pop) != int(meta_a3.get("sessions", -1))
            or int(pop.is_human.sum()) != int(meta_a.get("n_train_human", -1))):
        sys.exit("[-] population differs from the sessions Models A and A3 were refitted on - stop")
    S, _ = m10.load(a.cowrie)
    missing = [s for s in pop.session if s not in S]
    if missing:
        sys.exit(f"[-] {len(missing)} sessions are missing from the Cowrie log (first {missing[0]}) - stop")
    rows = []
    for sid in pop.session:
        lines, _, _ = fv2.input_lines(S[sid]["cmds"])
        f = fv2.json_prefix_features(lines, None, k)
        rows.append([f[c] for c in A3])
    X = np.array(rows, dtype=float).reshape(-1, len(A3))
    for i, c in enumerate(("log_gap_median", "gap_cv")):
        if not np.allclose(pop[c].values, X[:, i], atol=1e-9):
            sys.exit(f"[-] rebuilt {c} differs from the features CSV - stop")
    d = pop[["session", "group", "is_human"]].copy()
    for i, c in enumerate(A3):
        d[c] = X[:, i]
    if not same_as_frozen_a3(d, meta_a3, a3["clf"]):
        sys.exit("[-] refitting A3 on these rows does not give the frozen A3 - data differ - stop")
    checks = {"features_csv_sha256": sha,
              "csv_sha256_same_as_at_A3_freeze": sha == meta_a3.get("features_csv_sha256"),
              "population_same_as_frozen_A3": True}
    return d, k, checks, meta_a3


def diversity(y, preds):
    """Pairwise diversity of the voters on the LOGO predictions (Kuncheva & Whitaker 2003)."""
    out, names = {}, list(preds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, cb = preds[a] == y, preds[b] == y
            n11, n00 = int((ca & cb).sum()), int((~ca & ~cb).sum())
            n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
            den = n11 * n00 + n01 * n10
            out[f"{a}~{b}"] = {"disagreement": round((n01 + n10) / len(y), 4),
                               "double_fault": round(n00 / len(y), 4),
                               "Q": round((n11 * n00 - n01 * n10) / den, 4) if den else None}
    wrong = np.array([preds[n] != y for n in names]).sum(axis=0)
    out["sessions_wrong_by_0_1_2_3_members"] = [int((wrong == j).sum()) for j in range(4)]
    return out


# ---------------------------------------------------------------- stages
def eval_(a):
    m11 = module("11_eval_v2.py")
    d, k, checks, meta_a3 = a3_population(a)
    y, g = d.is_human.values, d.group.values
    configs = {**{n: m11.ALGOS[n] for n in m11.ORDER}, "LightGBM": make_lgbm,
               "VOTE": make_vote, "VOTE_SMOTE": make_vote_smote,
               "majority": lambda: DummyClassifier(strategy="most_frequent")}
    res, table, preds = {}, [], {}
    for name, make in configs.items():
        yp = m11.logo_predict(d, A3, make)
        preds[name] = yp
        m = m11.metrics(y, yp)
        m["MCC_95CI"] = m11.group_bootstrap(g, y, yp)
        m["per_group"] = m11.per_group(d, yp)
        res[name] = m
        print(f"[.] {name:<10} MCC {m['MCC']}", file=sys.stderr)
    yp = (d.log_max_rate <= 1.0).astype(int).values      # same rule as 11_eval_v2.py (10 chars/s)
    res["rule"] = {"rule": "Human if max implied typing rate (lines 2..k) <= 10 chars/s",
                   **m11.metrics(y, yp), "MCC_95CI": m11.group_bootstrap(g, y, yp)}
    for name, m in res.items():
        table.append({"config": name, **{x: m[x] for x in ("n", "TP", "FP", "TN", "FN", "TPR",
                                                           "FPR", "MCC")}, "MCC_95CI": m["MCC_95CI"]})
    majority = (np.array([preds[n] for n in MEMBERS]).sum(axis=0) >= 2).astype(int)
    ref = {"algo": meta_a3["algo"], "MCC_here": res[meta_a3["algo"]]["MCC"], "MCC_frozen": None}
    if os.path.exists(a.a3_report):           # written by 13_external_human.py fit
        ref["MCC_frozen"] = json.load(open(a.a3_report))["lab_LOGO_same_k_and_algo"]["A3"]["MCC"]
    ref["reproduced"] = (None if ref["MCC_frozen"] is None
                         else abs(ref["MCC_here"] - ref["MCC_frozen"]) < 1e-4)
    out = {"stage": "eval", "amendment": "PREREG_AMENDMENT_2", "k": k, "features": A3,
           "population": {"sessions": int(len(d)), "human": int(y.sum()),
                          "bot": int((y == 0).sum()), "groups": int(d.group.nunique())},
           "data_checks": checks, "versions": versions(),
           "frozen_A3_reference": ref,
           "vote_equals_majority_of_members": bool((majority == preds["VOTE"]).all()),
           "diversity_of_members": diversity(y, {n: preds[n] for n in MEMBERS}),
           "results": res,
           "note": "All configurations were fixed in PREREG_AMENDMENT_2 before this run; "
                   "nothing is chosen from these results (plain LOGO, no selection)."}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/eval.json", "w"), indent=1, default=str)
    pd.DataFrame(table).to_csv(f"{a.out}/eval_table.csv", index=False)
    print(pd.DataFrame(table).to_string(index=False))
    print(json.dumps({x: out[x] for x in ("population", "data_checks", "frozen_A3_reference",
                                           "vote_equals_majority_of_members",
                                           "diversity_of_members")}, indent=1))
    print(f"[+] {a.out}/eval.json  {a.out}/eval_table.csv")


def fit(a):
    path = f"{a.models}/model_E_v2.joblib"
    if os.path.exists(path) and not a.force:
        sys.exit(f"[-] {path} exists (frozen). Use --force only to rebuild it on purpose.")
    d, k, checks, meta_a3 = a3_population(a)
    clf = make_vote().fit(d[A3].values, d.is_human.values)
    meta = {"model": "E", "k": k, "algo": "VOTE-hard(LightGBM, LogReg, SVM)", "voting": "hard",
            "members": {n: str(p) for n, p in members()}, "features": A3,
            "trained_on": "lab dev sessions used to refit Models A and A3",
            "sessions": int(len(d)), "n_train_human": int(d.is_human.sum()),
            "features_csv_sha256": checks["features_csv_sha256"], "data_checks": checks,
            "versions": versions(), "sklearn": sklearn.__version__,
            "decision": "Human when at least 2 of the 3 members say Human; "
                        "score = share of member votes for Human",
            "amendment": "PREREG_AMENDMENT_2"}
    os.makedirs(a.models, exist_ok=True)
    joblib.dump({"clf": clf, "meta": meta}, path)
    h = sha256(path)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"stage": "fit", "model_E": meta, "model_E_sha256": h,
               "train_vote_shares": share_counts(vote_share(clf, d[A3].values))},
              open(f"{a.out}/model_E.json", "w"), indent=1, default=str)
    print(json.dumps(meta, indent=1, default=str))
    print(f"{h}  {path}")
    print(f"[+] {path}  {a.out}/model_E.json")


def load_models(a):
    e, a3 = joblib.load(a.model_e), joblib.load(a.model_a3)
    me, ma = e["meta"], a3["meta"]
    if me.get("model") != "E" or list(me["features"]) != A3 or int(me["k"]) != int(ma["k"]):
        sys.exit("[-] model E does not match Model A3 (k / features) - stop")
    return e["clf"], me, a3["clf"], ma, int(me["k"])


class _Capture:
    """Lets 15_fresh_A3.predict_split build the rows exactly as frozen, and keeps them."""
    def __init__(self, clf):
        self.clf, self.X = clf, None

    def predict(self, X):
        self.X = np.asarray(X, dtype=float)
        return self.clf.predict(self.X)


def score_humans(a):
    clf_e, meta_e, clf_a3, meta_a3, k = load_models(a)
    res = {"stage": "score-humans", "amendment": "PREREG_AMENDMENT_2",
           "model_E_sha256": sha256(a.model_e), "model_A3_sha256": sha256(a.model_a3), "k": k}
    os.makedirs(a.out, exist_ok=True)
    # 1) fresh participants and the red-team bot: the population of 15_fresh_A3.py
    if os.path.exists(a.features) and os.path.exists(a.sessions):
        m15 = module("15_fresh_A3.py")
        df, sess = pd.read_csv(a.features), pd.read_csv(a.sessions)
        S, _ = module("10_build_lab_v2.py").load(a.cowrie)
        lab = {}
        for split in ("fresh", "redteam"):
            cap = _Capture(clf_e)
            d, y_e = m15.predict_split(df, S, cap, k, split)
            if not len(d):
                continue
            _, y_a3 = m15.predict_split(df, S, clf_a3, k, split)
            share = vote_share(clf_e, cap.X)
            pd.DataFrame({"session": d.session, "group": d.group, "E": y_e, "A3": y_a3,
                          "E_vote_share": np.round(share, 4)}).to_csv(
                f"{a.out}/{split}_sessions.csv", index=False)
            r = {}
            for name, yp in (("E", y_e), ("A3", y_a3)):
                r[name] = {"reached_line_k": int(len(d)), "predicted_human": int(yp.sum()),
                           "per_group": {g: f"{int(x.sum())}/{len(x)}" for g, x in
                                         pd.Series(yp, index=d.group.values).groupby(level=0)}}
                if split == "fresh":
                    r[name]["TPR_at_decision"] = round(float(yp.mean()), 4)
                    r[name]["TPR_95CI_participant_bootstrap"] = m15.boot_ci(d.group.values, yp)
                else:
                    r[name]["TNR"] = round(1 - float(yp.mean()), 4)
            r["E_vote_shares"] = share_counts(share)
            r["E_human_A3_bot"] = int(((y_e == 1) & (y_a3 == 0)).sum())
            r["E_bot_A3_human"] = int(((y_e == 0) & (y_a3 == 1)).sum())
            if split == "fresh":
                all_fresh = sess[(sess.split == "fresh") & (sess.n_lines >= 1)]
                r["fresh_sessions_total"] = int(len(all_fresh))
                for name, yp in (("E", y_e), ("A3", y_a3)):
                    r[name]["TPR_operational"] = round(float(yp.sum()) / len(all_fresh), 4)
            lab[split] = r
        res["lab_held_out"] = lab
    # 2) KYPO (every trainee is human): the pipeline of 13_external_human.py score
    if os.path.isdir(a.kypo):
        m13 = module("13_external_human.py")
        qc, T = m13.kypo_load(a.kypo)
        rows, X = [], []
        for name, x in T.items():
            f = fv2.json_prefix_features(x["lines"], None, k)
            rows.append({"trainee": name, "training": x["training"],
                         "n_lines": len(x["lines"]), "reach": int(f is not None)})
            if f is not None:
                X.append([f[c] for c in A3])
        t = pd.DataFrame(rows)
        coarse = m13.whole_second_trainings(T)
        t["primary"] = (~t.training.isin(coarse)).astype(int)
        r = t.reach == 1
        X = np.array(X, dtype=float).reshape(-1, len(A3))
        for name, clf in (("E", clf_e), ("A3", clf_a3)):
            t[name] = 0
            if len(X):
                t.loc[r, name] = clf.predict(X)
        t["E_vote_share"] = np.nan
        if len(X):
            t.loc[r, "E_vote_share"] = np.round(vote_share(clf_e, X), 4)
        p, sec = t[(t.primary == 1) & r], t[(t.primary == 0) & r]
        with1 = t[(t.primary == 1) & (t.n_lines >= 1)]
        kres = {"trainees": int(len(t)), "whole_second_trainings_not_primary": coarse,
                "primary_trainees_with_bash_lines": int(len(with1))}
        for name in ("E", "A3"):
            y = p[name].values.astype(float)
            kres[name] = {"reached_line_k": int(len(p)), "predicted_human": int(y.sum()),
                          "human_recall": round(float(y.mean()), 4) if len(y) else None,
                          "human_recall_95CI": m13.boot_ci(y),
                          "operational_recall": round(float(y.sum()) / len(with1), 4)
                          if len(with1) else None,
                          "per_training": {g: f"{int(x[name].sum())}/{len(x)}"
                                           for g, x in p.groupby("training")},
                          "whole_second_trainings_secondary": {
                              "reached_line_k": int(len(sec)),
                              "predicted_human": int(sec[name].sum()),
                              "human_recall": round(float(sec[name].mean()), 4) if len(sec) else None}}
        kres["E_vote_shares"] = share_counts(p.E_vote_share.values)
        kres["E_human_A3_bot"] = int(((p.E == 1) & (p.A3 == 0)).sum())
        kres["E_bot_A3_human"] = int(((p.E == 0) & (p.A3 == 1)).sum())
        rec = kres["E"]["human_recall"]
        kres["E2_pass"] = bool(rec is not None and rec >= E2_LIMIT)
        res["kypo"] = kres
        res["kypo_quality"] = qc
        t.to_csv(f"{a.out}/kypo_trainees.csv", index=False)
    json.dump(res, open(f"{a.out}/humans.json", "w"), indent=1, default=str)
    print(json.dumps({x: res[x] for x in res if x != "kypo_quality"}, indent=1, default=str))
    print(f"[+] {a.out}/humans.json")


def score_bots(a):
    clf_e, meta_e, clf_a3, meta_a3, k = load_models(a)
    m12 = module("12_external_v2.py")
    files = sorted(glob.glob(a.days))
    if not files:
        sys.exit("[-] no day files match --days")
    qc, df, lines_of = m12.build(files)
    df = m12.label(df, "template", 5)
    camp = df[df.campaign == 1].copy()
    excluded = {}
    if a.exclude:
        for line in open(a.exclude):
            if line.strip() and not line.startswith("#"):
                h, _, why = line.strip().partition(" ")
                excluded[h] = why.strip()
    camp = camp[~camp.seq.isin(excluded)].copy()
    camp["reach"] = (camp.n_lines >= k).astype(int)
    idx = camp.index[camp.reach == 1]
    X = np.array([[fv2.json_prefix_features(lines_of[s], None, k)[c] for c in A3]
                  for s in camp.loc[idx, "session"]], dtype=float).reshape(-1, len(A3))
    res = {"stage": "score-bots", "amendment": "PREREG_AMENDMENT_2", "tag": a.tag,
           "files": [os.path.basename(f) for f in files], "label": "template", "kappa": 5,
           "model_E_sha256": sha256(a.model_e), "model_A3_sha256": sha256(a.model_a3),
           "quality": qc, "excluded_sequences": excluded,
           "campaign_sessions_ge1_line": int(len(camp)), "reached_line_k": int(len(idx))}
    per_all = None
    for name, clf in (("E", clf_e), ("A3", clf_a3)):
        camp["pred"] = 0
        if len(idx):
            camp.loc[idx, "pred"] = clf.predict(X)
        overall, per = m12.summarize(camp, "family")
        res[name] = {"false_human_rates": overall,
                     "per_sequence_secondary": m12.summarize(camp, "seq")[0],
                     "by_config_group": {f"g{g}": m12.summarize(x)[0]["conditional_FPR"]
                                         for g, x in camp.groupby("config_group")}}
        per = per.rename(columns={"fp": f"fp_{name}", "fpr": f"fpr_{name}"})
        per_all = per if per_all is None else per_all.merge(per[["family", f"fp_{name}", f"fpr_{name}"]],
                                                            on="family")
    if len(idx):
        res["E_vote_shares"] = share_counts(vote_share(clf_e, X))
    o = res["E"]["false_human_rates"]
    res["E1_pass"] = bool(o["conditional_FPR"] is not None and o["conditional_FPR"] <= E1_LIMIT
                          and o["macro_FPR_ge5"] is not None and o["macro_FPR_ge5"] <= E1_LIMIT)
    os.makedirs(a.out, exist_ok=True)
    if per_all is not None:
        per_all.sort_values("n", ascending=False).to_csv(f"{a.out}/bots_{a.tag}_per_family.csv",
                                                          index=False)
    json.dump(res, open(f"{a.out}/bots_{a.tag}.json", "w"), indent=1, default=str)
    print(json.dumps({x: res[x] for x in res if x != "quality"}, indent=1, default=str))
    print(f"[+] {a.out}/bots_{a.tag}.json  {a.out}/bots_{a.tag}_per_family.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["eval", "fit", "score-humans", "score-bots"])
    ap.add_argument("--features", default="data/processed/lab_v2_features.csv")
    ap.add_argument("--sessions", default="data/processed/lab_v2_sessions.csv")
    ap.add_argument("--cowrie", default=os.path.expanduser("~/cowrie/var/log/cowrie"))
    ap.add_argument("--model-a", dest="model_a", default="models/model_A_v2.joblib")
    ap.add_argument("--model-a3", dest="model_a3", default="models/model_A3_v2.joblib")
    ap.add_argument("--model-e", dest="model_e", default="models/model_E_v2.joblib")
    ap.add_argument("--a3-report", dest="a3_report", default="reports/v2/model_A3.json")
    ap.add_argument("--models", default="models")
    ap.add_argument("--kypo", default="data/external/kypo/data")
    ap.add_argument("--days", default="", help="glob of COW160 day files (*.json.gz)")
    ap.add_argument("--exclude", default="", help="exclusions file (12_external_v2.py format)")
    ap.add_argument("--tag", default="confirmatory3")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true", help="fit: overwrite model_E_v2.joblib")
    a = ap.parse_args()
    {"eval": eval_, "fit": fit, "score-humans": score_humans, "score-bots": score_bots}[a.stage](a)


if __name__ == "__main__":
    main()
