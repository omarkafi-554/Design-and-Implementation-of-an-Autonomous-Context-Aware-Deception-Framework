#!/usr/bin/env python3
"""13_external_human.py - pre-registered external tests on single-class data sets (PREREG §8b).

The external human set (KYPO shell commands: Svabensky et al., Data in Brief 2021,
doi:10.1016/j.dib.2021.107398; Zenodo 8136017 v4) has a timestamp per command but no login
event and no keystrokes. So log_first_latency and Model B cannot be computed there.

Model A3 = the three Model A features that need no login time:
  log_gap_median, gap_cv, log_max_rate (maximum over lines 2..k)
computed by fv2.json_prefix_features(lines, None, k). A3 uses the k and the algorithm frozen
for Model A and is fitted ONLY on the lab dev sessions used to refit Model A.
The external sets are used only for testing and are never pooled.

Stages
  inspect : KYPO counts only (records, types, order, duplicates, timestamp precision, trainees
            reaching line 3/4/5). No feature is computed and no prediction is made.
  fit     : refit A3 and save models/model_A3_v2.joblib; lab LOGO metrics of A3 and of the
            full Model A at the same k and algorithm (feature ablation, not nested).
  score   : A3 on KYPO (every trainee is human) -> human recall, 95 % bootstrap CI over
            trainees, recall per training definition, P4 (recall >= 0.80).
            With --days: A3 on COW160x4 days (every campaign session is a bot), with the
            pipeline, label and exclusions of 12_external_v2.py -> false-human rates.

KYPO input lines: records with cmd_type == "bash-command" and a non-empty command, sorted by
timestamp, exact duplicates (same timestamp and command) removed; one decision per trainee
(one *-useractions.json file = one trainee) on the first k lines.
A training whose bash timestamps are all whole seconds is excluded from the primary test
(two commands within one second would give a gap of 0); it is reported separately.

Usage
  python3 scripts/13_external_human.py inspect --kypo data/external/kypo/data
  python3 scripts/13_external_human.py fit
  python3 scripts/13_external_human.py score --kypo data/external/kypo/data \
          --days "data/external/cow160/confirm/*.json.gz" \
          --exclude reports/v2/external_confirmatory/exclusions.txt
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fv2  # noqa: E402

SEED, B, P4_LIMIT = 42, 2000, 0.80
A3 = ["log_gap_median", "gap_cv", "log_max_rate"]
FRAC = re.compile(r"\.(\d+)")


def module(name):
    """Import a frozen sibling script (file names start with a digit) without running it."""
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  os.path.join(HERE, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def iso_ts(s):
    """ISO-8601 with any number of fractional digits (Python 3.10 accepts only 3 or 6)."""
    s = str(s).strip().replace("Z", "+00:00")
    m = FRAC.search(s)
    if m:
        s = s[:m.start()] + "." + (m.group(1) + "000000")[:6] + s[m.end():]
    return datetime.fromisoformat(s).timestamp()


def kypo_files(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*-useractions.json"), recursive=True))
    if not files:
        sys.exit(f"[-] no *-useractions.json files under {root}")
    return files


def kypo_records(path):
    txt = open(path, encoding="utf-8", errors="replace").read().strip()
    if txt.startswith("["):
        return json.loads(txt), 0
    out, bad = [], 0
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return out, bad


def kypo_load(root):
    """{trainee: {"training", "lines": [(t, cmd)]}} + quality counters."""
    qc, T = Counter(), {}
    for f in kypo_files(root):
        rel = os.path.relpath(f, root)
        recs, bad = kypo_records(f)
        qc["files"] += 1
        qc["bad_json_lines"] += bad
        bash, n_frac = [], 0
        for r in recs:
            qc["records"] += 1
            qc[f"type:{r.get('cmd_type')}"] += 1
            if r.get("cmd_type") != "bash-command":
                continue
            cmd = (r.get("cmd") or "").strip()
            if not cmd:
                qc["bash_empty"] += 1
                continue
            raw = r.get("timestamp_str")
            try:
                t = iso_ts(raw)
            except (TypeError, ValueError):
                qc["bash_bad_timestamp"] += 1
                continue
            m = FRAC.search(str(raw))
            qc[f"bash_ts_fraction_digits:{len(m.group(1)) if m else 0}"] += 1
            n_frac += int(bool(m))
            bash.append((t, cmd))
        qc["bash_out_of_order"] += sum(1 for x, y in zip(bash, bash[1:]) if y[0] < x[0])
        bash.sort(key=lambda x: x[0])
        lines, seen = [], set()
        for t, c in bash:
            if (t, c) in seen:
                qc["bash_exact_duplicates_removed"] += 1
                continue
            seen.add((t, c))
            lines.append((t, c))
        T[rel] = {"training": rel.split(os.sep)[0], "lines": lines, "n_frac": n_frac}
    return dict(qc), T


def whole_second_trainings(T):
    """Trainings in which no bash timestamp has a fractional part."""
    by = defaultdict(lambda: [0, 0])
    for x in T.values():
        by[x["training"]][0] += x["n_frac"]
        by[x["training"]][1] += len(x["lines"])
    return sorted(tr for tr, (frac, n) in by.items() if n and frac == 0)


def boot_ci(y):
    if not len(y):
        return None
    rng = np.random.default_rng(SEED)
    v = [y[rng.integers(0, len(y), len(y))].mean() for _ in range(B)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def inspect(a):
    qc, T = kypo_load(a.kypo)
    n = {k: sum(1 for x in T.values() if len(x["lines"]) >= k) for k in fv2.K_CANDIDATES}
    per = Counter(x["training"] for x in T.values())
    coarse = whole_second_trainings(T)
    res = {"stage": "inspect", "kypo_root": a.kypo, "quality": qc,
           "trainees": len(T), "trainees_with_bash_lines": sum(1 for x in T.values() if x["lines"]),
           "trainees_reaching_line_k": n, "trainees_per_training": dict(per),
           "whole_second_trainings_not_primary": coarse,
           "trainees_primary": sum(1 for x in T.values() if x["training"] not in coarse)}
    out = f"{a.out}/external_human_kypo"
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(f"{out}/inspect.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"[+] {out}/inspect.json  (counts only: no features, no predictions)")


def fit(a):
    m10, m11 = module("10_build_lab_v2.py"), module("11_eval_v2.py")
    meta_a = joblib.load(a.model_a)["meta"]
    k, algo = int(meta_a["k"]), meta_a["algo"]
    feat = pd.read_csv(a.features)
    sha = hashlib.sha256(open(a.features, "rb").read()).hexdigest()
    if sha != meta_a.get("features_csv_sha256"):
        sys.exit("[-] features CSV differs from the one Model A was frozen on")
    pop = feat[(feat.split == "dev") & (feat.k == k)].dropna(subset=fv2.JSON_FEATURES)
    pop = pop.reset_index(drop=True)
    if len(pop) != int(meta_a.get("n_train", len(pop))):
        sys.exit("[-] A3 population differs from the sessions Model A was refitted on")
    S, _ = m10.load(a.cowrie)
    rows = []
    for sid in pop.session:
        lines, _, _ = fv2.input_lines(S[sid]["cmds"])
        f = fv2.json_prefix_features(lines, None, k)
        rows.append([f[c] for c in A3])
    X3 = pd.DataFrame(rows, columns=[c + "_a3" for c in A3])
    d = pd.concat([pop, X3], axis=1)
    for c in ("log_gap_median", "gap_cv"):
        if not np.allclose(d[c], d[c + "_a3"], atol=1e-9):
            sys.exit(f"[-] rebuilt {c} differs from the features CSV - stop")
    f3 = [c + "_a3" for c in A3]
    clf = m11.ALGOS[algo]().fit(d[f3].values, d.is_human.values)
    meta = {"model": "A3", "k": k, "algo": algo, "features": A3,
            "trained_on": "lab dev sessions used to refit Model A", "sessions": int(len(d)),
            "features_csv_sha256": sha, "sklearn": sklearn.__version__}
    joblib.dump({"clf": clf, "meta": meta}, f"{a.models}/model_A3_v2.joblib")
    abl = {}
    for name, cols in (("A_full", fv2.JSON_FEATURES), ("A3", f3)):
        yp = m11.logo_predict(d, cols, m11.ALGOS[algo])
        abl[name] = {**m11.metrics(d.is_human.values, yp),
                     "MCC_95CI": m11.group_bootstrap(d.group.values, d.is_human.values, yp)}
    res = {"stage": "fit", "k": k, "algo": algo, "model_A3": meta,
           "lab_LOGO_same_k_and_algo": abl,
           "note": "LOGO with the configuration chosen for Model A (not nested)"}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/model_A3.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
    print(f"[+] {a.models}/model_A3_v2.joblib  {a.out}/model_A3.json")


def score(a):
    obj = joblib.load(a.model_a3)
    clf, meta = obj["clf"], obj["meta"]
    k = int(meta["k"])
    qc, T = kypo_load(a.kypo)
    rows = []
    for name, x in T.items():
        f = fv2.json_prefix_features(x["lines"], None, k)
        rows.append({"trainee": name, "training": x["training"], "n_lines": len(x["lines"]),
                     "reach": int(f is not None), **({c: f[c] for c in A3} if f else {})})
    t = pd.DataFrame(rows)
    coarse = whole_second_trainings(T)
    t["primary"] = (~t.training.isin(coarse)).astype(int)
    t["pred"] = 0
    r = t.reach == 1
    if r.any():
        t.loc[r, "pred"] = clf.predict(t.loc[r, A3].values)
    p = t[(t.primary == 1) & r]
    y = p.pred.values.astype(float)
    with1 = t[(t.primary == 1) & (t.n_lines >= 1)]
    recall = round(float(y.mean()), 4) if len(y) else None
    sec = t[(t.primary == 0) & r]
    res = {"stage": "score", "model": meta, "kypo_quality": qc,
           "trainees": int(len(t)), "whole_second_trainings_not_primary": coarse,
           "primary_trainees_with_bash_lines": int(len(with1)),
           "reached_line_k": int(len(p)), "predicted_human": int(y.sum()),
           "human_recall": recall, "human_recall_95CI": boot_ci(y),
           "operational_recall": round(float(y.sum()) / len(with1), 4) if len(with1) else None,
           "per_training": {g: f"{int(x.pred.sum())}/{len(x)}"
                            for g, x in p.groupby("training")},
           "whole_second_trainings_secondary": {
               "reached_line_k": int(len(sec)), "predicted_human": int(sec.pred.sum()),
               "human_recall": round(float(sec.pred.mean()), 4) if len(sec) else None},
           "P4_pass": bool(recall is not None and recall >= P4_LIMIT)}
    out = f"{a.out}/external_human_kypo"
    os.makedirs(out, exist_ok=True)
    t.to_csv(f"{out}/trainees.csv", index=False)
    json.dump(res, open(f"{out}/score.json", "w"), indent=1, default=str)
    print(json.dumps({x: res[x] for x in res if x != "model"}, indent=1, default=str))
    print(f"[+] {out}/score.json  {out}/trainees.csv")
    if a.days:
        score_bots(a, clf, meta)


def score_bots(a, clf, meta):
    m12 = module("12_external_v2.py")
    files = sorted(glob.glob(a.days))
    if not files:
        sys.exit("[-] no day files match --days")
    k = int(meta["k"])
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
    camp["pred"] = 0
    idx = camp.index[camp.reach == 1]
    if len(idx):
        X = np.array([[fv2.json_prefix_features(lines_of[s], None, k)[c] for c in A3]
                      for s in camp.loc[idx, "session"]], dtype=float)
        camp.loc[idx, "pred"] = clf.predict(X)
    overall, per = m12.summarize(camp, "family")
    res = {"stage": "score_bots", "model": meta, "files": [os.path.basename(f) for f in files],
           "label": "template", "kappa": 5, "quality": qc,
           "excluded_sequences": {h: w for h, w in excluded.items()},
           "false_human_rates": overall,
           "per_sequence_secondary": m12.summarize(camp, "seq")[0]}
    out = f"{a.out}/external_bots_A3_{a.tag}"
    os.makedirs(out, exist_ok=True)
    per.sort_values("n", ascending=False).to_csv(f"{out}/per_family.csv", index=False)
    json.dump(res, open(f"{out}/score.json", "w"), indent=1, default=str)
    print(json.dumps({"false_human_rates": overall}, indent=1, default=str))
    print(f"[+] {out}/score.json  {out}/per_family.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inspect", "fit", "score"])
    ap.add_argument("--kypo", default="data/external/kypo/data")
    ap.add_argument("--features", default="data/processed/lab_v2_features.csv")
    ap.add_argument("--cowrie", default=os.path.expanduser("~/cowrie/var/log/cowrie"))
    ap.add_argument("--model-a", dest="model_a", default="models/model_A_v2.joblib")
    ap.add_argument("--model-a3", dest="model_a3", default="models/model_A3_v2.joblib")
    ap.add_argument("--models", default="models")
    ap.add_argument("--days", default="", help="glob of COW160 day files (*.json.gz)")
    ap.add_argument("--exclude", default="", help="exclusions file of 12_external_v2.py")
    ap.add_argument("--tag", default="confirmatory")
    ap.add_argument("--out", default="reports/v2")
    a = ap.parse_args()
    {"inspect": inspect, "fit": fit, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
