"""test_ensemble_v2.py - unit tests for 16_ensemble_v2.py (Model E), on synthetic data only.
Run from the project folder:  python3 scripts/test_ensemble_v2.py   (exit code = failures)"""
import os, sys, tempfile, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util

def module(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), os.path.join(HERE, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        ok.failed += 1
ok.failed = 0

warnings.simplefilter("error")              # any library warning makes the run fail
E = module("16_ensemble_v2.py")
m11, m13, m15 = module("11_eval_v2.py"), module("13_external_human.py"), module("15_fresh_A3.py")

# synthetic A3-like data: 40 humans (1), 90 bots (0)
rng = np.random.default_rng(0)
Xh = np.c_[rng.normal(0.8, 0.3, 40), rng.normal(0.6, 0.2, 40), rng.normal(0.3, 0.3, 40)]
Xb = np.c_[rng.normal(-0.5, 0.5, 90), rng.normal(0.3, 0.2, 90), rng.normal(1.5, 0.6, 90)]
X, y = np.r_[Xh, Xb], np.r_[np.ones(40, int), np.zeros(90, int)]
Xt = np.r_[np.c_[rng.normal(0.8, 0.3, 15), rng.normal(0.6, 0.2, 15), rng.normal(0.3, 0.3, 15)],
           np.c_[rng.normal(-0.5, 0.5, 15), rng.normal(0.3, 0.2, 15), rng.normal(1.5, 0.6, 15)]]

# 1) configuration fixed in the amendment
v = E.make_vote()
names = [n for n, _ in v.estimators]
ok(v.voting == "hard" and names == ["LightGBM", "LogReg", "SVM"], f"hard vote of {names}")
lg = v.estimators[0][1][-1].get_params()
want = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 7, "max_depth": 3,
        "min_child_samples": 10, "reg_lambda": 1.0, "class_weight": "balanced",
        "random_state": 42, "deterministic": True, "n_jobs": 1}
ok(all(lg[x] == w for x, w in want.items()), "LightGBM small-data configuration")
for i, algo in ((1, "LogReg"), (2, "SVM")):
    mine, frozen = v.estimators[i][1][-1].get_params(), m11.ALGOS[algo]()[-1].get_params()
    ok(mine == frozen, f"{algo} member = 11_eval_v2.py {algo} configuration")
ok(E.A3 == m13.A3 == m15.A3 == ["log_gap_median", "gap_cv", "log_max_rate"],
   "A3 features identical to 13_external_human.py and 15_fresh_A3.py")

# 2) hard voting: majority of the members, no probabilities, no ties
f = E.make_vote().fit(X, y)
ok(not hasattr(f, "predict_proba"), "hard voting has no predict_proba")
members = np.array([est.predict(Xt) for est in f.estimators_])
ok(((members.sum(axis=0) >= 2).astype(int) == f.predict(Xt)).all(), "prediction = majority of 3 members")
s = E.vote_share(f, Xt)
ok(set(np.round(s * 3).astype(int)) <= {0, 1, 2, 3}, "vote share in {0, 1/3, 2/3, 1}")
ok(((s >= 0.5).astype(int) == f.predict(Xt)).all(), "Human  <=>  at least 2 of 3 votes")
ok(list(f.classes_) == [0, 1], "classes_ = [0 (Bot), 1 (Human)]")

# 3) deterministic refit and joblib round trip
g = E.make_vote().fit(X, y)
ok((g.predict(Xt) == f.predict(Xt)).all() and np.allclose(E.vote_share(g, Xt), s), "refit is deterministic")
import joblib
p = os.path.join(tempfile.mkdtemp(prefix="test_ens_"), "e.joblib")
joblib.dump({"clf": f, "meta": {}}, p)
h = joblib.load(p)["clf"]
ok((h.predict(Xt) == f.predict(Xt)).all(), "joblib round trip keeps the predictions")

# 4) SMOTE (comparison only) resamples the training data in fit, never in predict
from imblearn.over_sampling import SMOTE
calls, orig = [], SMOTE.fit_resample
def spy(self, X_, y_, **kw):
    Xr, yr = orig(self, X_, y_, **kw)
    calls.append((np.bincount(y_).tolist(), np.bincount(yr).tolist()))
    return Xr, yr
SMOTE.fit_resample = spy
try:
    sm = E.make_vote_smote().fit(X, y)
    n_fit = len(calls)
    out = sm.predict(Xt)
finally:
    SMOTE.fit_resample = orig
ok(n_fit == 1 and calls[0] == ([90, 40], [90, 90]), f"SMOTE in fit only, 40 -> 90 humans {calls[:1]}")
ok(len(calls) == 1 and len(out) == len(Xt), "predict does not resample")
ok(all(e.get_params()["class_weight"] is None for _, e in sm.named_steps["vote"].estimators),
   "SMOTE variant: no class weights in the members")

# 5) diversity measures on a hand-made case
yy = np.array([1, 1, 0, 0])
d = E.diversity(yy, {"a": np.array([1, 0, 0, 0]), "b": np.array([1, 0, 1, 0]), "c": np.array([1, 1, 0, 0])})
ok(d["a~b"]["double_fault"] == 0.25 and d["a~b"]["disagreement"] == 0.25, "double fault / disagreement")
ok(d["sessions_wrong_by_0_1_2_3_members"] == [2, 1, 1, 0], "errors per session count")

print(f"\n{ok.failed} failed")
sys.exit(ok.failed)
