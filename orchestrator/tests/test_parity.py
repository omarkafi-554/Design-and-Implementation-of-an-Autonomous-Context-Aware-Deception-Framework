#!/usr/bin/env python3
"""test_parity.py - the live module must reproduce the frozen results exactly.
Fresh sessions p16-p20 from the raw Cowrie log -> live_classify, compared session by session
with the frozen model. Expected: A3 -> 12/17 Human (reports/v2/fresh_A3.json).
Read-only: writes nothing to data/, models/ or reports/."""
import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "orchestrator"))
sys.path.append(os.path.join(ROOT, "scripts"))
import fv2  # noqa: E402
from live_classify import Classifier  # noqa: E402


def load_lab_module():
    p = os.path.join(ROOT, "scripts", "10_build_lab_v2.py")
    spec = importlib.util.spec_from_file_location("build_lab", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(clf_live, label):
    feats = pd.read_csv(os.path.join(ROOT, "data/processed/lab_v2_features.csv"))
    S, _ = load_lab_module().load(os.path.expanduser("~/cowrie/var/log/cowrie"))
    k = clf_live.k
    d = feats[(feats.split == "fresh") & (feats.k == k)].dropna(subset=fv2.JSON_FEATURES)
    frozen = clf_live.json_model
    live_h, mism = 0, 0
    for _, row in d.iterrows():
        s = S[row.session]
        r = clf_live.classify(s["cmds"], s["login"])
        if frozen.needs_login:
            ref = frozen.clf.predict([[row[c] for c in frozen.features]])[0]
        else:   # A3: rebuilt without a login time, exactly as 15_fresh_A3.py
            lines, _, _ = fv2.input_lines(s["cmds"])
            f = fv2.json_prefix_features(lines, None, k)
            ref = frozen.clf.predict([[f[c] for c in frozen.features]])[0]
        live = r["verdict"] == "Human"
        live_h += live
        ok = live == (int(ref) == 1)
        mism += not ok
        print(f"  {row.group:4s} {row.session}  live={r['verdict']:9s} p={r['p_human']}  "
              f"frozen={'Human' if ref == 1 else 'Bot':5s} {'OK' if ok else 'MISMATCH'}")
    print(f"[{label}] live Human {live_h}/{len(d)}   mismatches: {mism}")
    return mism


if __name__ == "__main__":
    c = Classifier()
    print(f"[+] model {c.json_model.name}, k = {c.k}, theta = {c.theta}")
    bad = run(c, c.json_model.name)
    sys.exit(1 if bad else 0)
