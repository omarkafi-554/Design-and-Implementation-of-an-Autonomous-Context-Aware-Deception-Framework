#!/usr/bin/env python3
"""live_classify.py - Algorithm 2.10 CLASSIFY-LIVE (GP1, Week 1 of the task plan;
hard-vote support added for PREREG_AMENDMENT_2, 21 Oct 2026).

Classifies one SSH session at input line k with a FROZEN model.
  * Input lines and features come from scripts/fv2.py, imported unchanged (never copied),
    so live features are computed by exactly the code that built the training data.
  * Every model file is checked against the SHA-256 written in config.yaml; a mismatch stops.
  * Result: Human, Bot or Uncertain.
      Uncertain = the session has not reached line k yet, or a feature the model needs is
      missing. It is NOT a new probability band: theta stays at the pre-registered 0.5.
  * Missing keystroke data stays missing; it is never replaced by 0.
  * Two kinds of frozen model are accepted:
      - a model with predict_proba (A, A3, B): p_human = P(Human);
      - Model E, a hard-voting ensemble (16_ensemble_v2.py): it has no P(Human), so
        p_human = the share of member votes for Human (0, 1/3, 2/3 or 1), computed exactly
        as in 16_ensemble_v2.vote_share. Verdict = the ensemble's own predict (majority),
        i.e. Human when at least 2 of the 3 members say Human.
"""
import hashlib
import math
import os
import sys

import joblib
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # ~/deception-framework
sys.path.append(os.path.join(ROOT, "scripts"))
import fv2  # noqa: E402  (frozen feature code)

HUMAN = 1                                         # label coding of 10_build_lab_v2.py (is_human)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class FrozenModel:
    def __init__(self, path, expected_sha):
        path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if not isinstance(expected_sha, str) or len(expected_sha.strip()) != 64:
            raise RuntimeError(f"SHA-256 for {path} missing in config.yaml (write it in quotes)")
        got = sha256(path)
        if got != expected_sha.strip().lower():
            raise RuntimeError(f"SHA-256 mismatch for {path}: {got} - the model is not the frozen one")
        obj = joblib.load(path)
        self.clf, self.meta = obj["clf"], obj["meta"]
        self.path, self.sha = path, got
        self.name = str(self.meta["model"])
        self.k = int(self.meta["k"])
        self.features = list(self.meta["features"])
        self.needs_login = "log_first_latency" in self.features
        self.needs_tty = any(f in fv2.TTY_FEATURES for f in self.features)
        self.human_col = list(self.clf.classes_).index(HUMAN)
        # hard vote: sklearn removes predict_proba when voting="hard"
        self.hard_vote = getattr(self.clf, "voting", None) == "hard"
        if self.hard_vote:
            n = len(self.clf.estimators_)
            if n % 2 == 0:
                raise RuntimeError(f"{self.name}: {n} voters can tie - hard vote needs an odd number")
            if self.meta.get("voting") != "hard":
                raise RuntimeError(f"{self.name}: meta does not record voting = hard")
            self.score = "vote_share"
        elif hasattr(self.clf, "predict_proba"):
            self.score = "probability"
        else:
            raise RuntimeError(f"{self.name}: no predict_proba and not a hard vote - no p_human")

    def p_human(self, X):
        """P(Human), or for a hard vote the share of member votes for Human."""
        if self.hard_vote:
            labels = self.clf.classes_[self.clf.transform(X)]   # = 16_ensemble_v2.vote_share
            return float((labels == HUMAN).mean(axis=1)[0])
        return float(self.clf.predict_proba(X)[0][self.human_col])


class Classifier:
    def __init__(self, config_path=os.path.join(HERE, "config.yaml")):
        cfg = yaml.safe_load(open(config_path))["classifier"]
        self.k = int(cfg["k"])
        self.theta = float(cfg["theta"])
        self.json_model = FrozenModel(cfg["model_json"], cfg.get("model_json_sha256"))
        self.b_model = (FrozenModel(cfg["model_b"], cfg.get("model_b_sha256"))
                        if cfg.get("use_b") else None)
        for m in (self.json_model, self.b_model):
            if m is not None and m.k != self.k:
                raise RuntimeError(f"config k = {self.k} but {m.name} was frozen at k = {m.k}")
        if self.json_model.needs_tty:
            raise RuntimeError("model_json must be a JSON-only model (A, A3 or E)")

    def describe(self):
        ms = [self.json_model] + ([self.b_model] if self.b_model else [])
        return {"k": self.k, "theta": self.theta,
                "models": {m.name: {"algo": m.meta.get("algo"), "features": m.features,
                                    "score": m.score, "sha256": m.sha} for m in ms}}

    @staticmethod
    def _out(verdict, p, model, n_lines, reason):
        return {"verdict": verdict, "p_human": p, "model": model, "n_lines": n_lines, "reason": reason}

    def classify(self, cmd_events, login_ts=None, tty_records=None):
        """cmd_events: [(t, input_text)] of one session (cowrie.command.input, epoch seconds).
        login_ts: time of cowrie.login.success (None if unknown).
        tty_records: fv2.read_tty(...) of this session's own TTY log so far (or None)."""
        lines, _, _ = fv2.input_lines(cmd_events)
        n = len(lines)
        if n < self.k:
            return self._out("Uncertain", None, None, n, f"pending: {n}/{self.k} input lines")

        model, tty = self.json_model, None
        if self.b_model is not None and tty_records:
            tf = fv2.tty_prefix_features(tty_records, lines[self.k - 1][0], self.k)
            if tf["tty_input_writes"] > 0:          # Model B only with real keyboard input
                model, tty = self.b_model, tf

        # A3 and E are defined without a login time (as fitted in 13_external_human.py / 16)
        f = fv2.json_prefix_features(lines, login_ts if model.needs_login else None, self.k)
        if tty:
            f.update(tty)
        x = [f.get(name, float("nan")) for name in model.features]
        missing = [nm for nm, v in zip(model.features, x) if v is None or math.isnan(v)]
        if missing:
            return self._out("Uncertain", None, model.name, n, "missing: " + ", ".join(missing))

        X = np.array([x], dtype=float)
        p = model.p_human(X)
        if self.theta == 0.5:
            human = int(model.clf.predict(X)[0]) == HUMAN   # identical to the frozen evaluation
        else:
            human = p >= self.theta
        return self._out("Human" if human else "Bot", round(p, 4), model.name, n,
                         f"{model.name} at line {self.k}")


if __name__ == "__main__":
    import json
    c = Classifier()
    print(json.dumps(c.describe(), indent=1))
