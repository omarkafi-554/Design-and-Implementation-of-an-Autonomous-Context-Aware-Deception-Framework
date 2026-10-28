#!/usr/bin/env python3
"""test_classify.py - checks of live_classify.Classifier with the REAL frozen models.

The expected SHA-256 values are read from reports/v2/MODELS_SHA256.txt (the freeze record),
not from config.yaml, so these tests also prove that config.yaml is not needed to trust a model.
Every test writes its own temporary config; nothing in models/, reports/ or config.yaml changes.
Run:  python3 orchestrator/tests/test_classify.py   (or with pytest)"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "orchestrator"))
from live_classify import Classifier, fv2  # noqa: E402

M = {n: os.path.join(ROOT, "models", f"model_{n}_v2.joblib") for n in ("A", "B", "A3")}


def frozen_sha():
    """{absolute model path: sha256} from the freeze record."""
    out = {}
    for line in open(os.path.join(ROOT, "reports", "v2", "MODELS_SHA256.txt")):
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            out[os.path.normpath(os.path.join(ROOT, parts[1]))] = parts[0]
    return out


SHA = frozen_sha()


def cfg(json_model="A3", k=3, sha=None, use_b=False):
    p = os.path.join(tempfile.mkdtemp(prefix="acdf_cfg_"), "config.yaml")
    s = SHA[M[json_model]] if sha is None else sha
    with open(p, "w") as f:
        f.write(f'classifier:\n  model_json: {M[json_model]}\n  model_json_sha256: "{s}"\n'
                f'  use_b: {"true" if use_b else "false"}\n  model_b: {M["B"]}\n'
                f'  model_b_sha256: "{SHA.get(M["B"], "")}"\n  k: {k}\n  theta: 0.5\n')
    return p


def refused(path, needle):
    try:
        Classifier(path)
    except RuntimeError as e:
        assert needle in str(e), str(e)
        return
    raise AssertionError("the classifier should have refused this config")


LOGIN = 1000.0
SESSION = [(1003.2, "uname -a"), (1006.9, "cat /etc/passwd"), (1011.4, "ls -la /tmp"),
           (1015.0, "exit")]


def test_frozen_record_has_all_models():
    assert all(M[n] in SHA for n in ("A", "B", "A3")), sorted(SHA)


def test_pending_below_k_is_uncertain():
    c = Classifier(cfg())
    r = c.classify(SESSION[:2], LOGIN)
    assert (r["verdict"], r["model"], r["n_lines"]) == ("Uncertain", None, 2)
    assert r["reason"].startswith("pending: 2/3")


def test_artefact_lines_do_not_count():
    # Cowrie logs `hostname` again for $(hostname) within 1 s: still only 2 real lines
    c = Classifier(cfg())
    r = c.classify([(1003.0, "echo $(hostname)"), (1003.2, "hostname"), (1007.0, "id")], LOGIN)
    assert r["verdict"] == "Uncertain" and r["n_lines"] == 2


def test_a3_matches_the_frozen_model_directly():
    c = Classifier(cfg())
    r = c.classify(SESSION, LOGIN)
    lines, _, _ = fv2.input_lines(SESSION)
    f = fv2.json_prefix_features(lines, None, 3)          # A3: no login time
    ref = int(c.json_model.clf.predict([[f[n] for n in c.json_model.features]])[0])
    assert r["model"] == "A3" and r["verdict"] == ("Human" if ref == 1 else "Bot")
    assert 0.0 <= r["p_human"] <= 1.0 and r["n_lines"] == 4


def test_model_a_needs_the_login_time():
    c = Classifier(cfg("A"))
    r = c.classify(SESSION, None)
    assert r["verdict"] == "Uncertain" and r["reason"] == "missing: log_first_latency"
    assert c.classify(SESSION, LOGIN)["verdict"] in ("Human", "Bot")


def test_b_without_keystrokes_falls_back_to_json_model():
    c = Classifier(cfg(use_b=True))
    assert c.b_model is not None and c.b_model.name == "B"
    r = c.classify(SESSION, LOGIN, tty_records=None)
    assert r["model"] == "A3"


def test_wrong_sha_is_refused():
    refused(cfg(sha="0" * 63 + "1"), "SHA-256 mismatch")


def test_missing_sha_is_refused():
    refused(cfg(sha=""), "missing in config.yaml")


def test_wrong_k_is_refused():
    refused(cfg(k=4), "was frozen at k = 3")


def test_tty_model_as_json_model_is_refused():
    refused(cfg("B"), "JSON-only")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS {t.__name__}")
    print(f"[+] {len(tests)} tests passed")
