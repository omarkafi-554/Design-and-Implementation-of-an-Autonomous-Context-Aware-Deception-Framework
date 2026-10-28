#!/usr/bin/env python3
"""06_external_validation.py — verify the external COW160x4 subset and run the
same-type external validation of the Bot/Human classifier.

What it does
  1. VERIFY   : streams the day files, checks schema / parse errors / completeness,
                and writes a data-quality report (external_verification.json).
  2. FEATURES : rebuilds the timing/command features with the SAME _icd() function as
                scripts/01_features.py, so lab and external features are identical.
  3. LABELS   : external sessions carry no ground truth.  A provenance-style proxy label
                that does NOT use any classifier feature is assigned:
                  campaign_bot = the exact normalised command sequence was replayed by
                                 >= K distinct source IPs inside the subset
                                 (scripted botnet behaviour).
                  unattributed = everything else (bots AND possible humans; not scored).
  4. VALIDATE : trains every model of 03_evaluate.py on the 52 lab interactive sessions,
                using the TIMING feature set (the dataset has no TTY logs, so keystroke
                features cannot be measured), and scores it on campaign_bot sessions:
                  Bot recall  (= specificity, Human is the positive class)
                  false-Human rate with a 95 % Wilson interval.
                The FULL model is also run but reported as NOT informative (keystroke
                features are constant = 0 externally).
  5. SHIFT    : Kolmogorov-Smirnov distance per feature, lab vs external.

Usage
    python3 scripts/06_external_validation.py --external data/external/cow160 \
        --lab data/processed/labeled.csv --out reports/external
"""
import argparse
import glob
import gzip
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F01 = _load("features01", "01_features.py")     # same _icd() as the lab pipeline
E03 = _load("evaluate03", "03_evaluate.py")     # same models and feature lists

KEEP = {"cowrie.session.connect", "cowrie.session.closed", "cowrie.command.input",
        "cowrie.login.success", "cowrie.login.failed", "cowrie.client.size",
        "cowrie.log.closed"}
REQUIRED = {"eventid", "session"}


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.endswith(".gz") \
        else open(path, encoding="utf-8", errors="replace")


def _ts(ev):
    t = ev.get("ts") or ev.get("timestamp")
    return pd.Timestamp(t, tz="UTC") if t and "+" not in str(t) and not str(t).endswith("Z") \
        else pd.Timestamp(t)


def _iter(files, qc):
    for f in files:
        with _open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                qc["lines"] += 1
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    qc["malformed_lines"] += 1
                    continue
                if not REQUIRED.issubset(ev) or not ev.get("session"):
                    qc["missing_required"] += 1
                    continue
                yield ev


def pass1(files):
    """Data-quality counters + the set of sessions that contain at least one command."""
    qc = Counter()
    eid = Counter()
    connect, closed, with_cmd = set(), set(), set()
    seen = set()
    for ev in _iter(files, qc):
        e = ev["eventid"]
        eid[e] += 1
        if e not in KEEP:
            continue
        key = (ev["session"], e, ev.get("ts") or ev.get("timestamp"), ev.get("input"))
        h = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
        if h in seen:
            qc["duplicate_events"] += 1
        else:
            seen.add(h)
        if e == "cowrie.session.connect":
            connect.add(ev["session"])
        elif e == "cowrie.session.closed":
            closed.add(ev["session"])
        elif e == "cowrie.command.input":
            with_cmd.add(ev["session"])
    del seen
    qc["sessions_seen"] = len(connect | closed | with_cmd)
    qc["sessions_complete"] = len(connect & closed)
    qc["sessions_with_commands"] = len(with_cmd)
    qc["cmd_sessions_complete"] = len(with_cmd & connect & closed)
    return qc, eid, with_cmd & connect & closed


def pass2(files, keep_sessions):
    """Aggregate only the complete sessions that ran commands."""
    S = defaultdict(lambda: {"cmds": [], "duration": None, "pty": 0, "src_ip": None,
                             "config_group": None, "sensor": None, "user": None,
                             "failed": 0, "success": 0, "ttylog": 0, "ts_err": 0})
    qc = Counter()
    for ev in _iter(files, qc):
        sid = ev["session"]
        if sid not in keep_sessions:
            continue
        s = S[sid]
        e = ev["eventid"]
        s["src_ip"] = s["src_ip"] or ev.get("src_ip")
        s["config_group"] = s["config_group"] if s["config_group"] is not None else ev.get("group")
        s["sensor"] = s["sensor"] or ev.get("sensor")
        if e == "cowrie.command.input":
            try:
                s["cmds"].append((_ts(ev), (ev.get("input") or "").strip()))
            except (ValueError, TypeError):
                s["ts_err"] += 1
        elif e == "cowrie.session.closed":
            s["duration"] = float(ev.get("duration") or 0.0)
        elif e == "cowrie.client.size":
            s["pty"] = 1
        elif e == "cowrie.login.success":
            s["success"] += 1
            s["user"] = s["user"] or ev.get("username")
        elif e == "cowrie.login.failed":
            s["failed"] += 1
        elif e == "cowrie.log.closed":
            s["ttylog"] = 1
    return S


def features(S):
    rows = []
    for sid, s in S.items():
        cmds = sorted(s["cmds"], key=lambda x: x[0])
        times = [t for t, _ in cmds]
        inputs = [c for _, c in cmds]
        icd_mean, icd_std = F01._icd(times)          # identical to lab pipeline
        n = len(inputs)
        u = len(set(inputs))
        seq = "\n".join(" ".join(c.split()) for c in inputs)
        rows.append({
            "session": sid, "src_ip": s["src_ip"], "config_group": s["config_group"],
            "sensor": s["sensor"], "login_username": s["user"], "pty_requested": s["pty"],
            "ttylog_event": s["ttylog"], "ts_errors": s["ts_err"],
            "command_count": n, "unique_commands": u,
            "command_diversity": round(u / n, 4) if n else 0.0,
            "icd_mean": round(icd_mean, 4), "icd_std": round(icd_std, 4),
            "session_duration": s["duration"] or 0.0,
            # no TTY recordings are published with the dataset:
            "has_tty": 0, "single_char_ratio": 0.0, "keystroke_gap_median": 0.0,
            "backspaces": 0,
            "seq_hash": hashlib.sha1(seq.encode()).hexdigest()[:16],
        })
    return pd.DataFrame(rows)


def proxy_label(df, k):
    ips = df.groupby("seq_hash").src_ip.nunique()
    df["seq_ip_count"] = df.seq_hash.map(ips)
    df["proxy_label"] = np.where(df.seq_ip_count >= k, "campaign_bot", "unattributed")
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def lab_frame(path):
    df = pd.read_csv(path)
    df = df[(df.labeled) & (df.population == "interactive")].copy()
    for c in E03.FULL:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def validate(lab, ext, populations):
    res = {}
    for fname, feats in [("TIMING_only", E03.TIMING), ("FULL", E03.FULL)]:
        res[fname] = {}
        for mname, make in E03._models().items():
            clf = make().fit(lab[feats].values, lab.is_human.values)
            r = {}
            for pname, mask in populations.items():
                sub = ext[mask]
                if not len(sub):
                    r[pname] = {"n": 0}
                    continue
                pred = clf.predict(sub[feats].values)
                n_h = int(pred.sum())
                n = len(pred)
                r[pname] = {"n": n, "predicted_bot": n - n_h, "predicted_human": n_h,
                            "bot_recall": round((n - n_h) / n, 4),
                            "false_human_rate": round(n_h / n, 4),
                            "false_human_rate_95ci": wilson(n_h, n)}
                if pname.startswith("campaign_bot") and "config_group" in sub:
                    r[pname]["false_human_rate_by_config"] = {
                        str(g): round(float(p.mean()), 4)
                        for g, p in pd.Series(pred, index=sub.index).groupby(sub.config_group)}
            res[fname][mname] = r
    res["FULL"]["_note"] = ("Keystroke features are constant (0) on this dataset because no "
                            "TTY recordings are published; FULL-model results are therefore "
                            "driven by missing TTY data, not by behaviour, and are NOT used "
                            "as evidence.")
    return res


def shift(lab, ext_bot):
    out = {}
    for f in E03.TIMING:
        a, b = lab[lab.is_human == 0][f], lab[lab.is_human == 1][f]
        e = ext_bot[f]
        out[f] = {"median_lab_bot": round(float(a.median()), 3),
                  "median_lab_human": round(float(b.median()), 3),
                  "median_ext_campaign_bot": round(float(e.median()), 3),
                  "KS_ext_vs_lab_bot": round(float(ks_2samp(e, a).statistic), 3),
                  "KS_ext_vs_lab_human": round(float(ks_2samp(e, b).statistic), 3)}
    return out


def figures(lab, ext, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.logspace(-3, np.log10(300), 40)
    plt.figure(figsize=(7, 4))
    groups = [(lab[lab.is_human == 1].icd_mean, "Lab Human (p1-p10)", "#1C2B5E"),
              (lab[lab.is_human == 0].icd_mean, "Lab Bot (bot1-bot4)", "#C0392B"),
              (ext[ext.proxy_label == "campaign_bot"].icd_mean, "External campaign bots", "#95A5A6")]
    for v, lbl, c in groups:
        v = v[v > 0]
        if len(v):
            plt.hist(v, bins=bins, alpha=0.55, color=c, label=lbl, density=True)
    plt.xscale("log")
    plt.xlabel("Mean inter-command delay (s, log scale)")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}/fig_ext_icd_shift.png", dpi=160)
    plt.close()
    plt.figure(figsize=(7, 4))
    cb = ext[ext.proxy_label == "campaign_bot"].command_count.clip(upper=40)
    plt.hist(cb, bins=range(0, 42), color="#95A5A6", alpha=0.8, label="External campaign bots",
             density=True)
    plt.axvline(9, color="#C0392B", ls="--", label="Every lab bot (9 commands)")
    plt.xlabel("Commands per session (clipped at 40)")
    plt.ylabel("Density")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out}/fig_ext_command_count.png", dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True, help="dir with day files (*.json.gz / *.json)")
    ap.add_argument("--lab", default="data/processed/labeled.csv")
    ap.add_argument("--out", default="reports/external")
    ap.add_argument("--k", type=int, default=5, help="min distinct IPs for campaign_bot")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.external, "**", "*.json*"), recursive=True))
    files = [f for f in files if not f.endswith("MANIFEST.json")]
    if not files:
        raise SystemExit("[-] no *.json(.gz) files found")

    qc, eid, keep = pass1(files)
    S = pass2(files, keep)
    ext = proxy_label(features(S), a.k)
    ext.to_csv(f"{a.out}/external_sessions.csv", index=False)

    lab = lab_frame(a.lab)
    # Populations. The PTY filter defined first was dropped after data verification:
    # cowrie.client.size (PTY request) occurs in only ~0.03 % of sessions in COW160x4,
    # so it cannot identify shell sessions. ">= 2 commands" is used instead: it is the
    # minimum for an inter-command delay to exist, and excludes single exec commands.
    camp = ext.proxy_label == "campaign_bot"
    pops = {
        "campaign_bot__ge2cmd": camp & (ext.command_count >= 2),          # primary
        "campaign_bot__ge4cmd": camp & (ext.command_count >= 4),          # lab range (4-9+)
        "campaign_bot__any_ge1cmd": camp,                                 # all campaign bots
        "unattributed__ge2cmd": (~camp) & (ext.command_count >= 2),       # exploratory only
    }
    PRIMARY = "campaign_bot__ge2cmd"
    verification = {
        "files": [os.path.relpath(f, a.external) for f in files],
        "quality": dict(qc), "eventid_counts": dict(eid.most_common()),
        "unknown_eventids": sorted(e for e in eid if not e.startswith("cowrie.")),
        "features_built_for": len(ext),
        "sessions_with_ts_errors": int((ext.ts_errors > 0).sum()),
        "sessions_with_pty": int(ext.pty_requested.sum()),
        "client_size_events_per_connect": round(
            eid.get("cowrie.client.size", 0) / max(eid.get("cowrie.session.connect", 1), 1), 6),
        "ttylog_events_present": int(ext.ttylog_event.sum()),
        "proxy_label_counts": ext.proxy_label.value_counts().to_dict(),
        "population_sizes": {k: int(v.sum()) for k, v in pops.items()},
        "k_distinct_ips": a.k,
    }
    json.dump(verification, open(f"{a.out}/external_verification.json", "w"), indent=2,
              default=str)
    metrics = {"lab_training_set": {"n": int(len(lab)), "human": int(lab.is_human.sum()),
                                    "bot": int((lab.is_human == 0).sum())},
               "results": validate(lab, ext, pops),
               "primary_population": PRIMARY,
               "distribution_shift": shift(lab, ext[pops[PRIMARY]])}
    json.dump(metrics, open(f"{a.out}/external_metrics.json", "w"), indent=2, default=str)
    figures(lab, ext, a.out)

    print(f"[+] parsed {qc['lines']} lines, malformed {qc['malformed_lines']}, "
          f"complete command sessions {len(ext)}")
    print(f"    proxy labels: {verification['proxy_label_counts']}")
    print(f"    populations: {verification['population_sizes']}")
    for pname in ["campaign_bot__ge2cmd", "campaign_bot__ge4cmd", "campaign_bot__any_ge1cmd"]:
        print(f"  == {pname}")
        for m, r in metrics["results"]["TIMING_only"].items():
            p = r[pname]
            if p.get("n"):
                print(f"    TIMING {m:18s} n={p['n']:6d} bot_recall={p['bot_recall']:.3f}  "
                      f"false_human={p['false_human_rate']:.3f} CI={p['false_human_rate_95ci']}")
    print(f"[+] reports -> {a.out}/")


if __name__ == "__main__":
    main()
