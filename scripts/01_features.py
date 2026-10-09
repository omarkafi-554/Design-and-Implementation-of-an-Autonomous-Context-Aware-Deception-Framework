#!/usr/bin/env python3
"""01_features.py — build one feature row per Cowrie session from raw logs."""
import argparse
import glob
import json
import os
import struct

import numpy as np
import pandas as pd

_HDR = "<iLiiLL"
_HSZ = struct.calcsize(_HDR)
_OP_WRITE = 3
_DIR_INPUT = 1


def _read_tty(path):
    try:
        data = open(path, "rb").read()
    except OSError:
        return []
    out, i = [], 0
    while i + _HSZ <= len(data):
        op, _tty, length, dir_, sec, usec = struct.unpack(_HDR, data[i:i + _HSZ])
        i += _HSZ
        payload = data[i:i + length]
        i += length
        if op == _OP_WRITE and dir_ == _DIR_INPUT:
            out.append((sec + usec / 1e6, payload))
    return out


def _tty_features(path):
    inp = _read_tty(path)
    if not inp:
        return {}
    t = np.array([x[0] for x in inp])
    sizes = np.array([len(x[1]) for x in inp])
    gaps = np.diff(t)
    intra = [g for g, (_, p) in zip(gaps, inp[:-1]) if b"\r" not in p and b"\n" not in p]
    return {
        "tty_input_chunks": len(inp),
        "single_char_ratio": float((sizes == 1).mean()),
        "max_chunk_bytes": int(sizes.max()),
        "backspaces": int(sum(p.count(b"\x7f") + p.count(b"\x08") for _, p in inp)),
        "keystroke_gap_median": float(np.median(intra)) if intra else 0.0,
    }


def _load_events(cowrie_dir):
    ev = []
    for f in sorted(glob.glob(os.path.join(cowrie_dir, "cowrie.json*"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    df = pd.DataFrame(ev)
    df["ts"] = pd.to_datetime(df.timestamp, utc=True, format="ISO8601")
    return df


def _icd(times):
    if len(times) < 2:
        return 0.0, 0.0
    d = np.diff(sorted(t.timestamp() for t in times))
    d = d[(d > 0) & (d < 300)]
    if len(d) == 0:
        return 0.0, 0.0
    return float(d.mean()), float(d.std())


def build(cowrie_dir, tty_dir):
    df = _load_events(cowrie_dir)
    tty_map = {}
    for _, r in df[df.eventid == "cowrie.log.closed"].iterrows():
        tty_map[os.path.basename(r["ttylog"])] = r["session"]
    tty_feats = {}
    for p in glob.glob(os.path.join(tty_dir, "*")):
        b = os.path.basename(p)
        if b.startswith("."):
            continue
        sid = tty_map.get(b)
        if sid:
            tty_feats[sid] = _tty_features(p)
    rows = []
    for sid, g in df.groupby("session"):
        cmds = g[g.eventid == "cowrie.command.input"]
        login = g[g.eventid.isin(["cowrie.login.success", "cowrie.login.failed"])]
        username = login.username.dropna().iloc[0] if len(login.username.dropna()) else None
        closed = g[g.eventid == "cowrie.session.closed"]
        icd_mean, icd_std = _icd(list(cmds.ts))
        conn = g[g.eventid == "cowrie.session.connect"]
        row = {
            "session": sid,
            "src_ip": g.src_ip.iloc[0],
            "src_port": conn.src_port.iloc[0] if len(conn) and "src_port" in conn else None,
            "login_username": username,
            "start_utc": g.ts.min().isoformat(),
            "command_count": len(cmds),
            "unique_commands": cmds.input.nunique() if len(cmds) else 0,
            "failed_logins": int((g.eventid == "cowrie.login.failed").sum()),
            "success_logins": int((g.eventid == "cowrie.login.success").sum()),
            "session_duration": float(closed.duration.iloc[0]) if len(closed) else 0.0,
            "icd_mean": round(icd_mean, 4),
            "icd_std": round(icd_std, 4),
            "has_tty": int(sid in tty_feats),
            "single_char_ratio": 0.0, "max_chunk_bytes": 0,
            "backspaces": 0, "keystroke_gap_median": 0.0, "tty_input_chunks": 0,
        }
        row["command_diversity"] = round(row["unique_commands"] / row["command_count"], 4) if row["command_count"] else 0.0
        row.update(tty_feats.get(sid, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cowrie", required=True)
    ap.add_argument("--tty", required=True)
    ap.add_argument("--out", default="data/processed/sessions.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df = build(a.cowrie, a.tty)
    df.to_csv(a.out, index=False)
    print(f"[+] {len(df)} sessions -> {a.out}")


if __name__ == "__main__":
    main()
