"""
build_dataset.py
Engineers 8 behavioral features from Cowrie logs, labels each session
as Bot (0) or Human (1), and saves the labeled dataset.
Usage: python3 scripts/build_dataset.py
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

COWRIE_LOG_DIR = os.path.expanduser("~/cowrie/var/log/cowrie/")
OUTPUT_PATH    = "data/processed/labeled_dataset.csv"

ICD_THRESHOLD   = 4.0
CMD_THRESHOLD   = 2

def load_events(log_dir: str) -> list:
    events = []
    for filename in sorted(os.listdir(log_dir)):
        if not filename.startswith("cowrie.json"):
            continue
        filepath = os.path.join(log_dir, filename)
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events

def build_sessions(events: list) -> dict:
    sessions = {}
    for e in events:
        sid = e.get("session", "")
        if not sid:
            continue
        if sid not in sessions:
            sessions[sid] = {
                "src_ip"       : e.get("src_ip", ""),
                "commands"     : [],
                "cmd_times"    : [],
                "failed_logins": 0,
                "success_logins": 0,
                "duration"     : 0.0,
            }
        eid = e.get("eventid", "")
        ts  = e.get("timestamp", "")
        if eid == "cowrie.login.failed":
            sessions[sid]["failed_logins"] += 1
        elif eid == "cowrie.login.success":
            sessions[sid]["success_logins"] += 1
        elif eid == "cowrie.command.input":
            cmd = e.get("input", "").strip()
            if cmd:
                sessions[sid]["commands"].append(cmd)
                sessions[sid]["cmd_times"].append(ts)
        elif eid == "cowrie.session.closed":
            sessions[sid]["duration"] = float(e.get("duration", 0))
    return sessions

def compute_icd(times: list):
    """Returns (mean, std) of inter-command delays, excluding outliers."""
    if len(times) < 2:
        return 0.0, 0.0
    try:
        parsed = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times]
        delays = [(parsed[i+1] - parsed[i]).total_seconds()
                  for i in range(len(parsed) - 1)]
        delays = [d for d in delays if 0 < d < 300]
        if not delays:
            return 0.0, 0.0
        return float(np.mean(delays)), float(np.std(delays))
    except Exception:
        return 0.0, 0.0

def label(row) -> int:
    """Human = 1 if slow typing + multiple commands; Bot = 0 otherwise."""
    if row["icd_mean"] > ICD_THRESHOLD and row["command_count"] >= CMD_THRESHOLD:
        return 1
    return 0

def main():
    os.makedirs("data/processed", exist_ok=True)
    print("Loading Cowrie events ...")
    events   = load_events(COWRIE_LOG_DIR)
    sessions = build_sessions(events)
    print(f"Found {len(sessions)} unique sessions.")

    rows = []
    for sid, s in sessions.items():
        cmds            = s["commands"]
        command_count   = len(cmds)
        unique_commands = len(set(cmds))
        diversity       = unique_commands / command_count if command_count > 0 else 0.0
        icd_mean, icd_std = compute_icd(s["cmd_times"])
        rows.append({
            "session"          : sid,
            "src_ip"           : s["src_ip"],
            "command_count"    : command_count,
            "unique_commands"  : unique_commands,
            "command_diversity": round(diversity, 4),
            "failed_logins"    : s["failed_logins"],
            "success_logins"   : s["success_logins"],
            "session_duration" : round(s["duration"], 3),
            "icd_mean"         : round(icd_mean, 4),
            "icd_std"          : round(icd_std, 4),
        })

    df         = pd.DataFrame(rows)
    df["label"] = df.apply(label, axis=1)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nDataset saved to {OUTPUT_PATH}")
    print(f"Total sessions : {len(df)}")
    print(f"Bot  (label=0) : {(df.label == 0).sum()}")
    print(f"Human(label=1) : {(df.label == 1).sum()}")

if __name__ == "__main__":
    main()
