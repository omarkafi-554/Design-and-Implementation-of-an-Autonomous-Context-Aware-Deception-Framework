
import argparse
import json
import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
COWRIE_LOG_DIR = os.path.expanduser('~/cowrie/var/log/cowrie/')
MODEL_PATH = 'models/rf_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
FEATURES = ['command_count', 'unique_commands', 'command_diversity', 'failed_logins', 'success_logins', 'session_duration', 'icd_mean', 'icd_std']

def load_events(log_dir: str) -> list:
    events = []
    for filename in sorted(os.listdir(log_dir)):
        if not filename.startswith('cowrie.json'):
            continue
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r') as f:
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
        sid = e.get('session', '')
        if not sid:
            continue
        if sid not in sessions:
            sessions[sid] = {'src_ip': e.get('src_ip', ''), 'commands': [], 'cmd_times': [], 'failed_logins': 0, 'success_logins': 0, 'duration': 0.0, 'last_ts': e.get('timestamp', '')}
        eid = e.get('eventid', '')
        ts = e.get('timestamp', '')
        if ts:
            sessions[sid]['last_ts'] = max(sessions[sid]['last_ts'], ts)
        if eid == 'cowrie.login.failed':
            sessions[sid]['failed_logins'] += 1
        elif eid == 'cowrie.login.success':
            sessions[sid]['success_logins'] += 1
        elif eid == 'cowrie.command.input':
            cmd = e.get('input', '').strip()
            if cmd:
                sessions[sid]['commands'].append(cmd)
                sessions[sid]['cmd_times'].append(ts)
        elif eid == 'cowrie.session.closed':
            sessions[sid]['duration'] = float(e.get('duration', 0))
    return sessions

def compute_icd(times: list):
    if len(times) < 2:
        return (0.0, 0.0)
    try:
        parsed = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in times]
        delays = [(parsed[i + 1] - parsed[i]).total_seconds() for i in range(len(parsed) - 1)]
        delays = [d for d in delays if 0 < d < 300]
        if not delays:
            return (0.0, 0.0)
        return (float(np.mean(delays)), float(np.std(delays)))
    except Exception:
        return (0.0, 0.0)

def session_to_features(sid: str, s: dict) -> dict:
    cmds = s['commands']
    command_count = len(cmds)
    unique_commands = len(set(cmds))
    diversity = unique_commands / command_count if command_count > 0 else 0.0
    (icd_mean, icd_std) = compute_icd(s['cmd_times'])
    return {'session': sid, 'src_ip': s['src_ip'], 'command_count': command_count, 'unique_commands': unique_commands, 'command_diversity': round(diversity, 4), 'failed_logins': s['failed_logins'], 'success_logins': s['success_logins'], 'session_duration': round(s['duration'], 3), 'icd_mean': round(icd_mean, 4), 'icd_std': round(icd_std, 4), 'last_ts': s['last_ts']}

def main():
    parser = argparse.ArgumentParser(description='Live Bot/Human classification of a Cowrie session')
    parser.add_argument('--session', default=None, help='Specific session ID (default: most recent)')
    args = parser.parse_args()
    print('Loading trained model...')
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print('Reading Cowrie logs...')
    events = load_events(COWRIE_LOG_DIR)
    sessions = build_sessions(events)
    if not sessions:
        print('No sessions found in cowrie.json.')
        return
    if args.session:
        if args.session not in sessions:
            print(f"Session '{args.session}' not found.")
            return
        target_sid = args.session
    else:
        target_sid = max(sessions, key=lambda sid: sessions[sid]['last_ts'])
    row = session_to_features(target_sid, sessions[target_sid])
    print('\n' + '=' * 55)
    print(f"Session   : {row['session']}")
    print(f"Source IP : {row['src_ip']}")
    print(f"Last seen : {row['last_ts']}")
    print('-' * 55)
    for feat in FEATURES:
        print(f'  {feat:<20}: {row[feat]}')
    print('=' * 55)
    X = pd.DataFrame([row])[FEATURES]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    label = 'HUMAN' if pred == 1 else 'BOT'
    confidence = proba[pred] * 100
    print(f'\n>>> CLASSIFICATION RESULT: {label}  (confidence: {confidence:.1f}%)')
    print(f'    P(Bot)={proba[0] * 100:.1f}%   P(Human)={proba[1] * 100:.1f}%')
    print('=' * 55 + '\n')
if __name__ == '__main__':
    main()