
import json
import os
import pandas as pd

COWRIE_LOG_DIR = os.path.expanduser("~/cowrie/var/log/cowrie/")
OUTPUT_PATH    = "data/raw/cowrie_raw.csv"

def collect_logs(log_dir: str) -> pd.DataFrame:
    records = []
    for filename in sorted(os.listdir(log_dir)):
        if not filename.startswith("cowrie.json"):
            continue
        filepath = os.path.join(log_dir, filename)
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return pd.DataFrame(records)

def main():
    os.makedirs("data/raw", exist_ok=True)
    df = collect_logs(COWRIE_LOG_DIR)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} events to {OUTPUT_PATH}")
    print("\nEvent type distribution:")
    print(df["eventid"].value_counts().to_string())

if __name__ == "__main__":
    main()
