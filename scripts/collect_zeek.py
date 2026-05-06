import os
import pandas as pd

CONN_LOG    = "/opt/zeek/logs/current/conn.log"
OUTPUT_PATH = "data/raw/conn_raw.csv"

COLUMNS = [
    "ts", "uid", "orig_h", "orig_p", "resp_h", "resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes",
    "history", "orig_pkts", "orig_ip_bytes", "resp_pkts",
    "resp_ip_bytes", "tunnel_parents"
]

def main():
    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_csv(
        CONN_LOG,
        sep="\t",
        comment="#",
        names=COLUMNS,
        low_memory=False
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
