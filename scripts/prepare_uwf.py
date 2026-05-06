import pandas as pd

df1 = pd.read_csv("data/external/uwf_conn.csv")
df2 = pd.read_csv("data/external/uwf_conn2.csv")

df = pd.concat([df1, df2])
df = df[[
    "duration",
    "orig_pkts",
    "resp_pkts",
    "orig_bytes",
    "resp_bytes",
    "mitre_attack_tactics"
]]

df = df.dropna()

def label_map(x):
    if x == "Reconnaissance":
        return 0   
    else:
        return 1    

df["label"] = df["mitre_attack_tactics"].apply(label_map)

df_final = pd.DataFrame()

df_final["command_count"] = df["orig_pkts"]
df_final["login_attempts"] = df["resp_pkts"]
df_final["unique_passwords"] = df["orig_bytes"]

df_final["label"] = df["label"]

df_final.to_csv("data/processed/uwf_dataset.csv", index=False)

print("✅ UWF dataset prepared")
