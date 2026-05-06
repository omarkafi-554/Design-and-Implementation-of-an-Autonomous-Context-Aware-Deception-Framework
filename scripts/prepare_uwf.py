import pandas as pd

# تحميل dataset
df1 = pd.read_csv("data/external/uwf_conn.csv")
df2 = pd.read_csv("data/external/uwf_conn2.csv")

df = pd.concat([df1, df2])
# اختيار الأعمدة المهمة
df = df[[
    "duration",
    "orig_pkts",
    "resp_pkts",
    "orig_bytes",
    "resp_bytes",
    "mitre_attack_tactics"
]]

# حذف القيم الفارغة
df = df.dropna()

# 🧠 تحويل label
def label_map(x):
    if x == "Reconnaissance":
        return 0  # Bot
    else:
        return 1  # Human (تقريب)

df["label"] = df["mitre_attack_tactics"].apply(label_map)

# 🧠 إنشاء features مشابهة لمشروعك
df_final = pd.DataFrame()

df_final["command_count"] = df["orig_pkts"]
df_final["login_attempts"] = df["resp_pkts"]
df_final["unique_passwords"] = df["orig_bytes"]

df_final["label"] = df["label"]

# حفظ
df_final.to_csv("data/processed/uwf_dataset.csv", index=False)

print("✅ UWF dataset prepared")
