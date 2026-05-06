import json
import joblib
import pandas as pd

# 📥 تحميل الموديل
model = joblib.load("models/model.pkl")

log_file = "/home/omar/cowrie/var/log/cowrie/cowrie.json"

sessions = {}

# 📥 قراءة اللوج
with open(log_file) as f:
    for line in f:
        try:
            log = json.loads(line)
        except:
            continue

        session = log.get("session")

        if session not in sessions:
            sessions[session] = {
                "command_count": 0,
                "login_attempts": 0,
                "unique_passwords": set()
            }

        if log.get("eventid") == "cowrie.command.input":
            sessions[session]["command_count"] += 1

        if "login" in str(log.get("eventid")):
            sessions[session]["login_attempts"] += 1
            pwd = log.get("password")
            if pwd:
                sessions[session]["unique_passwords"].add(pwd)

# 🧠 تحويل ل dataframe
data = []

for s, v in sessions.items():
    data.append({
        "command_count": v["command_count"],
        "login_attempts": v["login_attempts"],
        "unique_passwords": len(v["unique_passwords"])
    })

df = pd.DataFrame(data)

# 🤖 prediction
predictions = model.predict(df)

# 🖥️ عرض النتائج
for i, row in enumerate(data):
    label = "Human" if predictions[i] == 1 else "Bot"
    print(f"Session {i+1}: {label}")
