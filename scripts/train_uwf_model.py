import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# تحميل البيانات
# ---------------------------
df = pd.read_csv("data/processed/uwf_dataset.csv")

# ---------------------------
# Features و Label
# ---------------------------
X = df[["command_count", "login_attempts", "unique_passwords"]]
y = df["label"]

# ---------------------------
# تقسيم البيانات
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data split done")
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ---------------------------
# تدريب الموديل (مع معالجة imbalance)
# ---------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\n✅ Model trained")

# ---------------------------
# التوقع
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Evaluation
# ---------------------------
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

# ---------------------------
# استخراج TP / TN / FP / FN
# ---------------------------
# نعتبر Bot (0) هو Positive
TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]

print("\n📊 Correct Security Metrics:")
print(f"TP (Bot detected correctly): {TP}")
print(f"FN (Bot missed): {FN}")
print(f"FP (Human flagged as Bot): {FP}")
print(f"TN (Human detected correctly): {TN}")
