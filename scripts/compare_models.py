import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("data/processed/uwf_dataset.csv")

X = df[["command_count", "login_attempts", "unique_passwords"]]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "SVM": SVC(class_weight="balanced")
}


for name, model in models.items():

    print(f"\n Model: {name}")
    print("-" * 40)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 Confusion Matrix:")
    print(cm)

    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

    print("\n📊 Security Metrics:")
    print(f"TP (Bot detected): {TP}")
    print(f"FN (Bot missed): {FN}")
    print(f"FP (False alarm): {FP}")
    print(f"TN (Human correct): {TN}")
