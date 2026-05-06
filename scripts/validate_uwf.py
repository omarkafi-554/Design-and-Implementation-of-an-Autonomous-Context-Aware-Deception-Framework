
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

UWF_PATH    = "data/processed/uwf_dataset.csv"
MODEL_PATH  = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"

FEATURES = [
    "command_count", "unique_commands", "command_diversity",
    "failed_logins", "success_logins", "session_duration",
    "icd_mean", "icd_std"
]

def label_map(tactic: str) -> int:
    """Reconnaissance → Bot (0); all other tactics → Human (1)."""
    return 0 if str(tactic).lower() == "reconnaissance" else 1

def main():
    if not os.path.exists(UWF_PATH):
        print(f"UWF dataset not found at {UWF_PATH}")
        print("Download UWF-ZeekData22 from https://datasets.uwf.edu and place the CSV there.")
        return

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = pd.read_csv(UWF_PATH)
    required = ["orig_pkts", "resp_pkts", "orig_bytes", "mitre_attack_tactics"]
    for col in required:
        if col not in df.columns:
            print(f"Missing column: {col}. Check dataset format.")
            return

    df = df[required].dropna()
    df["label"] = df["mitre_attack_tactics"].apply(label_map)

    max_bytes = df["orig_bytes"].max() if df["orig_bytes"].max() > 0 else 1
    feat = pd.DataFrame({
        "command_count"    : df["orig_pkts"],
        "unique_commands"  : df["resp_pkts"],
        "command_diversity": df["orig_bytes"] / max_bytes,
        "failed_logins"    : 0,
        "success_logins"   : 0,
        "session_duration" : 0,
        "icd_mean"         : 0,
        "icd_std"          : 0,
    })

    X_scaled = scaler.transform(feat[FEATURES])
    y_pred   = model.predict(X_scaled)
    y_true   = df["label"].values

    cm = confusion_matrix(y_true, y_pred)
    print("=== UWF-ZeekData22 Validation Results ===")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Bot  Predicted Human")
    print(f"  Actual Bot  [{cm[0,0]:10d}     {cm[0,1]:10d}  ]")
    print(f"  Actual Human[{cm[1,0]:10d}     {cm[1,1]:10d}  ]")
    print(f"\nTP={cm[0,0]}  FN={cm[0,1]}  FP={cm[1,0]}  TN={cm[1,1]}")
    print(f"\nFalse Negatives (missed attacks): {cm[0,1]}")
    print("\n" + classification_report(y_true, y_pred, target_names=["Bot","Human"]))

if __name__ == "__main__":
    main()
