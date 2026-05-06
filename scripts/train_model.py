
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (classification_report,
                                       confusion_matrix,
                                       roc_auc_score,
                                       roc_curve)
from sklearn.model_selection  import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing    import StandardScaler

DATASET_PATH = "data/processed/labeled_dataset.csv"
MODEL_PATH   = "models/rf_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
REPORTS_DIR  = "reports"

FEATURES = [
    "command_count", "unique_commands", "command_diversity",
    "failed_logins", "success_logins", "session_duration",
    "icd_mean", "icd_std"
]

def main():
    os.makedirs("models",    exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    X  = df[FEATURES]
    y  = df["label"]
    print(f"Dataset: {len(df)} sessions  |  Bot={( y==0).sum()}  Human={( y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train)
    X_te_s   = scaler.transform(X_test)

    param_grid = {
        "n_estimators"    : [50, 100, 200],
        "max_depth"       : [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    base = RandomForestClassifier(class_weight="balanced", random_state=42)
    grid = GridSearchCV(base, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
    grid.fit(X_tr_s, y_train)
    print(f"Best params: {grid.best_params_}")

    model  = grid.best_estimator_
    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Bot", "Human"]))
    auc = roc_auc_score(y_test, y_prob)
    print(f"AUC Score : {auc:.4f}")
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="f1_weighted")
    print(f"CV F1     : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nTP={cm[1,1]}  FP={cm[0,1]}  FN={cm[1,0]}  TN={cm[0,0]}")

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel  → {MODEL_PATH}")
    print(f"Scaler → {SCALER_PATH}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Bot", "Human"]); ax.set_yticklabels(["Bot", "Human"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(fpr, tpr, color="#1C2B5E", lw=2, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], "--", color="gray")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/roc_curve.png", dpi=180)
    plt.close()

    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    imp = imp.sort_values("importance")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    colors = ["#C0392B" if v == imp.importance.max() else "#1C2B5E" for v in imp.importance]
    ax3.barh(imp.feature, imp.importance, color=colors)
    ax3.set_xlabel("Importance Score"); ax3.set_title("Feature Importance")
    ax3.grid(axis="x", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=180)
    plt.close()

    bot   = df[df.label == 0]["icd_mean"]
    human = df[df.label == 1]["icd_mean"]
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.hist(bot[bot > 0],   bins=20, alpha=0.7, color="#C0392B", label="Bot")
    ax4.hist(human[human > 0], bins=15, alpha=0.7, color="#1C2B5E", label="Human")
    ax4.set_xlabel("ICD Mean (seconds)"); ax4.set_ylabel("Sessions")
    ax4.set_title("Inter-Command Delay: Bot vs Human")
    ax4.legend(); ax4.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/icd_comparison.png", dpi=180)
    plt.close()

    print(f"\nPlots saved to {REPORTS_DIR}/")

if __name__ == "__main__":
    main()
