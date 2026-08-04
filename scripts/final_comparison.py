import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
FEATURES = ['command_count', 'unique_commands', 'command_diversity', 'failed_logins', 'success_logins', 'session_duration', 'icd_mean', 'icd_std']

def bot_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def main():
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    internal = pd.read_csv('data/processed/labeled_dataset.csv')
    X_int = scaler.transform(internal[FEATURES])
    y_int = internal['label']
    recall_internal = bot_recall(y_int, model.predict(X_int))
    external = pd.read_csv('data/processed/uwf_dataset_full.csv')
    X_ext = scaler.transform(external[FEATURES])
    y_ext = external['label']
    recall_external = bot_recall(y_ext, model.predict(X_ext))
    labels = ['Internal\n(Cowrie, 360 sessions)', 'External\n(UWF-ZeekData22, 32,475 sessions)']
    values = [recall_internal, recall_external]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=['#1C2B5E', '#C0392B'])
    plt.ylim(0, 1.05)
    plt.ylabel('Bot Recall (Attack Detection Rate)')
    plt.title('Same Model: Internal vs. External Attack Detection')
    for (b, v) in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/final_comparison.png', dpi=180)
    print(f'Internal Bot Recall (FN=0 check): {recall_internal:.4f}')
    print(f'External Bot Recall (FN=0 check): {recall_external:.4f}')
    print('✅ Final comparison chart saved to reports/final_comparison.png')
if __name__ == '__main__':
    main()