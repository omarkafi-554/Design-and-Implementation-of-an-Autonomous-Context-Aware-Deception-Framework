import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
FEATURES = ['command_count', 'unique_commands', 'command_diversity', 'failed_logins', 'success_logins', 'session_duration', 'icd_mean', 'icd_std']

def main():
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    df = pd.read_csv('data/processed/uwf_dataset_full.csv')
    X = df[FEATURES]
    y = df['label']
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    (TP, FN, FP, TN) = (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    print('=== UWF-ZeekData22 Validation — Real Internal Model ===')
    print(f"\n{'':15}Predicted Bot   Predicted Human")
    print(f'Actual Bot     {cm[0, 0]:10d}      {cm[0, 1]:10d}')
    print(f'Actual Human   {cm[1, 0]:10d}      {cm[1, 1]:10d}')
    print(f'\nTP={TP}  FN={FN}  FP={FP}  TN={TN}')
    print('\n' + classification_report(y, y_pred, target_names=['Bot', 'Human']))
if __name__ == '__main__':
    main()