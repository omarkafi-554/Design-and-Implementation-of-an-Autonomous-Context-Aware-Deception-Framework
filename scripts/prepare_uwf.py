import pandas as pd
df1 = pd.read_csv('data/external/uwf_conn.csv')
df2 = pd.read_csv('data/external/uwf_conn2.csv')
df = pd.concat([df1, df2])
df = df[['duration', 'orig_pkts', 'resp_pkts', 'orig_bytes', 'resp_bytes', 'mitre_attack_tactics']].dropna()

def label_map(x):
    return 0 if x == 'Reconnaissance' else 1
df['label'] = df['mitre_attack_tactics'].apply(label_map)
df_final = pd.DataFrame()
df_final['command_count'] = df['orig_pkts']
df_final['session_duration'] = df['duration']
df_final['icd_mean'] = df['duration'] / df['orig_pkts'].replace(0, 1)
df_final['unique_commands'] = df['orig_pkts']
df_final['command_diversity'] = 1.0
df_final['failed_logins'] = 0
df_final['success_logins'] = 0
df_final['icd_std'] = 0.0
df_final['label'] = df['label']
df_final.to_csv('data/processed/uwf_dataset_full.csv', index=False)
print(f'✅ UWF full-feature dataset prepared: {len(df_final)} rows')
print(df_final.head())