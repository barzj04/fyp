# src/end_to_end_eval.py
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

rf     = joblib.load('models/rf_model.pkl')
iso    = joblib.load('models/iso_forest.pkl')
scaler = joblib.load('models/scaler.pkl')
df     = pd.read_csv('data/cwru_features.csv')

FEAT = ['rms','peak','kurtosis','skewness','std','crest_factor',
        'dom_freq','spec_centroid','spec_energy','band_low','band_mid']

_, X_test_df, _, y_test = train_test_split(
    df, df['label'], test_size=0.3, random_state=42, stratify=df['label'])

X_test = scaler.transform(X_test_df[FEAT].values)
y_test = y_test.values

# Full two-layer pipeline
final_preds = []
for i in range(len(X_test)):
    x = X_test[i:i+1]
    iso_pred = iso.predict(x)[0]
    
    if iso_pred == 1:  # Normal — IF says no anomaly
        final_preds.append(0)
    else:  # Anomaly — send to RF
        rf_pred = rf.predict(x)[0]
        final_preds.append(rf_pred)

final_preds = np.array(final_preds)

print("=== END-TO-END TWO-LAYER PIPELINE RESULTS ===")
print(classification_report(y_test, final_preds,
      target_names=['Normal','Degraded','Fault']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, final_preds))

# How many anomalies reached RF
anomalies = np.sum(iso.predict(X_test) == -1)
print(f"\nSamples flagged by IF: {anomalies}/{len(X_test)} ({round(anomalies/len(X_test)*100,1)}%)")
print(f"Samples passed directly as Normal: {len(X_test)-anomalies}")