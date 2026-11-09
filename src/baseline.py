# src/baseline.py  (run from the project root:  python src/baseline.py)
# Static RMS-threshold baseline, built the way Section 3.5.2 describes it:
#   thresholds come from Normal TRAINING windows only: mean + k1*SD and mean + k2*SD
#   below k1 -> Normal, between k1 and k2 -> Degraded, above k2 -> Fault
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# k values chosen BEFORE looking at results: 2 SD and 3 SD are the usual alarm / alert levels.
K1, K2 = 2, 3
FEAT = ['rms', 'peak', 'kurtosis', 'skewness', 'std', 'crest_factor',
        'dom_freq', 'spec_centroid', 'spec_energy', 'band_low', 'band_mid']

df = pd.read_csv('data/lab_features.csv')
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
y_test = test_df['label'].values

normal_rms = train_df.loc[train_df['label'] == 0, 'rms']      # Normal TRAINING data only
mu, sd = normal_rms.mean(), normal_rms.std()

def three_class(rms, k1, k2):
    t1, t2 = mu + k1 * sd, mu + k2 * sd
    return np.where(rms <= t1, 0, np.where(rms <= t2, 1, 2))

pred = three_class(test_df['rms'].values, K1, K2)
print(f"=== STATIC RMS THRESHOLD BASELINE (3-class, k1={K1}, k2={K2}) ===")
print(f"Thresholds: {mu + K1*sd:.4f} and {mu + K2*sd:.4f}  (Normal training RMS mean={mu:.4f}, SD={sd:.4f})")
print(f"Accuracy      : {accuracy_score(y_test, pred)*100:.2f}%")
print(f"Weighted F1   : {f1_score(y_test, pred, average='weighted'):.4f}")
print(classification_report(y_test, pred, target_names=['Normal', 'Degraded', 'Fault'], zero_division=0))
print(confusion_matrix(y_test, pred))

# Binary view (Normal vs not Normal) with ONE threshold, same task for baseline and RF
bin_pred = (test_df['rms'].values > mu + K1 * sd).astype(int)
rf     = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')
X_test_scaled = scaler.transform(test_df[FEAT].values)
rf_pred = rf.predict(X_test_scaled)
print("\n=== SAME TASK, BINARY (Normal vs not Normal) ===")
print(f"Threshold baseline: acc {accuracy_score(y_test > 0, bin_pred)*100:.2f}%  F1 {f1_score(y_test > 0, bin_pred):.4f}")
print(f"Random Forest     : acc {accuracy_score(y_test > 0, rf_pred > 0)*100:.2f}%  F1 {f1_score(y_test > 0, rf_pred > 0):.4f}")
print("\n=== THREE-CLASS COMPARISON ===")
print(f"Threshold baseline: acc {accuracy_score(y_test, pred)*100:.2f}%  weighted F1 {f1_score(y_test, pred, average='weighted'):.4f}")
print(f"Random Forest     : acc {accuracy_score(y_test, rf_pred)*100:.2f}%  weighted F1 {f1_score(y_test, rf_pred, average='weighted'):.4f}")

# Sensitivity: how much does the baseline depend on k?
rows = []
for k1, k2 in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5)]:
    p = three_class(test_df['rms'].values, k1, k2)
    rows.append({"k1": k1, "k2": k2, "accuracy_%": round(accuracy_score(y_test, p)*100, 2),
                 "weighted_F1": round(f1_score(y_test, p, average='weighted'), 4)})
sens = pd.DataFrame(rows)
print("\n=== SENSITIVITY TO k (report this range) ===")
print(sens.to_string(index=False))
sens.to_csv('outputs/baseline_sensitivity.csv', index=False)

# ── Inference latency (unchanged from the earlier version) ──
latencies = []
for sample in X_test_scaled:
    start = time.perf_counter()
    rf.predict(sample.reshape(1, -1))
    latencies.append((time.perf_counter() - start) * 1000)
avg, mn, mx = np.mean(latencies), np.min(latencies), np.max(latencies)
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(latencies) + 1), latencies, color='steelblue', linewidth=0.8, label='Latency')
plt.axhline(avg, color='orange', linestyle='--', label=f'Avg: {avg:.2f} ms')
plt.axhline(mn, color='green', linestyle='--', label=f'Min: {mn:.2f} ms')
plt.axhline(mx, color='red', linestyle='--', label=f'Max: {mx:.2f} ms')
plt.xlabel('Test Iteration'); plt.ylabel('Latency (ms)')
plt.title('ML Inference Latency Across Test Iterations')
plt.legend(); plt.tight_layout()
plt.savefig('outputs/inference_latency.png', dpi=300)
