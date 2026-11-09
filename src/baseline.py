import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

df = pd.read_csv('data/lab_features.csv')

_, X_test_df, _, y_test = train_test_split(
    df, df['label'], test_size=0.3, random_state=42, stratify=df['label'])

# Simple RMS threshold baseline
threshold = X_test_df['rms'].mean() + X_test_df['rms'].std()
baseline_pred = (X_test_df['rms'] > threshold).astype(int)

print("=== STATIC RMS THRESHOLD BASELINE ===")
print(f"Threshold value: {round(threshold, 4)}")
print(f"Baseline accuracy: {round(accuracy_score(y_test > 0, baseline_pred)*100, 2)}%")
print(f"Baseline F1: {round(f1_score(y_test > 0, baseline_pred), 4)}")
print()
print("=== YOUR RF MODEL (for comparison) ===")
import joblib
rf = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')
FEAT = ['rms','peak','kurtosis','skewness','std','crest_factor',
        'dom_freq','spec_centroid','spec_energy','band_low','band_mid']
X_test_scaled = scaler.transform(X_test_df[FEAT].values)
rf_pred = rf.predict(X_test_scaled)
import time
import matplotlib.pyplot as plt

latencies = []
for sample in X_test_scaled:
    start = time.perf_counter()
    rf.predict(sample.reshape(1, -1))
    end = time.perf_counter()
    latencies.append((end - start) * 1000)

avg = np.mean(latencies)
mn = np.min(latencies)
mx = np.max(latencies)

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(latencies)+1), latencies, color='steelblue', linewidth=0.8, label='Latency')
plt.axhline(avg, color='orange', linestyle='--', label=f'Avg: {avg:.2f} ms')
plt.axhline(mn, color='green', linestyle='--', label=f'Min: {mn:.2f} ms')
plt.axhline(mx, color='red', linestyle='--', label=f'Max: {mx:.2f} ms')
plt.xlabel('Test Iteration')
plt.ylabel('Latency (ms)')
plt.title('ML Inference Latency Across Test Iterations')
plt.legend()
plt.tight_layout()
plt.savefig('inference_latency.png', dpi=300)
plt.show()
print(f"RF accuracy: {round(accuracy_score(y_test, rf_pred)*100, 2)}%")
print(f"RF F1 (binary): {round(f1_score(y_test > 0, rf_pred > 0), 4)}")