import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

# ── Config ─────────────────────────────────────────────
FEAT_COLS = ["rms","peak","kurtosis","skewness","std","crest_factor","dom_freq","spec_centroid","spec_energy","band_low","band_mid"]
LABELS    = {0: "Normal", 1: "Degraded", 2: "Fault"}

# 'lab' = deployed models (trained on lab data). 'cwru' = old benchmark data.
DATASET = "lab"
CSV = {"lab": "data/lab_features.csv", "cwru": "data/cwru_features.csv"}[DATASET]

# ── Load models + data ─────────────────────────────────
rf     = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
df     = pd.read_csv(CSV)

X = scaler.transform(df[FEAT_COLS].values)
y = df["label"].values

# ── SHAP TreeExplainer ─────────────────────────────────
print("Initialising SHAP TreeExplainer...")
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
print("✅ SHAP values computed")

# ── Global feature importance plot ─────────────────────
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=FEAT_COLS, class_names=["Normal","Degraded","Fault"], show=False)
plt.tight_layout()
plt.savefig("outputs/shap_global_summary.png", dpi=150)
plt.close()
print("✅ Global SHAP summary saved → outputs/shap_global_summary.png")

# ── Local explanation — one Fault sample ───────────────
fault_idx = np.where(y == 2)[0][0]
x_sample  = X[fault_idx:fault_idx+1]
pred      = rf.predict(x_sample)[0]
prob      = rf.predict_proba(x_sample)[0]

print(f"\nSample prediction: {LABELS[pred]} "f"({round(prob[pred]*100, 1)}% confidence)")

# Get SHAP values for single sample
shap_single = explainer(x_sample)

plt.figure(figsize=(10, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_single.values[0, :, pred],
        base_values=shap_single.base_values[0, pred],
        data=x_sample[0],
        feature_names=FEAT_COLS
    ),
    show=False
)
plt.tight_layout()
plt.savefig("outputs/shap_local_fault.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Local SHAP waterfall saved → outputs/shap_local_fault.png")

# ── Top features per class ─────────────────────────────
print("\nTop 5 features per class (mean |SHAP|):")
shap_all = explainer(X)
for i, cls in enumerate(["Normal","Degraded","Fault"]):
    mean_abs = np.abs(shap_all.values[:, :, i]).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:5]
    print(f"\n{cls}:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank}. {FEAT_COLS[idx]}: {round(mean_abs[idx], 4)}")