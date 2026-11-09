# src/shap_bar.py
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

rf     = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')

FEAT = ['rms','peak','kurtosis','skewness','std','crest_factor',
        'dom_freq','spec_centroid','spec_energy','band_low','band_mid']
CLASS_NAMES = ['Normal', 'Degraded', 'Fault']

# ── Change this for each dataset ──────────────────────
# The saved models are trained on lab data, so keep this as "lab".
DATASET = "lab"
CSV     = {"lab": "data/lab_features.csv", "cwru": "data/cwru_features.csv"}[DATASET]

df = pd.read_csv(CSV)
X  = scaler.transform(df[FEAT].values)

explainer   = shap.TreeExplainer(rf)
shap_vals = explainer(X)

for k, cls in enumerate(CLASS_NAMES):
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_vals.values[:, :, k], X,
        feature_names=FEAT,
        plot_type="bar",
        show=False
    )
    plt.title(f"Global SHAP Feature Importance — {cls} ({DATASET.upper()})")
    plt.tight_layout()
    plt.savefig(f'outputs/{DATASET}_shap_bar_{cls.lower()}.png', dpi=150)
    plt.close()
    print(f"✅ Saved {DATASET}_shap_bar_{cls.lower()}.png")