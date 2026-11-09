import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft, rfftfreq

# ── Config ─────────────────────────────────────────────
#INPUT_CSV  = "data/cwru_raw_labelled.csv"
#OUTPUT_CSV = "data/cwru_features.csv"
INPUT_CSV  = "data/lab_combined.csv"
OUTPUT_CSV = "data/lab_features.csv"
WINDOW     = 256
STEP       = 128   # 50% overlap
FS         = 12000

# ── Feature extraction per window ──────────────────────
def extract_features(w):
    w = np.array(w)
    
    # Time domain
    rms      = np.sqrt(np.mean(w**2))
    peak     = np.max(np.abs(w))
    kurt     = kurtosis(w)
    skewness = skew(w)
    std      = np.std(w)
    crest    = peak / rms if rms > 0 else 0

    # Frequency domain
    fft_vals = np.abs(rfft(w))
    freqs    = rfftfreq(len(w), d=1/FS)
    dom_freq  = freqs[np.argmax(fft_vals)]
    spec_cent = np.sum(freqs * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0
    spec_enrg = np.sum(fft_vals**2)
    band_low  = np.sum(fft_vals[(freqs >= 0)    & (freqs < 1000)]**2)
    band_mid  = np.sum(fft_vals[(freqs >= 1000) & (freqs < 5000)]**2)

    return [rms, peak, kurt, skewness, std, crest,
            dom_freq, spec_cent, spec_enrg, band_low, band_mid]

COLS = ["rms","peak","kurtosis","skewness","std","crest_factor",
        "dom_freq","spec_centroid","spec_energy","band_low","band_mid"]

# ── Main ───────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
rows = []

# Handle both CWRU (has 'source' column) and lab data (no 'source' column)
if "source" in df.columns:
    groups = df.groupby("source")
    for source, group in groups:
        signal = group["amplitude"].values
        labels = group["label"].values
        i = 0
        while i + WINDOW <= len(signal):
            feats = extract_features(signal[i:i+WINDOW])
            rows.append(feats + [int(labels[i]), source])
            i += STEP
else:
    # Lab data — use ax column, no source grouping
    signal = df["ax"].values
    labels = df["label"].values
    print(f"Processing lab data: {len(signal):,} samples")
    i = 0
    while i + WINDOW <= len(signal):
        feats = extract_features(signal[i:i+WINDOW])
        rows.append(feats + [int(labels[i]), "lab"])
        i += STEP

feat_df = pd.DataFrame(rows, columns=COLS + ["label", "source"])
feat_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Feature matrix: {feat_df.shape}")
print("\nClass distribution:")
print(feat_df["label"].value_counts().sort_index().rename({0:"Normal", 1:"Degraded", 2:"Fault"}))