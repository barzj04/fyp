import scipy.io
import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt

# ── Config ────────────────────────────────────────────
DATA_DIR   = "data/cwru/"
OUTPUT_CSV = "data/cwru_raw_labelled.csv"
FS         = 12000   # CWRU sampling rate 12kHz
CUTOFF     = 5000    # low-pass cutoff Hz
ORDER      = 4

# ── File → label mapping ──────────────────────────────
FILES = {
    "Normal_0.mat": 0,  # Normal
    "IR007_0.mat":  1,  # Degraded
    "IR014_0.mat":  2,  # Fault
    "IR021_0.mat":  2,  # Fault
}

# ── Butterworth low-pass filter ───────────────────────
def butter_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)

# ── Load .mat file ─────────────────────────────────────
def load_mat(filepath):
    mat = scipy.io.loadmat(filepath)
    key = [k for k in mat.keys() if 'DE_time' in k][0]
    return mat[key].flatten()

# ── Main ──────────────────────────────────────────────
all_rows = []

for filename, label in FILES.items():
    path = os.path.join(DATA_DIR, filename)
    print(f"Loading {filename}...")
    
    signal = load_mat(path)
    signal = butter_filter(signal, CUTOFF, FS, ORDER)
    
    for val in signal:
        all_rows.append({
            "amplitude": round(float(val), 6),
            "label":     label,
            "source":    filename
        })

df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Done! Saved {len(df):,} samples → {OUTPUT_CSV}")
print("\nClass distribution:")
print(df["label"].value_counts().sort_index()
      .rename({0: "Normal", 1: "Degraded", 2: "Fault"}))