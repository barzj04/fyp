import pandas as pd
import glob
import os

lab_dir    = "data/lab/"
output_csv = "data/lab_combined.csv"

files = glob.glob(lab_dir + "*.csv")
files = sorted(files)

dfs = []
for f in files:
    df = pd.read_csv(f)
    print(f"Loaded: {os.path.basename(f)} — {len(df):,} samples")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(output_csv, index=False)

print(f"\n✅ Combined: {len(combined):,} total samples")
print(combined['label'].value_counts().sort_index()
      .rename({0:'Normal', 1:'Degraded', 2:'Fault'}))