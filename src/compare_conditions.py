import pandas as pd
import numpy as np
import glob
import os

files = glob.glob("data/lab/*.csv")

if not files:
    print("No lab data files found in data/lab/")
else:
    summary = []
    for f in sorted(files):
        df = pd.read_csv(f)
        summary.append({
            "file":    os.path.basename(f),
            "samples": len(df),
            "label":   df['label'].iloc[0],
            "ax_mean": round(df['ax'].mean(), 4),
            "ax_std":  round(df['ax'].std(), 4),
            "ax_max":  round(df['ax'].abs().max(), 4),
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    print("\nAverage by condition:")
    print(summary_df.groupby('label')[['ax_mean','ax_std','ax_max']].mean().round(4))