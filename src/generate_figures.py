import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Class distribution charts ──────────────────────
for name, path in [("CWRU", "data/cwru_features.csv"), 
                    ("Lab", "data/lab_features.csv")]:
    df = pd.read_csv(path)
    counts = df['label'].value_counts().sort_index()
    labels = ['Normal', 'Degraded', 'Fault']
    colors = ['#1D9E75', '#BA7517', '#A32D2D']
    
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, counts.values, color=colors, edgecolor='white')
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha='center', fontsize=10)
    plt.title(f'{name} Dataset — Class Distribution')
    plt.ylabel('Number of Feature Windows')
    plt.tight_layout()
    plt.savefig(f'outputs/{name.lower()}_class_distribution.png', dpi=150)
    plt.close()
    print(f"✅ {name} class distribution saved")

# ── 2. Algorithm comparison charts ────────────────────
for name, path in [("CWRU", "outputs/cwru_algorithm_comparison.csv"),
                    ("Lab", "outputs/lab_algorithm_comparison.csv")]:
    try:
        df = pd.read_csv(path)
        colors = ['#1F3864' if i == 0 else '#AAAAAA' 
                  for i in range(len(df))]
        
        plt.figure(figsize=(8, 4))
        bars = plt.barh(df['Algorithm'], df['Accuracy(%)'], 
                       color=colors, edgecolor='white')
        for bar, val in zip(bars, df['Accuracy(%)']):
            plt.text(val - 0.8, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', ha='right',
                    color='white', fontsize=9, fontweight='bold')
        plt.xlabel('Accuracy (%)')
        plt.title(f'{name} — Algorithm Comparison')
        plt.xlim(60, 101)
        plt.tight_layout()
        plt.savefig(f'outputs/{name.lower()}_algorithm_comparison.png', dpi=150)
        plt.close()
        print(f"✅ {name} algorithm comparison chart saved")
    except:
        print(f"⚠️ {name} algorithm comparison CSV not found — skip")

print("\n✅ All figures generated → outputs/")