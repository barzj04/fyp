import pandas as pd
import numpy as np
import joblib
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────
INPUT_CSV = "data/lab_features.csv"
MODEL_DIR  = "models/"
FEAT_COLS  = ["rms","peak","kurtosis","skewness","std","crest_factor", "dom_freq","spec_centroid","spec_energy","band_low","band_mid"]

# ── Load data ──────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
X  = df[FEAT_COLS].values
y  = df["label"].values

# ── Scale ──────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, MODEL_DIR + "scaler.pkl")
print("✅ Scaler saved")

# ── Train/test split ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# ── SMOTE on training set only ─────────────────────────
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {pd.Series(y_train_bal).value_counts().sort_index().to_dict()}")

# ── Hyperparameter Tuning ──────────────────────────────  ← ADD HERE
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 5]
}
grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid.fit(X_train_bal, y_train_bal)
print("Best params:", grid.best_params_)
print("Best F1:", round(grid.best_score_, 4))

# Use best params for final RF
best_params = grid.best_params_

# ── Algorithm comparison ───────────────────────────────
algorithms = {
    "Random Forest":   RandomForestClassifier(**best_params, class_weight="balanced", random_state=42),
    "kNN (k=5)":       KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)":       SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    "Naive Bayes":     GaussianNB(),
    "Decision Tree":   DecisionTreeClassifier(class_weight="balanced", random_state=42),
}

print("\n" + "="*65)
print(f"{'Algorithm':<20} {'Accuracy':>10} {'F1 (weighted)':>15} {'Time(s)':>10}")
print("="*65)

results = []
for name, model in algorithms.items():
    start = time.time()
    model.fit(X_train_bal, y_train_bal)
    elapsed = round(time.time() - start, 2)
    
    y_pred   = model.predict(X_test)
    acc      = round(accuracy_score(y_test, y_pred) * 100, 2)
    f1       = round(f1_score(y_test, y_pred, average="weighted"), 4)
    
    print(f"{name:<20} {acc:>9}% {f1:>15} {elapsed:>9}s")
    results.append({"Algorithm": name, "Accuracy(%)": acc, "F1_weighted": f1, "Time(s)": elapsed})

print("="*65)

# ── Save comparison table ──────────────────────────────
pd.DataFrame(results).to_csv("outputs/algorithm_comparison.csv", index=False)
print("\n✅ Comparison table saved → outputs/algorithm_comparison.csv")

# ── Train final Random Forest + save ──────────────────
print("\nTraining final Random Forest...")
rf = RandomForestClassifier(**best_params, class_weight="balanced", random_state=42)
rf.fit(X_train_bal, y_train_bal)
joblib.dump(rf, MODEL_DIR + "rf_model.pkl")

print("\nRandom Forest — Detailed Report:")
print(classification_report(y_test, rf.predict(X_test),
      target_names=["Normal","Degraded","Fault"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf.predict(X_test)))

# ── Train Isolation Forest on Normal only ──────────────
print("\nTraining Isolation Forest on Normal data only...")
X_normal = X_scaled[y == 0]
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_normal)
joblib.dump(iso, MODEL_DIR + "iso_forest.pkl")

# Quick IF evaluation
scores = iso.decision_function(X_scaled)
preds  = iso.predict(X_scaled)   # 1=normal, -1=anomaly
anomaly_detected = (preds == -1)
actual_fault     = (y != 0)

tp  = np.sum(anomaly_detected & actual_fault)
fp  = np.sum(anomaly_detected & ~actual_fault)
fn  = np.sum(~anomaly_detected & actual_fault)
detection_rate = round(tp / (tp + fn) * 100, 2) if (tp+fn) > 0 else 0
fpr            = round(fp / (fp + np.sum(~anomaly_detected & ~actual_fault)) * 100, 2)

print(f"\nIsolation Forest Results:")
print(f"  Detection Rate : {detection_rate}%  (target > 90%)")
print(f"  False Positive : {fpr}%  (target < 10%)")

print("\n✅ All models saved to models/")

# ── Confusion Matrix ───────────────────────────────────
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, rf.predict(X_test))
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Degraded', 'Fault'],
            yticklabels=['Normal', 'Degraded', 'Fault'])
dataset_name = "Lab Dataset" if "lab" in INPUT_CSV else "CWRU Dataset"
plt.title(f'Random Forest Confusion Matrix — {dataset_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'outputs/{dataset_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=150)
plt.close()
print(f"✅ Confusion matrix saved → outputs/{dataset_name.lower().replace(' ', '_')}_confusion_matrix.png")