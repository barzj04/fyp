# src/session_eval.py  (run from the project root:  python src/session_eval.py)
# 1) Per-session table: recorded duration, raw samples, windows, train/test windows.
# 2) Session-wise check: hold out whole recording sessions, so no window from the
#    test session is seen in training (the main 70/30 split is random at window level).
import glob, os, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

FEAT = ["rms", "peak", "kurtosis", "skewness", "std", "crest_factor",
        "dom_freq", "spec_centroid", "spec_energy", "band_low", "band_mid"]
WINDOW, STEP = 256, 128

feat = pd.read_csv('data/lab_features.csv')
files = sorted(glob.glob('data/lab/*.csv'))                    # same files, same order as merge_lab_data.py
lens, meta = [], []
for f in files:
    d = pd.read_csv(f)
    ts = d['timestamp'].dropna().astype(float).values
    dt = np.diff(ts)
    lens.append(len(d))
    meta.append({"session": os.path.basename(f)[:-4], "label": int(d['label'].iloc[0]),
                 "raw_samples": len(d), "duration_s": round((ts[-1] - ts[0]) / 1000, 1),
                 "rate_hz": round(len(d) / ((ts[-1] - ts[0]) / 1000), 1),
                 "median_gap_ms": float(np.median(dt)), "longest_gap_s": round(dt.max() / 1000, 1)})
combined_rows = len(pd.read_csv('data/lab_combined.csv'))
assert sum(lens) == combined_rows, (f"data/lab has {sum(lens)} rows but lab_combined.csv has {combined_rows}. "
                                    "Move extra recordings (e.g. the Oct-15 files) out of data/lab/ first.")
starts = np.cumsum([0] + lens)

n = len(feat); y = feat['label'].values
ws = np.arange(n) * STEP; we = ws + WINDOW                       # window i covers samples ws..we of the joined stream
sess = np.full(n, -1)                                           # -1 = window spans two sessions
for k in range(len(files)):
    sess[(ws >= starts[k]) & (we <= starts[k + 1])] = k

# which windows the main 70/30 split put in train / test
tr_idx, te_idx = train_test_split(np.arange(n), test_size=0.3, random_state=42, stratify=y)
is_test = np.zeros(n, bool); is_test[te_idx] = True

for k, m in enumerate(meta):
    inside = sess == k
    m["windows"] = int(inside.sum()); m["train_windows"] = int((inside & ~is_test).sum()); m["test_windows"] = int((inside & is_test).sum())
tab = pd.DataFrame(meta)
span = sess == -1
total = {"session": "windows spanning two sessions", "windows": int(span.sum()),
         "train_windows": int((span & ~is_test).sum()), "test_windows": int((span & is_test).sum())}
tab = pd.concat([tab, pd.DataFrame([total])], ignore_index=True)
os.makedirs('outputs', exist_ok=True)
tab.to_csv('outputs/session_table.csv', index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 20)
print(tab.to_string(index=False))
print(f"\nTotal windows {n} | train {len(tr_idx)} | test {len(te_idx)}")

# ── Session-wise evaluation (scaler and SMOTE fitted on the training sessions only) ──
rf_ref = joblib.load('models/rf_model.pkl').get_params()
params = {k: rf_ref[k] for k in ("n_estimators", "max_depth", "min_samples_leaf")}
X = feat[FEAT].values
keep = sess >= 0

def run(folds, name):
    pred = np.full(n, -1)
    for held in folds:
        te = np.isin(sess, held); tr = keep & ~te
        sc = StandardScaler().fit(X[tr])
        Xb, yb = SMOTE(random_state=42).fit_resample(sc.transform(X[tr]), y[tr])
        rf = RandomForestClassifier(**params, class_weight="balanced", random_state=42).fit(Xb, yb)
        pred[te] = rf.predict(sc.transform(X[te]))
    m = keep
    print(f"\n{name}: accuracy {accuracy_score(y[m], pred[m])*100:.2f}%  weighted F1 {f1_score(y[m], pred[m], average='weighted'):.4f}")
    print(confusion_matrix(y[m], pred[m]))

run([[k] for k in range(len(files))], "Leave-one-session-out (9 folds)")
run([[j, 3 + j, 6 + j] for j in range(3)], "3 folds, one session of each class held out")
