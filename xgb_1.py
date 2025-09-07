# XGB classifier intended to predict stocks likely to outperform. Run with output/ containing processed_dataset.csv and processed_2024_dataset.csv.

import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

INPUT_CSV = "output/processed_dataset.csv"
INPUT_2024_CSV = "output/processed_2024_dataset.csv"
OUTPUT_CSV = "output/xgb_1_predictions_2024.csv"

TARGET_PRECISION = 0.90
TARGET_F1 = 0.70
MIN_POS_FRAC_TRAIN = 0.002  
USE_CALIBRATION = True  
RANDOM_STATE = 42

def eval_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float):
    y_pred = (scores >= thr).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    k = int(y_pred.sum())
    return p, r, f, k

def pick_train_threshold(y_train: np.ndarray, s_train: np.ndarray) -> float:
    n = len(y_train)
    min_pos = int(np.ceil(MIN_POS_FRAC_TRAIN * n)) if MIN_POS_FRAC_TRAIN > 0 else 0

    uniq = np.unique(s_train)
    grid = np.linspace(0.01, 0.99, 199)
    qu = np.unique(np.quantile(s_train, q=np.linspace(0.50, 0.999, 200)))
    candidates = np.unique(np.clip(np.concatenate([uniq, grid, qu]), 1e-6, 1 - 1e-6))

    feasible, prec_ok, stats = [], [], []
    for t in candidates:
        p, r, f, k = eval_at_threshold(y_train, s_train, t)
        stats.append((t, p, r, f, k))
        if k >= min_pos:
            if p >= TARGET_PRECISION and f >= TARGET_F1:
                feasible.append((t, p, r, f, k))
            if p >= TARGET_PRECISION:
                prec_ok.append((t, p, r, f, k))

    if feasible:
        feasible.sort(key=lambda x: (x[2], x[0]), reverse=True)
        return float(feasible[0][0])

    if prec_ok:
        prec_ok.sort(key=lambda x: (x[3], x[2], x[0]), reverse=True)
        return float(prec_ok[0][0])

    prec_only = [row for row in stats if row[1] >= TARGET_PRECISION]
    if prec_only:
        prec_only.sort(key=lambda x: (x[3], x[2], x[0]), reverse=True)
        return float(prec_only[0][0])

    stats.sort(key=lambda x: (x[1], x[3], x[0]), reverse=True)
    return float(stats[0][0])

print("Loading datasets...")
df = pd.read_csv(INPUT_CSV)
df_2024 = pd.read_csv(INPUT_2024_CSV)

train_df = df[df["year"].between(2005, 2022)].copy()
test_df  = df[df["year"] == 2023].copy()

feature_cols = [c for c in train_df.columns if c not in ["ticker", "year", "beat_index"]]
X_train = train_df[feature_cols].values
y_train = train_df["beat_index"].astype(int).values
X_test  = test_df[feature_cols].values
y_test  = test_df["beat_index"].astype(int).values

neg = int((y_train == 0).sum()); pos = int((y_train == 1).sum())
scale_pos_weight = float(neg) / max(float(pos), 1.0)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=700,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    min_child_weight=1.0,
    gamma=0.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="aucpr",
)
model.fit(X_train_s, y_train)

if USE_CALIBRATION:
    cal = CalibratedClassifierCV(model, cv=5, method="isotonic")
    cal.fit(X_train_s, y_train)
    proba_train = cal.predict_proba(X_train_s)[:, 1]
    proba_test  = cal.predict_proba(X_test_s)[:, 1]
    scorer_for_2024 = cal
else:
    proba_train = model.predict_proba(X_train_s)[:, 1]
    proba_test  = model.predict_proba(X_test_s)[:, 1]
    scorer_for_2024 = model

print(f"PR-AUC (TRAIN): {average_precision_score(y_train, proba_train):.4f}")

print("Selecting decision threshold for Precision>=%.2f and F1>=%.2f..." % (TARGET_PRECISION, TARGET_F1))
thr = pick_train_threshold(y_train, proba_train)

p_tr, r_tr, f_tr, k_tr = eval_at_threshold(y_train, proba_train, thr)
train_pct = float((proba_train <= thr).mean() * 100.0)

print(f"Threshold: {thr:.6f}")
print(f"TRAIN -> Precision: {p_tr:.3f}, Recall: {r_tr:.3f}, F1: {f_tr:.3f}, Positives: {k_tr}")
print(f"Equivalent train percentile: top {100.0 - train_pct:.2f}% (>= {thr:.6f})")
print(f"Equivalent train Top-K: {k_tr} of {len(y_train)}")

p_te, r_te, f_te, k_te = eval_at_threshold(y_test, proba_test, thr)
print(f"TEST  -> Precision: {p_te:.3f}, Recall: {r_te:.3f}, F1: {f_te:.3f}, Positives: {k_te}")

X_2024 = df_2024[feature_cols].values
X_2024_s = scaler.transform(X_2024)
proba_2024 = scorer_for_2024.predict_proba(X_2024_s)[:, 1]

pred_2024 = pd.DataFrame({
    "ticker": df_2024["ticker"].values,
    "year": df_2024["year"].values,
    "winner_probability": proba_2024
})
top_50 = pred_2024.sort_values(by="winner_probability", ascending=False).head(50)
print("Top 50 predicted winners for 2024:")
print(top_50[["ticker", "winner_probability"]])

top_50.to_csv(OUTPUT_CSV, index=False)
print(f"Saved top 50 to {OUTPUT_CSV}")


