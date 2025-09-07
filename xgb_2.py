# XGB classifier intended to predict stocks likely to underperform. Run with output/ containing processed_dataset.csv and processed_2024_dataset.csv.

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import os

INPUT_CSV = "output/processed_dataset.csv"
INPUT_2024_CSV = "output/processed_2024_dataset.csv"
OUTPUT_CSV = "output/xgb_2_predictions_2024.csv"

print("Loading datasets...")
df = pd.read_csv(INPUT_CSV)
df_2024 = pd.read_csv(INPUT_2024_CSV)

train_df = df[df["year"].between(2005, 2022)]
test_df = df[df["year"] == 2023]

feature_cols = [c for c in train_df.columns if c not in ["ticker", "year", "beat_index"]]
if not feature_cols:
    raise ValueError("No feature columns found. Check processed_dataset.csv.")
X_train = train_df[feature_cols]
X_test = test_df[feature_cols]
y_train = 1 - train_df["beat_index"] 
y_test = 1 - test_df["beat_index"]
X_2024 = df_2024[feature_cols]

if y_train.isna().any() or y_test.isna().any():
    raise ValueError("Missing labels in train or test data. Check beat_index column.")

num_negative = sum(y_train == 0)  
num_positive = sum(y_train == 1)  
scale_pos_weight = num_negative / num_positive if num_positive > 0 else 1
print(f"Class imbalance ratio (negative/positive): {scale_pos_weight:.3f}")

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_2024_scaled = scaler.transform(X_2024)

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("Optimizing threshold for training F1...")
proba_train = model.predict_proba(X_train_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_train, proba_train)
f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if thresholds[best_idx] <= 1.0 else 0.5
best_f1 = f1_scores[best_idx]
print(f"Best training F1: {best_f1:.3f} at threshold {best_threshold:.3f}")

y_train_pred = (model.predict_proba(X_train_scaled)[:, 1] >= best_threshold).astype(int)
y_test_pred = (model.predict_proba(X_test_scaled)[:, 1] >= best_threshold).astype(int)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
print(f"Train F1 Score: {train_f1:.3f}")
print(f"Test F1 Score: {test_f1:.3f}")

if train_f1 <= 0.75 or test_f1 <= 0.75:
    print("Warning: F1 score <= 0.75 on train or test.")
else:
    print("F1 scores meets expectations.")
          
print("Predicting on 2024 data...")
proba_2024 = model.predict_proba(X_2024_scaled)[:, 1]
predictions_2024 = pd.DataFrame({
    "ticker": df_2024["ticker"],
    "year": df_2024["year"],
    "loser_probability": proba_2024
})

top_50_losers = predictions_2024.sort_values(by="loser_probability", ascending=False).head(50)
print("Top 50 predicted losers for 2024:")
print(top_50_losers[["ticker", "loser_probability"]])

print(f"Saving top 50 loser rankings to {OUTPUT_CSV}...")
top_50_losers.to_csv(OUTPUT_CSV, index=False)
print("Done")
