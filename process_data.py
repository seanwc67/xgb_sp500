# Process yearly_dataset.csv. Run with output/ folder containing yearly_dataset.csv.

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

INPUT_CSV = "output/yearly_dataset.csv"
SPY_CSV = "output/spy_data.csv"
OUTPUT_CSV = "output/processed_dataset.csv"
OUTPUT_2024_CSV = "output/processed_2024_dataset.csv"
EXCLUDED_COLS = ["year", "ticker"]
PRICE_COL = ["price_change"]

def log_missing_pct(df, step_name):
    if df.empty:
        print(f"{step_name}: DataFrame is empty, missing %: 100%")
        return
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_pct = round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 100.0
    print(f"{step_name}: missing %: {missing_pct}%")

print("Loading yearly_dataset.csv")
df = pd.read_csv(INPUT_CSV).copy()
log_missing_pct(df, "After loading")

df = df.drop(columns=["price_begin", "price_end"], errors="ignore")
log_missing_pct(df, "After dropping price_begin/end")

print("Saving SPY data and removing from dataset")
spy_df = df[df["ticker"] == "SPY"][["year", "price_change"]]
spy_df[["year", "price_change"]].to_csv(SPY_CSV, index=False)
df = df[df["ticker"] != "SPY"]
log_missing_pct(df, "After removing SPY")

print("Removing ticker+year combos with missing price_change")
df = df[df["price_change"].notna()]
log_missing_pct(df, "After removing missing price_change")

print("Removing feature columns for specific years with >50% missing")
feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS + PRICE_COL]
for year in df["year"].unique():
    year_mask = df["year"] == year
    if not df[year_mask].empty:
        for col in feature_cols[:]:
            if col in df.columns:
                missing_rate = df.loc[year_mask, col].isna().mean()
                if missing_rate > 0.5 or df.loc[year_mask, col].isna().all():
                    df.loc[year_mask, col] = np.nan
                    if df[col].isna().all():
                        df = df.drop(columns=[col])
                        feature_cols.remove(col)
log_missing_pct(df, "After removing feature+year combos")

print("Removing ticker+year combos with >50% missing")
for (ticker, year), group in df.groupby(["ticker", "year"]):
    if not group.empty:
        if group[feature_cols].isna().mean().mean() > 0.5:
            df = df[~((df["ticker"] == ticker) & (df["year"] == year))]
log_missing_pct(df, "After removing ticker+year combos")

total_cells = df.size
missing_cells = df.isna().sum().sum()
missing_pct = round(missing_cells / total_cells * 100, 2) if total_cells > 0 else 100.0
if missing_pct >= 5:
    print(f"After deletion missing %: {missing_pct}% >= 5%; stopping.")
    exit()
else:
    print(f"After deletion missing %: {missing_pct}% < 5%; continuing.")

    print("Filling missing with per-ticker median across years")
    for ticker in df["ticker"].unique():
        ticker_mask = df["ticker"] == ticker
        for col in feature_cols:
            if col in df.columns:
                median_val = df.loc[ticker_mask, col].median()
                df.loc[ticker_mask & df[col].isna(), col] = median_val
    log_missing_pct(df, "After per-ticker median fill")

    print("Filling remaining missing with global median per year")
    for year in df["year"].unique():
        year_mask = df["year"] == year
        if not df[year_mask].empty:
            for col in feature_cols:
                if col in df.columns:
                    global_median = df.loc[year_mask, col].median()
                    df.loc[year_mask & df[col].isna(), col] = global_median
    log_missing_pct(df, "After global median fill")

    print("Removing ticker+year combos with >10% z-scores > ±5")
    all_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
    deleted_ticker_years = []
    for (ticker, year), group in df.groupby(["ticker", "year"]):
        if not group.empty:
            extreme_count = 0
            year_mask = df["year"] == year
            for col in all_cols:
                if col in group.columns and group[col].dtype in ["float64", "int64"]:
                    mean_val = df.loc[year_mask, col].mean()
                    std_val = df.loc[year_mask, col].std()
                    if std_val > 0:
                        z_score = (group[col].iloc[0] - mean_val) / std_val
                        if abs(z_score) > 5:
                            extreme_count += 1
            extreme_pct = (extreme_count / len(all_cols)) * 100 if all_cols else 0
            if extreme_pct > 10:
                deleted_ticker_years.append((ticker, year))
                df = df[~((df["ticker"] == ticker) & (df["year"] == year))]
    log_missing_pct(df, "After removing extreme z-scores")

    print("Capping remaining extreme values to within z-scores of ±5")
    for col in all_cols:
        if col in df.columns and df[col].dtype in ["float64", "int64"]:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                z_scores = (df[col] - mean_val) / std_val
                valid_values = df[col][z_scores.abs() <= 5]
                max_val = valid_values.max() if not valid_values.empty else df[col].max()
                min_val = valid_values.min() if not valid_values.empty else df[col].min()
                df.loc[(z_scores > 5) & (df[col].notna()), col] = max_val
                df.loc[(z_scores < -5) & (df[col].notna()), col] = min_val
    log_missing_pct(df, "After capping extreme values")

    print("Adding label column beat_index")
    df["beat_index"] = np.nan

    spy_prices = pd.read_csv(SPY_CSV).set_index("year")["price_change"]
    for year in df["year"].unique():
        if year < 2024:
            next_year = year + 1
            if next_year in spy_prices.index:
                spy_change = spy_prices[next_year]
                year_mask = df["year"] == year
                df.loc[year_mask, "beat_index"] = (
                    (df.loc[year_mask, "price_change"] > spy_change).astype(int)
                )

    # Drop whole ticker–year combos if beat_index is entirely NaN (for pre-2024)
    deleted_ticker_years = []
    for (ticker, year), group in df.groupby(["ticker", "year"]):
        if year < 2024 and group["beat_index"].isna().all():
            deleted_ticker_years.append((ticker, year))
            df = df[~((df["ticker"] == ticker) & (df["year"] == year))]

    print(f"Deleted {len(deleted_ticker_years)} unlabeled ticker-year combos.")

    # Proper check: only consider pre-2024
    pre2024 = df[df["year"] < 2024]
    if not pre2024.empty:
        missing = pre2024["beat_index"].isna().mean() * 100
        if missing == 0:
            print("beat_index has no missing values prior to 2024")
        else:
            print(f"beat_index missing {missing:.2f}% prior to 2024")



    print("Removing price_change")
    df = df.drop(columns=PRICE_COL, errors="ignore")
    log_missing_pct(df, "After removing price_change")

    print("Splitting into 2005-2023 and 2024 datasets")
    df_main = df[df["year"].between(2005, 2023)]
    df_2024 = df[df["year"] == 2024]
    log_missing_pct(df_main, "2005-2023 dataset")
    log_missing_pct(df_2024, "2024 dataset")

    print("Checking for missing values")
    missing_found = False
    for df_subset, name in [(df_main, "2005-2023"), (df_2024, "2024")]:
        for col in df_subset.columns:
            if col == "beat_index" and name == "2024":
                continue 
            missing_rows = df_subset[df_subset[col].isna()][["ticker", "year"]]
            for _, row in missing_rows.iterrows():
                print(f"Missing value in {col} for {row['ticker']} in {row['year']} ({name})")
                missing_found = True
    if not missing_found:
        print("No missing values")

    print("Saving datasets")
    df_main.to_csv(OUTPUT_CSV, index=False)
    df_2024.to_csv(OUTPUT_2024_CSV, index=False)
    print(f"Saved processed dataset to {OUTPUT_CSV}")
    print(f"Saved 2024 dataset to {OUTPUT_2024_CSV}")