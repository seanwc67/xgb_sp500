# Create dataset of S&P 500 constituents (2005-2024). Run with data/ and output/ folders present.

import pandas as pd
import numpy as np
import os

CONSTITUENTS_CSV = "data/sp500_constituents.csv"
PRICES_CSV = "data/prices_equity.csv"
PRICES_SPY_CSV = "data/prices_spy.csv"
DATASET_CSV = "output/yearly_dataset.csv"
EXCLUDED_COLS = ["ticker", "year", "date", "symbol"]

# CSV paths if loading
FUNDAMENTALS_CSVS = {
    "income_statement": "data/income_statement.csv",
    "balance_sheet": "data/balance_sheet.csv",
    "cash_flow": "data/cash_flow.csv",
    "ratios": "data/ratios.csv",
    "key_metrics": "data/key_metrics.csv",
    "income_growth": "data/income_growth.csv",
    "balance_growth": "data/balance_growth.csv",
    "cashflow_growth": "data/cashflow_growth.csv",
    "financial_growth": "data/financial_growth.csv",
}

# # --- Retrieval mode (uncomment to use, comment to load CSVs) ---
# FMP_API_KEY = "YOUR_API_KEY_HERE"  # Replace with a valid key from financialmodelingprep.com
# FMP_BASE = "https://financialmodelingprep.com/stable"
# SLEEP_BETWEEN_CALLS = 0.1  # Time between API calls (seconds)

# ENDPOINTS = [
#     {"name": "income_statement", "url": f"{FMP_BASE}/income-statement", "params": {"period": "annual", "limit": 100}},
#     {"name": "balance_sheet", "url": f"{FMP_BASE}/balance-sheet-statement", "params": {"period": "annual", "limit": 100}},
#     {"name": "cash_flow", "url": f"{FMP_BASE}/cash-flow-statement", "params": {"period": "annual", "limit": 100}},
#     {"name": "ratios", "url": f"{FMP_BASE}/ratios", "params": {"period": "annual", "limit": 100}},
#     {"name": "key_metrics", "url": f"{FMP_BASE}/key-metrics", "params": {"period": "annual", "limit": 100}},
#     {"name": "income_growth", "url": f"{FMP_BASE}/income-statement-growth", "params": {"period": "annual", "limit": 100}},
#     {"name": "balance_growth", "url": f"{FMP_BASE}/balance-sheet-statement-growth", "params": {"period": "annual", "limit": 100}},
#     {"name": "cashflow_growth", "url": f"{FMP_BASE}/cash-flow-statement-growth", "params": {"period": "annual", "limit": 100}},
#     {"name": "financial_growth", "url": f"{FMP_BASE}/financial-growth", "params": {"period": "annual", "limit": 100}},
# ]
#
# def fmp_get(url: str, params: dict, ticker: str = None) -> pd.DataFrame:
#     params = dict(params, apikey=FMP_API_KEY)
#     time.sleep(SLEEP_BETWEEN_CALLS)
#     try:
#         r = requests.get(url, params=params)
#         if r.status_code != 200:
#             print(f"Error for {ticker if ticker else 'constituents'}: HTTP {r.status_code}")
#             return pd.DataFrame()
#         data = r.json()
#         if isinstance(data, dict):
#             data = [data]
#         df = pd.DataFrame(data or [])
#         if df.empty:
#             print(f"No data for {ticker if ticker else 'constituents'}")
#             return pd.DataFrame()
#         if ticker:
#             df["ticker"] = ticker
#         return df
#     except Exception as e:
#         print(f"Exception for {ticker if ticker else 'constituents'}: {str(e)}")
#         return pd.DataFrame()
#
# print("Retrieving S&P 500 constituents")
# constituents = fmp_get(f"{FMP_BASE}/sp500-constituent", {})
# if constituents.empty:
#     print("Error: Failed to retrieve constituents, checking CSV")
#     if os.path.exists(CONSTITUENTS_CSV):
#         print("Loading sp500_constituents.csv")
#         try:
#             constituents = pd.read_csv(CONSTITUENTS_CSV)
#             tickers = sorted(constituents["symbol"].astype(str).str.upper().unique()) + ["SPY"]
#             print(f"Loaded {len(tickers)} tickers from CSV")
#         except Exception as e:
#             print(f"Error loading constituents CSV: {str(e)}")
#             tickers = ["SPY"]
#     else:
#         print("Error: No constituents data or CSV, using SPY only")
#         tickers = ["SPY"]
# else:
#     tickers = sorted(constituents["symbol"].astype(str).str.upper().unique()) + ["SPY"]
#     if os.path.exists(CONSTITUENTS_CSV):
#         print("Warning: sp500_constituents.csv already exists, overwriting")
#     constituents[["symbol"]].to_csv(CONSTITUENTS_CSV, index=False)
# print(f"Got {len(tickers)} tickers")
#
# print("Retrieving prices")
# price_frames = []
# total_tickers = len(tickers)
# for i, ticker in enumerate(tickers, 1):
#     df = fmp_get(f"{FMP_BASE}/historical-price-eod/dividend-adjusted", {"symbol": ticker, "from": "2005-01-01"}, ticker=ticker)
#     if not df.empty:
#         df["date"] = pd.to_datetime(df["date"], errors="coerce")
#         df = df[["ticker", "date", "adjClose"]].rename(columns={"adjClose": "adj_close"})
#         price_frames.append(df)
#     if i % 100 == 0 or i == total_tickers:
#         print(f"prices {i}/{total_tickers}")
# if price_frames:
#     prices = pd.concat(price_frames, ignore_index=True)
#     spy_prices = prices[prices["ticker"] == "SPY"]
#     constituents_prices = prices[prices["ticker"] != "SPY"]
#     if os.path.exists(PRICES_SPY_CSV):
#         print("Warning: prices_spy.csv already exists, overwriting")
#     if os.path.exists(PRICES_CSV):
#         print("Warning: prices_equity.csv already exists, overwriting")
#     spy_prices.to_csv(PRICES_SPY_CSV, index=False)
#     constituents_prices.to_csv(PRICES_CSV, index=False)
# else:
#     print("Error: No price data retrieved, checking CSVs")
#     if os.path.exists(PRICES_CSV) and os.path.exists(PRICES_SPY_CSV):
#         print("Loading prices from prices_equity.csv and prices_spy.csv")
#         try:
#             prices = pd.concat([
#                 pd.read_csv(PRICES_CSV),
#                 pd.read_csv(PRICES_SPY_CSV)
#             ], ignore_index=True)
#             print(f"Loaded prices for {len(prices['ticker'].unique())} tickers")
#         except Exception as e:
#             print(f"Error loading prices CSVs: {str(e)}")
#             prices = pd.DataFrame(columns=["ticker", "date", "adj_close"])
#     else:
#         print("Error: One or both price CSVs not found, using empty prices")
#         prices = pd.DataFrame(columns=["ticker", "date", "adj_close"])
# print("Prices retrieval done")
#
# print("Starting fundamentals retrieval")
# fund = {}
# for ep in ENDPOINTS:
#     name, rows = ep["name"], []
#     total_tickers = len(tickers[:-1])  # exclude SPY
#     print(f"Retrieving {name} for {total_tickers} tickers")
#     for i, ticker in enumerate(tickers[:-1], 1):  # exclude SPY
#         df = fmp_get(ep["url"], ep["params"], ticker=ticker)
#         if not df.empty:
#             df["date"] = pd.to_datetime(df["date"], errors="coerce")
#             df["ticker"] = df["ticker"].astype(str).str.upper()
#             rows.append(df)
#         if i % 100 == 0 or i == total_tickers:
#             print(f"{name} {i}/{total_tickers}")
#     if rows:
#         fund[name] = pd.concat(rows, ignore_index=True)
#         if os.path.exists(f"data/{name}.csv"):
#             print(f"Warning: {name}.csv already exists, overwriting")
#         print(f"Saving {name} to data/{name}.csv")
#         fund[name].to_csv(f"data/{name}.csv", index=False)
#     else:
#         print(f"Error: No data retrieved for {name}, checking CSV")
#         csv_path = f"data/{name}.csv"
#         if os.path.exists(csv_path):
#             print(f"Loading {name}.csv")
#             try:
#                 fund[name] = pd.read_csv(csv_path)
#                 fund[name]["year"] = pd.to_datetime(fund[name]["date"], errors="coerce").dt.year.astype("Int64")
#                 fund[name] = fund[name][fund[name]["year"].between(2005, 2024)]
#                 print(f"Loaded {name}")
#             except Exception as e:
#                 print(f"Error loading {name}.csv: {str(e)}")
#                 fund[name] = pd.DataFrame()
#         else:
#             print(f"Error: No data for {name} and no CSV found, using empty DataFrame")
#             fund[name] = pd.DataFrame()
# print("Fundamentals retrieval done")

# --- Loading mode (uncomment to use, comment to use retrieval) ---
# Ensure data/ contains sp500_constituents.csv, prices_equity.csv, prices_spy.csv,
# income_statement.csv, balance_sheet.csv, cash_flow.csv, ratios.csv, key_metrics.csv,
# income_growth.csv, balance_growth.csv, cashflow_growth.csv, financial_growth.csv.
print("Loading constituents from sp500_constituents.csv")
if os.path.exists(CONSTITUENTS_CSV):
    try:
        constituents = pd.read_csv(CONSTITUENTS_CSV)
        tickers = sorted(constituents["symbol"].astype(str).str.upper().unique()) + ["SPY"]
        print(f"Loaded {len(tickers)} tickers")
    except Exception as e:
        print(f"Error loading constituents: {str(e)}")
        tickers = ["SPY"]
else:
    print("Error: sp500_constituents.csv not found, using SPY only")
    tickers = ["SPY"]

print("Loading prices from prices_equity.csv and prices_spy.csv")
if os.path.exists(PRICES_CSV) and os.path.exists(PRICES_SPY_CSV):
    try:
        prices = pd.concat([
            pd.read_csv(PRICES_CSV),
            pd.read_csv(PRICES_SPY_CSV)
        ], ignore_index=True)
        print(f"Loaded prices for {len(prices['ticker'].unique())} tickers")
    except Exception as e:
        print(f"Error loading prices: {str(e)}")
        prices = pd.DataFrame(columns=["ticker", "date", "adj_close"])
else:
    print("Error: One or both price CSVs not found, using empty prices")
    prices = pd.DataFrame(columns=["ticker", "date", "adj_close"])

print("Computing price summary")
prices["year"] = pd.to_datetime(prices["date"], errors="coerce").dt.year
prices = prices[prices["year"].between(2005, 2024)]
if not prices.empty:
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    price_summary = (prices.groupby(["ticker", "year"])
                    .agg(price_begin=("adj_close", "first"), price_end=("adj_close", "last"))
                    .reset_index())
    price_summary["price_change"] = ((price_summary["price_end"] / price_summary["price_begin"] - 1) * 100).round(2)
else:
    print("Error: No valid price data, using empty price summary")
    price_summary = pd.DataFrame(columns=["ticker", "year", "price_begin", "price_end", "price_change"])
print("Price summary done")

print("Loading fundamentals")
fund = {}
for name, csv_path in FUNDAMENTALS_CSVS.items():
    print(f"Loading {name}.csv")
    if os.path.exists(csv_path):
        try:
            fund[name] = pd.read_csv(csv_path)
            fund[name]["year"] = pd.to_datetime(fund[name]["date"], errors="coerce").dt.year.astype("Int64")
            fund[name] = fund[name][fund[name]["year"].between(2005, 2024)]
            print(f"Loaded {name}")
        except Exception as e:
            print(f"Error loading {name}.csv: {str(e)}")
            fund[name] = pd.DataFrame()
    else:
        print(f"Error: {name}.csv not found, using empty DataFrame")
        fund[name] = pd.DataFrame()
print("Fundamentals loading done")

def first_by_year(df: pd.DataFrame, cols: list[str], prefix: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ticker", "year"] + [f"{prefix}_{c}" for c in cols])
    d = df.copy()
    d["ticker"] = d.get("ticker", d.get("symbol", "")).astype(str).str.upper()
    d = d[d["year"].between(2005, 2024)]
    keep_cols = [c for c in cols if c not in EXCLUDED_COLS]
    d = d[["ticker", "year"] + keep_cols].dropna(subset=["year"]).sort_values(["ticker", "year"])
    d = d.groupby(["ticker", "year"], as_index=False).first()
    for c in keep_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce", downcast="float")
    d = d.rename(columns={c: f"{prefix}_{c}" for c in keep_cols})
    return d

print("Merging yearly data")
data_frames = []
for name in fund:
    cols = [c for c in fund[name].columns if c not in EXCLUDED_COLS]
    df = first_by_year(fund[name], cols, prefix=name)
    if not df.empty:
        data_frames.append(df)

panel = price_summary[price_summary["year"].between(2005, 2024)][["ticker", "year", "price_begin", "price_end", "price_change"]]
for df in data_frames:
    panel = panel.merge(df, on=["ticker", "year"], how="outer")
panel = panel.sort_values(["year", "ticker"]).reset_index(drop=True)

print(f"Saving to {DATASET_CSV}")
try:
    panel.to_csv(DATASET_CSV, index=False)
    print("Done")
except Exception as e:
    print(f"Error saving {DATASET_CSV}: {str(e)}")