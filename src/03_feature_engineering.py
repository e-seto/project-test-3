import os
import glob
import pandas as pd
import numpy as np

# ── 0. Load all CSVs ──────────────────────────────────────────────────────────
DATA_ROOT = "data/raw"

all_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True)
print(f"Found {len(all_files)} CSV files")

dfs = []
for fp in sorted(all_files):
    df = pd.read_csv(fp)
    folder = os.path.basename(os.path.dirname(fp))
    df["_group"] = folder
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded: {len(data):,}")

# ── 1. Haversine Distance ─────────────────────────────────────────────────────
# Measures straight-line distance (km) between customer's home and merchant
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

data["haversine_dist_km"] = haversine(
    data["lat"], data["long"],
    data["merch_lat"], data["merch_long"]
)

# ── 2. Temporal Features ──────────────────────────────────────────────────────
data["trans_datetime"]  = pd.to_datetime(data["trans_datetime"])
data["hour"]            = data["trans_datetime"].dt.hour
data["day_of_week"]     = data["trans_datetime"].dt.dayofweek   # 0=Mon, 6=Sun
data["day_name"]        = data["trans_datetime"].dt.day_name()

# ── 3. Amount Deviation ───────────────────────────────────────────────────────
# Compare each transaction amount to that customer's historical average
cust_avg = data.groupby("cc_num")["amt"].transform("mean")
cust_std = data.groupby("cc_num")["amt"].transform("std").fillna(1)

data["amt_deviation"]        = data["amt"] - cust_avg           # raw $ difference
data["amt_deviation_ratio"]  = data["amt"] / cust_avg           # how many times above avg
data["amt_deviation_zscore"] = (data["amt"] - cust_avg) / cust_std  # standardized

# ── Preview ───────────────────────────────────────────────────────────────────
new_cols = ["cc_num", "amt", "haversine_dist_km", "hour", "day_of_week",
            "day_name", "amt_deviation", "amt_deviation_ratio",
            "amt_deviation_zscore", "is_fraud"]

print("\n── Sample rows (5 legit, 5 fraud) ──────────────────────────────────")
sample = pd.concat([
    data[data["is_fraud"] == 0].head(5),
    data[data["is_fraud"] == 1].head(5)
])
print(sample[new_cols].to_string(index=False))

print("\n── New feature statistics ───────────────────────────────────────────")
print(data[["haversine_dist_km", "hour", "day_of_week",
            "amt_deviation", "amt_deviation_ratio",
            "amt_deviation_zscore"]].describe().round(3).to_string())

print("\n── Fraud vs Legit averages for new features ─────────────────────────")
print(data.groupby("is_fraud")[["haversine_dist_km", "amt_deviation",
                                 "amt_deviation_ratio",
                                 "amt_deviation_zscore"]].mean().round(3).to_string())

print("\n── Null check on new features ───────────────────────────────────────")
print(data[["haversine_dist_km", "hour", "day_of_week",
            "amt_deviation", "amt_deviation_ratio",
            "amt_deviation_zscore"]].isnull().sum().to_string())

# ── Save enriched dataset ─────────────────────────────────────────────────────
output_path = "data/processed/data_engineered.csv"
data.to_csv(output_path, index=False)
print(f"\nEnriched dataset saved to: {output_path}")
print(f"Final shape: {data.shape}")