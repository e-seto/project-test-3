import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

# ── 1. Feature Engineering (from previous step) ───────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

data["haversine_dist_km"]     = haversine(data["lat"], data["long"], data["merch_lat"], data["merch_long"])
data["trans_datetime"]        = pd.to_datetime(data["trans_datetime"])
data["hour"]                  = data["trans_datetime"].dt.hour
data["day_of_week"]           = data["trans_datetime"].dt.dayofweek
cust_avg                      = data.groupby("cc_num")["amt"].transform("mean")
cust_std                      = data.groupby("cc_num")["amt"].transform("std").fillna(1)
data["amt_deviation"]         = data["amt"] - cust_avg
data["amt_deviation_ratio"]   = data["amt"] / cust_avg
data["amt_deviation_zscore"]  = (data["amt"] - cust_avg) / cust_std

print("Feature engineering complete.")

# ── 2. Drop columns ───────────────────────────────────────────────────────────
DROP_COLS = [
    "cc_num",           # PII — high cardinality card number
    "merchant",         # PII — high cardinality merchant name
    "trans_datetime",   # replaced by hour + day_of_week
    "lat", "long",      # replaced by haversine_dist_km
    "merch_lat", "merch_long",  # replaced by haversine_dist_km
    "_group",           # metadata, not a model feature
]

data = data.drop(columns=DROP_COLS)
print(f"\nAfter dropping columns: {data.shape}")
print(f"Remaining columns: {list(data.columns)}")

# ── 3. Encode categorical columns ─────────────────────────────────────────────
# Low cardinality cols: gender, category, state, job, city — safe to label encode
CAT_COLS = ["gender", "category", "state", "job", "city"]

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le
    print(f"Encoded '{col}': {len(le.classes_)} unique values → {le.classes_.tolist()}")

# ── 4. Stratified Train / Test Split ─────────────────────────────────────────
# Split BEFORE undersampling — test set must reflect real-world class distribution
X = data.drop(columns=["is_fraud"])
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserves fraud ratio in both splits
)

print(f"\n── Train/Test Split ─────────────────────────────────────────────────")
print(f"Train size: {len(X_train):,}  |  Fraud: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
print(f"Test size:  {len(X_test):,}   |  Fraud: {y_test.sum():,}  ({y_test.mean()*100:.4f}%)")

# ── 5. Undersample majority class (training set ONLY) ────────────────────────
# Keep all fraud rows, randomly sample legitimate rows to create a balanced ratio
# Using 10:1 ratio (legit:fraud) — keeps more data than 1:1 but reduces imbalance
train_df = pd.concat([X_train, y_train], axis=1)

fraud_df  = train_df[train_df["is_fraud"] == 1]
legit_df  = train_df[train_df["is_fraud"] == 0]

# 10:1 undersample ratio — adjust UNDERSAMPLE_RATIO to change balance
UNDERSAMPLE_RATIO = 10
n_legit_keep = len(fraud_df) * UNDERSAMPLE_RATIO

legit_sampled = legit_df.sample(n=n_legit_keep, random_state=42)
train_balanced = pd.concat([fraud_df, legit_sampled]).sample(frac=1, random_state=42)

X_train_bal = train_balanced.drop(columns=["is_fraud"])
y_train_bal = train_balanced["is_fraud"]

print(f"\n── After Undersampling (train only) ────────────────────────────────")
print(f"Balanced train size: {len(X_train_bal):,}")
print(f"  Legitimate: {(y_train_bal == 0).sum():,}  ({(y_train_bal == 0).mean()*100:.1f}%)")
print(f"  Fraud:      {(y_train_bal == 1).sum():,}  ({(y_train_bal == 1).mean()*100:.1f}%)")
print(f"  Ratio:      {UNDERSAMPLE_RATIO}:1 (legit:fraud)")
print(f"\nTest set unchanged (real-world distribution):")
print(f"  Legitimate: {(y_test == 0).sum():,}  ({(y_test.mean()*100):.4f}% fraud)")
print(f"  Fraud:      {(y_test == 1).sum():,}")

# ── 6. Final feature summary ──────────────────────────────────────────────────
print(f"\n── Final Feature List ({X_train_bal.shape[1]} features) ───────────────────────────")
for col in X_train_bal.columns:
    print(f"  {col}")

# ── 7. Save preprocessed splits ──────────────────────────────────────────────
X_train_bal.to_csv("data/processed/X_train.csv", index=False)
y_train_bal.to_csv("data/processed/y_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv",   index=False)
y_test.to_csv("data/processed/y_test.csv",   index=False)

print(f"\nSaved:")
print(f"  X_train.csv  {X_train_bal.shape}")
print(f"  y_train.csv  {y_train_bal.shape}")
print(f"  X_test.csv   {X_test.shape}")
print(f"  y_test.csv   {y_test.shape}")
print("\nPreprocessing complete. Ready for modelling.")