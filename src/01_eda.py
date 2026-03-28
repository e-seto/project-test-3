import os
import glob
import pandas as pd

# ── 1. Load all CSVs from the cleaned_data_files folder ──────────────────────
DATA_ROOT = "data/raw"

all_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True)
print(f"Found {len(all_files)} CSV files\n")

# Read and tag each file with its demographic group (folder name)
dfs = []
for fp in sorted(all_files):
    df = pd.read_csv(fp)
    folder = os.path.basename(os.path.dirname(fp))   # e.g. cleaned_50up_f_r
    df["_group"] = folder
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# ── 2. Basic shape & dtypes ───────────────────────────────────────────────────
print("=" * 60)
print("SHAPE")
print("=" * 60)
print(f"Rows:    {data.shape[0]:,}")
print(f"Columns: {data.shape[1]}\n")

print("=" * 60)
print("COLUMNS & DATA TYPES")
print("=" * 60)
print(data.dtypes.to_string())
print()

# ── 3. Missing values ─────────────────────────────────────────────────────────
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = data.isnull().sum()
missing_pct = (missing / len(data) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0]

if missing_df.empty:
    print("No missing values found.")
else:
    print(missing_df.to_string())
print()

# ── 4. Class balance ──────────────────────────────────────────────────────────
print("=" * 60)
print("FRAUD / LEGITIMATE CLASS BALANCE")
print("=" * 60)
counts = data["is_fraud"].value_counts()
pcts   = data["is_fraud"].value_counts(normalize=True) * 100
print(counts.to_string())
balance = pd.DataFrame({
    "Label":   ["Legitimate (0)", "Fraud (1)"],
    "Count":   [counts.get(0, 0), counts.get(1, 0)],
    "Percent": [round(pcts.get(0, 0), 4), round(pcts.get(1, 0), 4)]
})
print(balance.to_string(index=False))
print()

# ── 5. Per-group class balance ────────────────────────────────────────────────
print("=" * 60)
print("FRAUD RATE PER DEMOGRAPHIC GROUP")
print("=" * 60)
group_balance = (
    data.groupby("_group")["is_fraud"]
    .agg(
        Total="count",
        Fraud_Count="sum",
    )
    .assign(Fraud_Rate_Pct=lambda x: (x["Fraud_Count"] / x["Total"] * 100).round(4))
)
print(group_balance.to_string())
print()

# ── 6. Basic descriptive stats for numeric columns ───────────────────────────
print("=" * 60)
print("DESCRIPTIVE STATISTICS (numeric columns)")
print("=" * 60)
print(data.describe().round(2).to_string())