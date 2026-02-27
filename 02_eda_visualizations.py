import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── 0. Load all CSVs ──────────────────────────────────────────────────────────
DATA_ROOT = "cleaned_data_files"  # change if needed

all_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True)
dfs = []
for fp in sorted(all_files):
    df = pd.read_csv(fp)
    folder = os.path.basename(os.path.dirname(fp))
    df["_group"] = folder
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(data):,} rows from {len(all_files)} files")

# ── Parse datetime & extract time features ───────────────────────────────────
data["trans_datetime"] = pd.to_datetime(data["trans_datetime"])
data["hour"]       = data["trans_datetime"].dt.hour
data["day_of_week"] = data["trans_datetime"].dt.day_name()

DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── Age bins ──────────────────────────────────────────────────────────────────
data["age_group"] = pd.cut(
    data["age"],
    bins=[0, 24, 34, 44, 54, 64, 120],
    labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
)

# ── Colour palette ────────────────────────────────────────────────────────────
LEGIT_COLOR = "#4C72B0"
FRAUD_COLOR = "#DD4949"
sns.set_theme(style="whitegrid", font_scale=1.1)

# =============================================================================
# FIGURE 1 — Fraud Rate by Hour of Day & Day of Week
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Fraud Rate by Time", fontsize=16, fontweight="bold", y=1.01)

# Hour of day
hour_fraud = (
    data.groupby("hour")["is_fraud"]
    .agg(fraud_rate="mean")
    .reset_index()
)
hour_fraud["fraud_rate_pct"] = hour_fraud["fraud_rate"] * 100

sns.barplot(
    data=hour_fraud, x="hour", y="fraud_rate_pct",
    color=FRAUD_COLOR, ax=axes[0]
)
axes[0].set_title("Fraud Rate by Hour of Day")
axes[0].set_xlabel("Hour (0 = midnight)")
axes[0].set_ylabel("Fraud Rate (%)")
axes[0].tick_params(axis="x", rotation=0)

# Day of week
dow_fraud = (
    data.groupby("day_of_week")["is_fraud"]
    .agg(fraud_rate="mean")
    .reindex(DAY_ORDER)
    .reset_index()
)
dow_fraud["fraud_rate_pct"] = dow_fraud["fraud_rate"] * 100

sns.barplot(
    data=dow_fraud, x="day_of_week", y="fraud_rate_pct",
    color=FRAUD_COLOR, ax=axes[1]
)
axes[1].set_title("Fraud Rate by Day of Week")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("plot1_fraud_by_time.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot1_fraud_by_time.png")

# =============================================================================
# FIGURE 2 — Fraud Rate by Merchant Category
# =============================================================================
cat_fraud = (
    data.groupby("category")["is_fraud"]
    .agg(fraud_rate="mean", total="count")
    .reset_index()
    .sort_values("fraud_rate", ascending=False)
)
cat_fraud["fraud_rate_pct"] = cat_fraud["fraud_rate"] * 100

fig, ax = plt.subplots(figsize=(14, 6))
sns.barplot(
    data=cat_fraud, x="category", y="fraud_rate_pct",
    color=FRAUD_COLOR, ax=ax
)
ax.set_title("Fraud Rate by Merchant Category", fontsize=15, fontweight="bold")
ax.set_xlabel("Merchant Category")
ax.set_ylabel("Fraud Rate (%)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("plot2_fraud_by_category.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot2_fraud_by_category.png")

# =============================================================================
# FIGURE 3 — Transaction Amount Distribution: Fraud vs Legitimate
# =============================================================================
fraud_data  = data[data["is_fraud"] == 1]["amt"]
legit_data  = data[data["is_fraud"] == 0]["amt"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Transaction Amount: Fraud vs Legitimate", fontsize=16, fontweight="bold")

# Histogram (capped at 99th percentile for readability)
cap = data["amt"].quantile(0.99)
axes[0].hist(legit_data.clip(upper=cap), bins=60, alpha=0.6, color=LEGIT_COLOR, label="Legitimate", density=True)
axes[0].hist(fraud_data.clip(upper=cap),  bins=60, alpha=0.6, color=FRAUD_COLOR,  label="Fraud",       density=True)
axes[0].set_title("Amount Distribution (capped at 99th pct)")
axes[0].set_xlabel("Transaction Amount ($)")
axes[0].set_ylabel("Density")
axes[0].legend()

# Box plot
amt_plot = data[["amt", "is_fraud"]].copy()
amt_plot["Transaction"] = amt_plot["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
amt_plot_capped = amt_plot.copy()
amt_plot_capped["amt"] = amt_plot_capped["amt"].clip(upper=cap)

sns.boxplot(
    data=amt_plot_capped, x="Transaction", y="amt",
    palette={"Legitimate": LEGIT_COLOR, "Fraud": FRAUD_COLOR},
    ax=axes[1]
)
axes[1].set_title("Amount Box Plot (capped at 99th pct)")
axes[1].set_xlabel("")
axes[1].set_ylabel("Transaction Amount ($)")

# Print summary stats
print("\nAmount summary stats:")
print(data.groupby("is_fraud")["amt"].describe().round(2))

plt.tight_layout()
plt.savefig("plot3_amount_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot3_amount_distribution.png")

# =============================================================================
# FIGURE 4A — Fraud Rate by Age Group
# =============================================================================
age_fraud = (
    data.groupby("age_group", observed=True)["is_fraud"]
    .agg(fraud_rate="mean", total="count")
    .reset_index()
)
age_fraud["fraud_rate_pct"] = age_fraud["fraud_rate"] * 100

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=age_fraud, x="age_group", y="fraud_rate_pct",
    color=FRAUD_COLOR, ax=ax
)
ax.set_title("Fraud Rate by Age Group", fontsize=15, fontweight="bold")
ax.set_xlabel("Age Group")
ax.set_ylabel("Fraud Rate (%)")
plt.tight_layout()
plt.savefig("plot4a_fraud_by_age.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot4a_fraud_by_age.png")

# =============================================================================
# FIGURE 4B — Top 20 Jobs by Fraud Rate (min 50 transactions)
# =============================================================================
job_fraud = (
    data.groupby("job")["is_fraud"]
    .agg(fraud_rate="mean", total="count")
    .reset_index()
)
job_fraud = job_fraud[job_fraud["total"] >= 50]   # filter low-count jobs
job_fraud["fraud_rate_pct"] = job_fraud["fraud_rate"] * 100
job_fraud = job_fraud.sort_values("fraud_rate", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(14, 7))
sns.barplot(
    data=job_fraud, x="fraud_rate_pct", y="job",
    color=FRAUD_COLOR, ax=ax
)
ax.set_title("Top 20 Jobs by Fraud Rate (min 50 transactions)", fontsize=15, fontweight="bold")
ax.set_xlabel("Fraud Rate (%)")
ax.set_ylabel("Job Title")
plt.tight_layout()
plt.savefig("plot4b_fraud_by_job.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot4b_fraud_by_job.png")

print("\nAll plots saved successfully.")