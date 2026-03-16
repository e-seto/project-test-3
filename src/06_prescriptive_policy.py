import pandas as pd
import numpy as np
import json
import joblib

# ── 0. Load model and thresholds ──────────────────────────────────────────────
print("Loading XGBoost model...")
model = joblib.load("models/xgboost.joblib")

print("Loading policy thresholds...")
with open("models/policy_thresholds.json") as f:
    thresholds = json.load(f)

REVIEW_THRESHOLD = thresholds["review"]
BLOCK_THRESHOLD  = thresholds["block"]

print(f"  ALLOW  -> score < {REVIEW_THRESHOLD}")
print(f"  REVIEW -> {REVIEW_THRESHOLD} <= score < {BLOCK_THRESHOLD}")
print(f"  BLOCK  -> score >= {BLOCK_THRESHOLD}\n")

# ── 1. Load test data ─────────────────────────────────────────────────────────
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# ── 2. Get fraud probability scores ──────────────────────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]

# ── 3. Apply policy ───────────────────────────────────────────────────────────
def apply_policy(score):
    if score >= BLOCK_THRESHOLD:
        return "BLOCK"
    elif score >= REVIEW_THRESHOLD:
        return "REVIEW"
    else:
        return "ALLOW"

decisions = pd.Series(y_proba).apply(apply_policy)

# ── 4. Results ────────────────────────────────────────────────────────────────
results = pd.DataFrame({
    "fraud_score":    y_proba,
    "decision":       decisions,
    "actual_label":   y_test.values,
})

print("=" * 60)
print("POLICY DECISION BREAKDOWN")
print("=" * 60)
counts = decisions.value_counts()
for decision in ["ALLOW", "REVIEW", "BLOCK"]:
    n = counts.get(decision, 0)
    print(f"  {decision:6s}: {n:,} transactions ({n/len(decisions)*100:.1f}%)")

print("\n" + "=" * 60)
print("FRAUD CAUGHT BY DECISION")
print("=" * 60)
for decision in ["ALLOW", "REVIEW", "BLOCK"]:
    subset = results[results["decision"] == decision]
    actual_fraud = subset["actual_label"].sum()
    total = len(subset)
    print(f"  {decision:6s}: {actual_fraud:,} actual fraud out of {total:,} transactions")

# ── 5. Save results ───────────────────────────────────────────────────────────
results.to_csv("data/processed/policy_decisions.csv", index=False)
print("\nSaved decisions to data/processed/policy_decisions.csv")
print("\nPrescriptive policy complete.")
