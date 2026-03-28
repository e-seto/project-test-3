import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load data and best model ──────────────────────────────────────────────
print("Loading data and model...")
X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

model = joblib.load("models/xgboost.joblib")
print(f"Model loaded: XGBoost")
print(f"Test set: {X_test.shape}\n")

# ── 1. Compute SHAP values ──────────────────────────────────────────────────
print("Computing SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# ── 2. SHAP Summary Plot (global feature importance with direction) ─────────
print("Generating SHAP summary plot...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot — Feature Impact on Fraud Prediction", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/plot8_shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot8_shap_summary.png\n")

# ── 3. SHAP Bar Plot (mean absolute SHAP values) ────────────────────────────
print("Generating SHAP bar plot...")
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.bar(shap_values, show=False)
plt.title("Mean |SHAP| — Global Feature Importance", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/plot9_shap_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot9_shap_bar.png\n")

# ── 4. SHAP Waterfall Plot (single prediction explanation) ──────────────────
# Show one fraud case and one legitimate case
fraud_indices = y_test[y_test == 1].index
legit_indices = y_test[y_test == 0].index

print("Generating waterfall plot for a FRAUD transaction...")
fraud_idx = fraud_indices[0]
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_values[fraud_idx], show=False)
plt.title(f"SHAP Waterfall — Fraud Transaction (index {fraud_idx})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/plot10_shap_waterfall_fraud.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot10_shap_waterfall_fraud.png\n")

print("Generating waterfall plot for a LEGITIMATE transaction...")
legit_idx = legit_indices[0]
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_values[legit_idx], show=False)
plt.title(f"SHAP Waterfall — Legitimate Transaction (index {legit_idx})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/plot11_shap_waterfall_legit.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot11_shap_waterfall_legit.png\n")

# ── 5. Top feature SHAP dependence plot ─────────────────────────────────────
mean_shap = np.abs(shap_values.values).mean(axis=0)
top_feature = X_test.columns[np.argmax(mean_shap)]

print(f"Generating dependence plot for top feature: {top_feature}...")
fig, ax = plt.subplots(figsize=(10, 6))
shap.dependence_plot(top_feature, shap_values.values, X_test, show=False, ax=ax)
ax.set_title(f"SHAP Dependence — {top_feature}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/plot12_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot12_shap_dependence.png\n")

print("SHAP analysis complete.")
