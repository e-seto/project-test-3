import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
X_test  = pd.read_csv("data/processed/X_test.csv")
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Fraud in test: {y_test.sum():,} ({y_test.mean()*100:.4f}%)\n")

# ── Load pre-trained ensemble models (no retraining needed) ──────────────────
print("Loading saved ensemble models...")
rf  = joblib.load("models/random_forest.joblib")
xgb = joblib.load("models/xgboost.joblib")
ada = joblib.load("models/adaboost.joblib")
print("All models loaded.\n")

# ── 1. Helper function ────────────────────────────────────────────────────────
def evaluate(name, y_test, y_pred, y_proba, train_time):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    auc       = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimate", "Fraud"], digits=4))
    print(f"AUC-ROC:   {auc:.4f}" if auc else "AUC-ROC:   N/A")
    print(f"Recall:    {recall:.4f}  ({tp:,} caught, {fn:,} missed)")
    print(f"Precision: {precision:.4f}  ({fp:,} false alarms)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Train Time: {train_time:.1f}s")

    return {
        "AUC-ROC":           round(auc, 4) if auc else "N/A",
        "Recall":            round(recall, 4),
        "Precision":         round(precision, 4),
        "F1-Score":          round(f1, 4),
        "TP (fraud caught)": tp,
        "FP (false alarms)": fp,
        "FN (missed fraud)": fn,
        "Train Time (s)":    round(train_time, 1),
    }

results = {}

# ── 2. Baseline 1 — Majority Class ───────────────────────────────────────────
print("\n>>> BASELINE 1: Majority Class Classifier")
start = time.time()
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
t = time.time() - start

results["Majority Class\n(Baseline 1)"] = evaluate(
    "Majority Class (always legitimate)",
    y_test, dummy.predict(X_test),
    dummy.predict_proba(X_test)[:, 1], t
)
print("\n*** NOTE: 99.45% accuracy by NEVER flagging fraud.")
print("    This exposes why accuracy is meaningless for imbalanced data.")

# ── 3. Baseline 2 — Logistic Regression ──────────────────────────────────────
print("\n>>> BASELINE 2: Logistic Regression")
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

start = time.time()
lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                         random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
t = time.time() - start

results["Logistic Regression\n(Baseline 2)"] = evaluate(
    "Logistic Regression",
    y_test, lr.predict(X_test_scaled),
    lr.predict_proba(X_test_scaled)[:, 1], t
)

# ── 4. Ensemble models — loaded from disk, no retraining ─────────────────────
ensemble_map = {
    "Random Forest": (rf,  X_test),
    "XGBoost":       (xgb, X_test),
    "AdaBoost":      (ada, X_test),
}

for name, (model, X) in ensemble_map.items():
    print(f"\n>>> {name}  (loaded from saved_models/)")
    results[name] = evaluate(
        name, y_test,
        model.predict(X),
        model.predict_proba(X)[:, 1],
        train_time=0.0   # already trained — load time is instant
    )

# ── 5. Full comparison table ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FULL MODEL COMPARISON — BASELINES vs ENSEMBLE MODELS")
print("=" * 70)
summary = pd.DataFrame(results).T
print(summary.to_string())

print("\n── Key takeaways ─────────────────────────────────────────────────────")
lr_f1  = results["Logistic Regression\n(Baseline 2)"]["F1-Score"]
xgb_f1 = results["XGBoost"]["F1-Score"]
rf_f1  = results["Random Forest"]["F1-Score"]
ada_f1 = results["AdaBoost"]["F1-Score"]

print(f"  Majority Class F1:      0.0000  (catches zero fraud)")
print(f"  Logistic Regression F1: {lr_f1:.4f}  (ML baseline)")
print(f"  Random Forest F1:       {rf_f1:.4f}  (+{rf_f1-lr_f1:.4f} vs baseline)")
print(f"  AdaBoost F1:            {ada_f1:.4f}  (+{ada_f1-lr_f1:.4f} vs baseline)")
print(f"  XGBoost F1:             {xgb_f1:.4f}  (+{xgb_f1-lr_f1:.4f} vs baseline)  ← BEST")

# ── 6. ROC curve — all models ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

plot_configs = [
    ("Majority Class (Baseline 1)", dummy,  X_test,        "#aaaaaa", "--"),
    ("Logistic Regression (Baseline 2)", lr, X_test_scaled, "#ff7f0e", "--"),
    ("Random Forest",  rf,  X_test, "#4C72B0", "-"),
    ("AdaBoost",       ada, X_test, "#9467bd", "-"),
    ("XGBoost (Best)", xgb, X_test, "#DD4949", "-"),
]

for label, model, X, color, ls in plot_configs:
    proba = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})",
            color=color, lw=2, linestyle=ls)

ax.plot([0,1], [0,1], "k:", lw=1, label="Random Baseline")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curve — All Models vs Baselines", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/plot10_roc_all_models.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved plot10_roc_all_models.png")

# ── 7. F1 bar chart ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(results.keys())
f1_scores   = [results[m]["F1-Score"] for m in model_names]
colors      = ["#aaaaaa", "#ff7f0e", "#4C72B0", "#DD4949", "#9467bd"]

bars = ax.bar([m.replace("\n", "\n") for m in model_names],
              f1_scores, color=colors, edgecolor="white", linewidth=1.5)

for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("F1-Score Comparison — Baselines vs Ensemble Models",
             fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.0)
ax.axhline(y=lr_f1, color="#ff7f0e", linestyle="--", alpha=0.5, label="LR baseline")
plt.tight_layout()
plt.savefig("plots/plot11_f1_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot11_f1_comparison.png")

print("\nBaseline comparison complete.")