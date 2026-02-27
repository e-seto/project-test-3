import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load preprocessed splits ──────────────────────────────────────────────
print("Loading preprocessed data...")
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
X_test  = pd.read_csv("X_test.csv")
y_test  = pd.read_csv("y_test.csv").squeeze()

print(f"Train: {X_train.shape}  |  Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"Test:  {X_test.shape}   |  Fraud in test:  {y_test.sum():,}  ({y_test.mean()*100:.4f}%)")
print(f"Features: {list(X_train.columns)}\n")

# ── 1. Define models ──────────────────────────────────────────────────────────
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1           # use all CPU cores
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # already balanced via undersampling
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    ),
}

# ── 2. Train & evaluate each model ───────────────────────────────────────────
results = {}

for name, model in models.items():
    print("=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Training time: {train_time:.1f}s")

    # Predict
    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc   = roc_auc_score(y_test, y_pred_proba)
    cm    = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    recall    = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results[name] = {
        "model":      model,
        "y_pred":     y_pred,
        "y_proba":    y_pred_proba,
        "auc":        auc,
        "recall":     recall,
        "precision":  precision,
        "f1":         f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "train_time": train_time,
    }

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4))
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix:  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}\n")

# ── 3. Summary comparison table ───────────────────────────────────────────────
print("=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    name: {
        "AUC-ROC":    round(r["auc"], 4),
        "Recall":     round(r["recall"], 4),
        "Precision":  round(r["precision"], 4),
        "F1-Score":   round(r["f1"], 4),
        "TP (fraud caught)":   r["tp"],
        "FP (false alarms)":   r["fp"],
        "FN (missed fraud)":   r["fn"],
        "Train Time (s)":      round(r["train_time"], 1),
    }
    for name, r in results.items()
}).T

print(summary.to_string())

best_model_name = summary["F1-Score"].astype(float).idxmax()
print(f"\nBest model by F1-Score: {best_model_name}")
best_model_name_auc = summary["AUC-ROC"].astype(float).idxmax()
print(f"Best model by AUC-ROC:  {best_model_name_auc}")

# ── 4. Confusion matrix plots ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices (Test Set)", fontsize=15, fontweight="bold")

for ax, (name, r) in zip(axes, results.items()):
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAUC={r['auc']:.4f}  F1={r['f1']:.4f}")

plt.tight_layout()
plt.savefig("plot5_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot5_confusion_matrices.png")

# ── 5. ROC curve comparison ───────────────────────────────────────────────────
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#4C72B0", "#DD4949", "#2ca02c"]

for (name, r), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.4f})", color=color, lw=2)

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plot6_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot6_roc_curves.png")

# ── 6. Feature importance (best model) ───────────────────────────────────────
best = results[best_model_name]["model"]

if hasattr(best, "feature_importances_"):
    fi = pd.Series(best.feature_importances_, index=X_train.columns)
    fi = fi.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    fi.plot(kind="bar", color="#4C72B0", ax=ax)
    ax.set_title(f"Feature Importance — {best_model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance Score")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("plot7_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved plot7_feature_importance.png")

    print(f"\nFeature importances ({best_model_name}):")
    print(fi.round(4).to_string())

print("\nModelling complete.")
