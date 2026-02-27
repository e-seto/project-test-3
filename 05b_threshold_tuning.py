import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load data & retrain XGBoost ───────────────────────────────────────────
print("Loading data...")
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
X_test  = pd.read_csv("X_test.csv")
y_test  = pd.read_csv("y_test.csv").squeeze()

print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
    n_jobs=-1, verbosity=0
)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}\n")

# ── 1. Sweep thresholds from 0.01 to 0.99 ────────────────────────────────────
thresholds  = np.arange(0.01, 1.00, 0.01)
precisions, recalls, f1s = [], [], []
tps, fps, fns, tns       = [], [], [], []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    tps.append(tp); fps.append(fp); fns.append(fn); tns.append(tn)
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred, zero_division=0))

precisions = np.array(precisions)
recalls    = np.array(recalls)
f1s        = np.array(f1s)
tps        = np.array(tps)
fps        = np.array(fps)
fns        = np.array(fns)

# ── 2. Find key thresholds ────────────────────────────────────────────────────
# Best F1 — balanced fraud catch vs false alarms
best_f1_idx       = np.argmax(f1s)
best_f1_threshold = thresholds[best_f1_idx]

# Best recall >= 0.95 with highest precision — catch nearly all fraud
high_recall_mask  = recalls >= 0.95
if high_recall_mask.any():
    high_recall_idx       = np.where(high_recall_mask)[0][np.argmax(precisions[high_recall_mask])]
    high_recall_threshold = thresholds[high_recall_idx]
else:
    high_recall_idx       = np.argmax(recalls)
    high_recall_threshold = thresholds[high_recall_idx]

# Best precision >= 0.60 with highest recall — fewer false alarms
high_prec_mask = precisions >= 0.60
if high_prec_mask.any():
    high_prec_idx       = np.where(high_prec_mask)[0][np.argmax(recalls[high_prec_mask])]
    high_prec_threshold = thresholds[high_prec_idx]
else:
    high_prec_idx       = np.argmax(precisions)
    high_prec_threshold = thresholds[high_prec_idx]

print("=" * 60)
print("KEY THRESHOLDS")
print("=" * 60)

for label, idx, t in [
    ("Default (0.50)",        np.argmin(np.abs(thresholds - 0.50)), 0.50),
    ("Best F1",               best_f1_idx,       best_f1_threshold),
    ("High Recall (≥0.95)",   high_recall_idx,   high_recall_threshold),
    ("High Precision (≥0.60)",high_prec_idx,     high_prec_threshold),
]:
    print(f"\n  {label}  →  threshold = {t:.2f}")
    print(f"    Recall:    {recalls[idx]:.4f}  ({int(tps[idx]):,} fraud caught,  {int(fns[idx]):,} missed)")
    print(f"    Precision: {precisions[idx]:.4f}  ({int(fps[idx]):,} false alarms)")
    print(f"    F1-Score:  {f1s[idx]:.4f}")

# ── 3. Plot 1 — Precision / Recall / F1 vs Threshold ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(thresholds, precisions, label="Precision", color="#4C72B0", lw=2)
ax.plot(thresholds, recalls,    label="Recall",    color="#DD4949", lw=2)
ax.plot(thresholds, f1s,        label="F1-Score",  color="#2ca02c", lw=2)

# Mark key thresholds
for label, t, color in [
    ("Default\n0.50",  0.50,                "#888888"),
    (f"Best F1\n{best_f1_threshold:.2f}",   best_f1_threshold,   "#2ca02c"),
    (f"High Recall\n{high_recall_threshold:.2f}", high_recall_threshold, "#DD4949"),
    (f"High Prec\n{high_prec_threshold:.2f}",    high_prec_threshold,   "#4C72B0"),
]:
    ax.axvline(x=t, linestyle="--", color=color, alpha=0.7, lw=1.5)
    ax.text(t + 0.005, 0.05, label, fontsize=8, color=color)

ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig("plot8_threshold_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved plot8_threshold_metrics.png")

# ── 4. Plot 2 — TP and FP counts vs Threshold ────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.plot(thresholds, tps, color="#DD4949", lw=2, label="Fraud Caught (TP)")
ax1.plot(thresholds, fns, color="#ff7f0e", lw=2, linestyle="--", label="Fraud Missed (FN)")
ax2.plot(thresholds, fps, color="#4C72B0", lw=2, label="False Alarms (FP)")

ax1.set_xlabel("Classification Threshold")
ax1.set_ylabel("Fraud Transactions", color="#DD4949")
ax2.set_ylabel("False Alarms (Legitimate flagged)", color="#4C72B0")
ax1.set_title("Fraud Caught vs False Alarms vs Threshold", fontsize=14, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

for t, color in [(0.50, "#888888"), (best_f1_threshold, "#2ca02c"),
                  (high_recall_threshold, "#DD4949"), (high_prec_threshold, "#4C72B0")]:
    ax1.axvline(x=t, linestyle="--", color=color, alpha=0.5, lw=1.2)

plt.tight_layout()
plt.savefig("plot9_tp_fp_vs_threshold.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plot9_tp_fp_vs_threshold.png")

# ── 5. Full threshold table (every 0.05) ─────────────────────────────────────
print("\n" + "=" * 60)
print("THRESHOLD SWEEP TABLE (every 0.05)")
print("=" * 60)
step_indices = [np.argmin(np.abs(thresholds - t)) for t in np.arange(0.05, 1.00, 0.05)]
table = pd.DataFrame({
    "Threshold":  [round(thresholds[i], 2) for i in step_indices],
    "Recall":     [round(recalls[i], 4)    for i in step_indices],
    "Precision":  [round(precisions[i], 4) for i in step_indices],
    "F1":         [round(f1s[i], 4)        for i in step_indices],
    "Fraud Caught (TP)":  [int(tps[i])     for i in step_indices],
    "Fraud Missed (FN)":  [int(fns[i])     for i in step_indices],
    "False Alarms (FP)":  [int(fps[i])     for i in step_indices],
})
print(table.to_string(index=False))

# ── 6. Recommended thresholds for prescriptive policy ────────────────────────
print("\n" + "=" * 60)
print("RECOMMENDED THRESHOLDS FOR PRESCRIPTIVE POLICY")
print("=" * 60)
print(f"""
  These thresholds feed directly into your Block/Review/Allow policy:

  BLOCK threshold  →  {high_prec_threshold:.2f}
    Transactions above this are auto-declined.
    High confidence fraud — minimizes false alarms on hard blocks.

  REVIEW threshold →  {best_f1_threshold:.2f}
    Transactions between REVIEW and BLOCK go to manual review.
    Balanced catch rate — worth a human look.

  ALLOW threshold  →  below {best_f1_threshold:.2f}
    Transactions below this are auto-approved.
    Low fraud probability — not worth flagging.

  Save these values — you will use them in 06_prescriptive_policy.py
""")