import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(name, model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_proba = y_pred.astype(float)

    y_pred = (y_proba >= 0.5).astype(int)

    logger.info("Results for %s", name)
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))
    logger.info(
        "Classification report (threshold=0.5):\n%s",
        classification_report(y_test, y_pred, digits=3),
    )

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        logger.info("ROC-AUC: %.4f", roc_auc)
        logger.info("PR-AUC : %.4f", pr_auc)
    except ValueError:
        logger.info("ROC-AUC / PR-AUC not defined (single class in predictions).")

    return y_proba


def threshold_sweep(y_test, y_proba, label="model"):
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        cost_fp = 1.0
        cost_fn = 10.0
        cost = cost_fp * fp + cost_fn * fn

        rows.append(
            {
                "threshold": t,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "fp": fp,
                "fn": fn,
                "cost": cost,
            }
        )

    df_thr = pd.DataFrame(rows)
    logger.info("Threshold sweep for %s (sorted by cost):\n%s", label, df_thr.sort_values("cost").head(10))
    return df_thr


def print_feature_importance(name, model):
    if not hasattr(model, "named_steps"):
        return
    if "prep" not in model.named_steps:
        return
    prep = model.named_steps["prep"]
    try:
        feature_names = prep.get_feature_names_out()
    except AttributeError:
        return

    est = model.named_steps.get("model")
    if not hasattr(est, "feature_importances_"):
        return

    importances = est.feature_importances_
    order = np.argsort(importances)[::-1]
    top_k = min(20, len(order))

    lines = []
    for idx in order[:top_k]:
        lines.append(f"{feature_names[idx]}: {importances[idx]:.4f}")
    logger.info("Top features for %s:\n%s", name, "\n".join(lines))

