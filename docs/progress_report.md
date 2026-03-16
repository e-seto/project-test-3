# Credit Card Fraud Detection — Progress Report

## Overview

This project builds a machine learning pipeline to detect credit card fraud and convert model predictions into actionable business decisions. The goal is not just to classify transactions, but to recommend what to *do* with each one — approve, review, or block.

---

## Pipeline Summary

```
Raw Data
   |
   v
Cleaning & Feature Engineering
   |
   v
Preprocessing (undersample, train/test split)
   |
   v
Baseline Models (Majority Class, Logistic Regression)
   |
   v
Ensemble Models (Random Forest, XGBoost, AdaBoost)
   |
   v
Threshold Tuning
   |
   v
Prescriptive Policy (ALLOW / REVIEW / BLOCK)
```

---

## Dataset

- **695,763** total transactions in test set
- **3,806** actual fraud cases (0.55% base rate)
- Severe class imbalance handled via undersampling (10:1 ratio in training)

---

## Baseline Models

Two simple baselines were established before training ensemble models.

| Model | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|
| Majority Class | 0.50 | 0.00 | 0.00 | 0.00 |
| Logistic Regression | 0.88 | 0.76 | 0.08 | 0.15 |

The majority class baseline catches zero fraud — it just approves everything. Logistic regression catches more fraud but generates 32,497 false alarms. These serve as the floor that ensemble models need to beat.

---

## Ensemble Models

Three ensemble models were trained from scratch on the preprocessed data.

| Model | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|
| Random Forest | 0.9933 | 0.91 | 0.27 | 0.41 |
| **XGBoost** | **0.9957** | **0.88** | **0.45** | **0.59** |
| AdaBoost | 0.9877 | 0.77 | 0.31 | 0.44 |

**XGBoost** was selected as the best model — highest AUC-ROC and F1-score, with the best balance between catching fraud and avoiding false alarms.

### Model Configurations

- **Random Forest** — 100 trees, max depth 20, balanced class weights
- **XGBoost** — 200 trees, learning rate 0.1, max depth 6, subsample 0.8
- **AdaBoost** — 100 shallow decision trees, learning rate 0.5

---

## Threshold Tuning

By default, a model flags a transaction as fraud if its probability score is above 0.50. This is arbitrary. Instead, every threshold from 0.01 to 0.99 was tested and three optimal cutoffs were identified:

| Threshold | Value | Fraud Caught | False Alarms | Purpose |
|---|---|---|---|---|
| High Recall | ~0.05 | 3,739 | 36,452 | Catch as much fraud as possible |
| High Precision | 0.69 | 3,125 | 2,002 | Only flag high-confidence fraud |
| Best F1 | 0.87 | 2,752 | 612 | Best balance of precision and recall |

The High Precision (0.69) and Best F1 (0.87) thresholds were used to define the three-tier policy below.

---

## Prescriptive Policy

Rather than a binary fraud/not-fraud output, the two thresholds define three business actions for every transaction:

```
Score:   0 ──────────────|─────────────|──────────────── 1
                        0.69          0.87

Action:       ALLOW    |    REVIEW   |     BLOCK
```

- **ALLOW** — score below 0.69 → auto-approve, low fraud risk
- **REVIEW** — score between 0.69 and 0.87 → send to human analyst
- **BLOCK** — score above 0.87 → auto-decline, high confidence fraud

### Results on Test Set

| Decision | Transactions | Actual Fraud | Fraud Rate |
|---|---|---|---|
| ALLOW | 690,528 (99.2%) | 663 | 0.1% |
| REVIEW | 1,871 (0.3%) | 391 | 20.9% |
| BLOCK | 3,364 (0.5%) | 2,752 | 81.8% |

- **72.3%** of all fraud is caught automatically (no human needed)
- **81.8%** of blocked transactions are genuine fraud
- Only **0.8%** of all transactions require any action at all

---

## Key Takeaway

The model correctly approves 99.2% of transactions automatically. Of the remaining 0.8%, it auto-blocks the ones it is most confident about (81.8% precision) and routes uncertain cases to a human reviewer. This minimizes both missed fraud and wrongful declines.

---

## Next Steps

- Evaluate policy performance over time as new fraud patterns emerge
- Explore cost-sensitive optimization (weighting false negatives vs false positives by dollar amount)
- Consider retraining periodically as transaction patterns shift
