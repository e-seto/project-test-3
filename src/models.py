from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor

HAS_XGB = False
try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except Exception:
    HAS_XGB = False


def build_models(X_train, y_train):
    preprocessor = build_preprocessor(X_train)

    class_counts = y_train.value_counts()
    n_neg = class_counts.get(0, 0)
    n_pos = class_counts.get(1, 1)
    if n_pos == 0:
        scale_pos = 1.0
    else:
        scale_pos = n_neg / n_pos

    log_reg = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    class_weight="balanced",
                ),
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    ada_base = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
        class_weight="balanced",
    )

    ada = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                AdaBoostClassifier(
                    estimator=ada_base,
                    n_estimators=200,
                    learning_rate=0.5,
                    random_state=42,
                ),
            ),
        ]
    )

    models = {
        "LogReg-baseline": log_reg,
        "RandomForest": rf,
        "AdaBoost": ada,
    }

    if HAS_XGB:
        xgb = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=5,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        tree_method="hist",
                        random_state=42,
                        scale_pos_weight=scale_pos,
                    ),
                ),
            ]
        )
        models["XGBoost"] = xgb

    return models

