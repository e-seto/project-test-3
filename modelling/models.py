from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocessing import build_preprocessor

HAS_XGB = False
try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except Exception:
    HAS_XGB = False


def build_models(X_train):
    preprocessor = build_preprocessor(X_train)

    log_reg = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000, solver="liblinear")),
        ]
    )

    def make_imb_pipeline(est):
        return ImbPipeline(
            steps=[
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", est),
            ]
        )

    rf = make_imb_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
    )

    ada = make_imb_pipeline(
        AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42,
        )
    )

    models = {
        "LogReg-baseline": log_reg,
        "RandomForest": rf,
        "AdaBoost": ada,
    }

    if HAS_XGB:
        xgb = make_imb_pipeline(
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
            )
        )
        models["XGBoost"] = xgb

    return models

