# saving the XGBoost model that will be deployed and its thresholds

import joblib
import json
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modelling.features import add_features, load_data
from modelling.preprocessing import build_preprocessor
from modelling.evaluation import threshold_sweep, evaluate_model


def train():
    df = load_data("cleaned_data_files")
    df = add_features(df)

    for col in ["trans_datetime", "cc_num"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    class_counts = y_train.value_counts()
    n_neg = class_counts.get(0, 0)
    n_pos = class_counts.get(1, 1)
    if n_pos == 0:
        scale_pos = 1.0
    else:
        scale_pos = n_neg / n_pos


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

    print("Training XGBoost model...")
    xgb.fit(X_train, y_train)

    # threshold sweep on XGBoost
    y_proba = xgb.predict_proba(X_test)[:, 1]
    df_thr = threshold_sweep(y_test, y_proba, label="XGBoost")

    # save results
    with open("thresholds_xgb.json", "w") as f:
        json.dump(df_thr.to_dict(orient="records"), f, indent=2)

    # save full pipeline
    joblib.dump(xgb, "model.joblib")

    print("Model saved as model.joblib and threshold saved as threshold.json.")


if __name__ == "__main__":
    train()