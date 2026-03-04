import logging
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

from .features import add_features, load_data
from .evaluation import evaluate_model, print_feature_importance, threshold_sweep
from .models import build_models


def main():
    log_dir = Path("outputs") / "run-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    root.handlers = [file_handler, stream_handler]

    logger = logging.getLogger("modelling")

    logger.info("Loading data")
    df = load_data("cleaned_data_files")
    df = add_features(df)

    for col in ["trans_datetime", "cc_num"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)

    logger.info("Train/test split (stratified 80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models(X_train)

    results = {}
    for name, model in models.items():
        logger.info("Fitting %s", name)
        model.fit(X_train, y_train)
        y_proba = evaluate_model(name, model, X_test, y_test)
        results[name] = y_proba
        if "RandomForest" in name or "XGBoost" in name:
            print_feature_importance(name, model)

    for key in ["RandomForest", "XGBoost", "AdaBoost"]:
        if key in results:
            threshold_sweep(y_test, results[key], label=key)
            break


if __name__ == "__main__":
    main()

