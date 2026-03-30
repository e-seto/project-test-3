from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    numeric_cols = [
        "amt",
        "city_pop",
        "age",
        "haversine_km",
        "hour",
        "month",
        "cust_amt_mean",
        "amt_dev_from_mean",
    ]
    categorical_cols = ["gender", "category", "state", "job", "day_of_week", "merchant"]

    numeric_cols = [c for c in numeric_cols if c in X.columns]
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    num_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_cols,
            ),
        ]
    )
    return preprocessor

