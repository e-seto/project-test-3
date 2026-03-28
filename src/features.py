from pathlib import Path

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def load_data(data_dir="data/raw"):
    data_path = Path(data_dir)
    files = list(data_path.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found under {data_dir}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def add_features(df):
    df = df.copy()
    df["trans_datetime"] = pd.to_datetime(df["trans_datetime"])
    df["hour"] = df["trans_datetime"].dt.hour
    df["day_of_week"] = df["trans_datetime"].dt.day_name()
    df["month"] = df["trans_datetime"].dt.month
    df["haversine_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    df["cust_amt_mean"] = df.groupby("cc_num")["amt"].transform("mean")
    df["amt_dev_from_mean"] = df["amt"] - df["cust_amt_mean"]
    return df

