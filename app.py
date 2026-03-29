# Flask app for deployment

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))

# load trained model, label encoders, thresholds, and amt stats
model = joblib.load(os.path.join(BASE_DIR, "models/xgboost.joblib"))
encoders = joblib.load(os.path.join(BASE_DIR, "models/label_encoders.joblib"))
amt_stats = joblib.load(os.path.join(BASE_DIR, "models/amt_stats.joblib"))

with open(os.path.join(BASE_DIR, "models/policy_thresholds.json")) as f:
    thresholds = json.load(f)

REVIEW_THRESHOLD = thresholds["review"]
BLOCK_THRESHOLD = thresholds["block"]

# day name to number mapping
DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}


def prepare_features(data):
    """Transform raw JSON input into the 15 features the model expects."""
    amt = float(data["amt"])
    cust_amt_mean = float(data.get("cust_amt_mean", amt_stats["mean"]))
    cust_amt_std = float(data.get("cust_amt_std", amt_stats["std"]))
    if cust_amt_std == 0:
        cust_amt_std = 1.0

    # label-encode categorical columns
    gender = encoders["gender"].transform([data["gender"]])[0]
    city = encoders["city"].transform([data["city"]])[0]
    state = encoders["state"].transform([data["state"]])[0]
    job = encoders["job"].transform([data["job"]])[0]
    category = encoders["category"].transform([data["category"]])[0]

    # day_of_week: accept name or number
    dow = data.get("day_of_week", 0)
    if isinstance(dow, str) and dow in DAY_MAP:
        dow = DAY_MAP[dow]
    else:
        dow = int(dow)

    row = {
        "gender": gender,
        "city": city,
        "state": state,
        "zip": int(data["zip"]),
        "city_pop": int(data["city_pop"]),
        "job": job,
        "category": category,
        "amt": amt,
        "age": int(data["age"]),
        "haversine_dist_km": float(data.get("haversine_km", data.get("haversine_dist_km", 77.0))),
        "hour": int(data["hour"]),
        "day_of_week": dow,
        "amt_deviation": amt - cust_amt_mean,
        "amt_deviation_ratio": amt / cust_amt_mean if cust_amt_mean != 0 else 1.0,
        "amt_deviation_zscore": (amt - cust_amt_mean) / cust_amt_std,
    }

    return pd.DataFrame([row])


# routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/descriptive", methods=["GET"])
def descriptive():
    return render_template("descriptive.html")


@app.route("/diagnostic", methods=["GET"])
def diagnostic():
    return render_template("diagnostic.html")


@app.route("/predictive", methods=["GET", "POST"])
def predictive():
    if request.method == "GET":
        return render_template("predictive.html")

    if request.method == "POST":
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            data = request.get_json()
            df = prepare_features(data)

            proba = model.predict_proba(df)[0][1]

            if proba < REVIEW_THRESHOLD:
                decision = "legit"
            elif proba < BLOCK_THRESHOLD:
                decision = "manual_review"
            else:
                decision = "block"

            return jsonify({
                "fraud_probability": float(proba),
                "decision": decision,
            })

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return jsonify({"error": str(e)}), 400


@app.route("/prescriptive", methods=["GET"])
def prescriptive():
    return render_template("prescriptive.html")


if __name__ == "__main__":
    app.run(debug=True)
