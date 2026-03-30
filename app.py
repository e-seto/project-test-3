# Flask app for deployment

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# load trained pipeline
model = joblib.load(MODEL_PATH)

# thresholds
LEGIT_THRESHOLD = 0.60
REVIEW_THRESHOLD = 0.85

# routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/descriptive", methods=["GET"])
def descriptive():
    # TODO: replace with actual queries and visualizations
    return render_template("descriptive.html")


@app.route("/diagnostic", methods=["GET"])
def diagnostic():
    return render_template("diagnostic.html")


@app.route("/predictive", methods=["GET", "POST"])
def predictive():
    if request.method == "GET":
        # render the predictive page
        return render_template("predictive.html")

    if request.method == "POST":
        try:
            # check if the request contains JSON
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            # get the transaction data
            data = request.get_json()
            if not isinstance(data, dict):
                return jsonify({"error": "Request body must be a JSON object"}), 400

            # convert to data for model
            df = pd.DataFrame([data])

            # predict probability
            proba = model.predict_proba(df)[0][1]

            # decision based on thresholds
            if proba < LEGIT_THRESHOLD:
                decision = "legit"
            elif proba < REVIEW_THRESHOLD:
                decision = "manual_review"
            else:
                decision = "block"

            # return the prediction and decision as a JSON response
            return jsonify({
                "fraud_probability": float(proba),
                "decision": decision,
            })

        except Exception as e:
            # log the error
            print(f"Error in prediction: {str(e)}")
            return jsonify({"error": str(e)}), 400


@app.route("/prescriptive", methods=["GET"])
def prescriptive():
    # TODO: prescriptive insights based on model's predictions
    return render_template("prescriptive.html")


if __name__ == "__main__":
    app.run(debug=True)
