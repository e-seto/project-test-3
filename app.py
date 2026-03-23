# Flask app for deployment

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from flask import Flask, render_template, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# load trained pipeline
model = joblib.load("model.joblib")

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
    fraud_rate = 0.55
    total_transactions = 3500000
    fraudulent_transactions = total_transactions * (fraud_rate / 100)

    return render_template("descriptive.html", fraud_rate=fraud_rate, total_transactions=total_transactions, fraudulent_transactions=fraudulent_transactions)

@app.route("/predictive", methods=["GET", "POST"])
def predictive():
    if request.method == "GET":
        # render the predictive page
        return render_template('predictive.html')
    
    if request.method == "POST":
        try:
            # check if the request contains JSON
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400

            # get the transaction data
            data = request.get_json()

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
                "decision": decision
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