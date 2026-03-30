# Flask app for deployment

import os
from flask import Flask, redirect, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# thresholds
LEGIT_THRESHOLD = 0.60
REVIEW_THRESHOLD = 0.85

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


@app.route("/predictive", methods=["GET"])
def predictive():
    preditive_link = "https://project-wyfi.onrender.com/predictive"
    return redirect(preditive_link)



@app.route("/prescriptive", methods=["GET"])
def prescriptive():
    return render_template("prescriptive.html")



