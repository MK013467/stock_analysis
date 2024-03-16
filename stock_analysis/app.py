from flask import Flask, request, render_template
import numpy as np
import pickle as pkl
from src.pipeline import predict_pipeline

## Route for a homepage
from tensorflow.keras.models import load_model
import os
from src.exception_handler import CustomException
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route("/predict", methods=['POST'])
def predict():

    ticker = request.form.get("ticker")
    pipeline = PredictPipeline(ticker)
    try:
        prediction = pipeline.predict()[-1]
        if prediction == 0:
            prediction = "Up"
        else: prediction = "Down"
        return render_template("result.html", prediction=prediction)
    except CustomException as e:
        return render_template("error.html")


if __name__ == "__main__":
    app.run( host="0.0.0.0", port=5000)