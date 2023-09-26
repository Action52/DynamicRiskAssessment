"""
This script serves as an API for ML diagnostics and results.
It provides four endpoints for models prediction, scoring, summary statistics,
and other diagnostics.
"""

import json
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

import scoring

# Import your custom modules
from diagnostics import (
    dataframe_summary,
    execution_time,
    missing_data,
    outdated_packages_list,
)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

# Load config.json and set up variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_input_path = os.path.join(config["input_folder_path"])
dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["prod_deployment_path"])

df = pd.read_csv(f"{dataset_csv_path}/finaldata.csv", index_col=False)

with open(f"{model_path}/trainedmodel.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    """Returns models predictions as a list."""
    file_location = request.json["file_location"]
    df = pd.read_csv(file_location, index_col=False)
    X = df[[col for col in df.columns if col not in ["exited", "corporation"]]]
    prediction_result = model.predict(X).tolist()
    return jsonify({"prediction": prediction_result}), 200


@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    """Returns the F1 score of the models."""
    scoring_result = scoring.score_model(df, model, log_folder=model_path)
    return jsonify({"score": scoring_result}), 200


@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summary_stats():
    """Returns summary statistics."""
    summary_statistics = dataframe_summary(df)
    return jsonify({"summary_stats": summary_statistics}), 200


@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    """Returns timing, missing data, and outdated package information."""

    timing_results = execution_time(
        input_folder_path=dataset_input_path,
        output_folder_path=dataset_csv_path,
        output_model_folder=model_path)
    missing_data_results = missing_data(df)
    outdated_packages = outdated_packages_list()
    return (
        jsonify(
            {
                "timing": timing_results,
                "missing_data": missing_data_results,
                "outdated_packages": outdated_packages,
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
