import json
import os
import pickle

import pandas as pd
from sklearn import metrics

import logging

"""
This script is used for scoring a pre-trained logistic regression models.
It loads test data and calculates an F1 score for the models relative to the 
test data. The F1 score is then written to a file called 'latestscore.txt'.
"""


def load_config():
    """Load configuration from config.json file."""
    with open("config.json", "r") as f:
        return json.load(f)


config = load_config()
model_path = os.path.join(config["output_model_path"])
test_data_path = os.path.join(config["test_data_path"])


def score_model(df, model, log_folder=None, log_file_name="latestscore.txt"):
    logging.info("Running scoring script")
    """Load a trained models, score it against test data, and write the F1 score
    to 'latestscore.txt'."""

    X = df[[col for col in df.columns if col not in ["exited", "corporation"]]]
    y_true = df["exited"]
    y_pred = model.predict(X)

    f1_score = metrics.f1_score(y_true, y_pred)
    logging.info("F1 Score: ", f1_score)

    with open(f"{log_folder}/{log_file_name}", "w") as score_file:
        score_file.write(f"{f1_score}\n")
    return f1_score


if __name__ == "__main__":
    df = pd.read_csv(f"{test_data_path}/testdata.csv")
    with open(f"{model_path}/trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)
    score_model(df, model, log_folder=model_path)
