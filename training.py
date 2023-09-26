import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression

import logging

"""
This script performs the training on the dataset.
"""


def load_config():
    """Load configuration from config.json file."""
    with open("config.json", "r") as f:
        return json.load(f)


config = load_config()

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])


def train_model(df, output_folder=None,
                output_file_name=None):
    """Train a logistic regression models and save it as a pickle file."""
    logging.info("Running training script")
    X = df[[col for col in df.columns if col not in ["exited", "corporation"]]]
    y = df["exited"]
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="ovr",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )
    lr.fit(X, y)

    with open(f"{output_folder}/{output_file_name}", "wb") as f:
        pickle.dump(lr, f)
        logging.info(f"Saved model to {output_folder}/{output_file_name}")


if __name__ == "__main__":
    df = pd.read_csv(f"{dataset_csv_path}/finaldata.csv",
                     index_col=False)
    train_model(df, output_folder=model_path,
                output_file_name="trainedmodel.pkl")
