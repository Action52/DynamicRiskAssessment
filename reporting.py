"""
This script calculates the confussion matrix and stores it as png.
"""

import json
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
output_model_path = os.path.join(config["output_model_path"])
test_data_path = os.path.join(config["test_data_path"])


def score_model(
        df_test: pd.DataFrame,
        model: LogisticRegression,
        output_model_path,
        output_model_file="confusionmatrix.png"):
    """
    Calculate a confusion matrix using the test data and the deployed models.
    Write the confusion matrix to the workspace.
    """
    X_test = df_test[
        [col for col in df_test.columns if col not in ["exited", "corporation"]]
    ]
    y_test = df_test["exited"]

    # Get predictions
    y_pred = model.predict(X_test)

    # Generate confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save confusion matrix plot
    plt.savefig(f"{output_model_path}/{output_model_file}")


if __name__ == "__main__":
    # Load the deployed models
    with open(f"{output_model_path}/trainedmodel.pkl", "rb") as modelfile:
        model = pickle.load(modelfile)

    # Load the test data
    df_test = pd.read_csv(f"{test_data_path}/testdata.csv")
    score_model(df_test, model, output_model_path=output_model_path)
