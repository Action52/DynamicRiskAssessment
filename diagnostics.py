"""
This script performs diagnostic checks on the models and dataset.
"""

import json
import os
import pickle
import subprocess
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression

from ingestion import merge_multiple_dataframe
from training import train_model

import logging


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


config = load_config()
dataset_output_csv_path = os.path.join(config["output_folder_path"])
prod_deploy_path = os.path.join(config["prod_deployment_path"])
dataset_input_csv_path = os.path.join(config["input_folder_path"])
model_folder = os.path.join(config["output_model_path"])

def model_predictions(df: pd.DataFrame, model: LogisticRegression):
    """Get models predictions."""
    X = df[[col for col in df.columns if col not in ["exited", "corporation"]]]
    return list(model.predict(X))


def dataframe_summary(df: pd.DataFrame):
    """Get summary statistics for specific columns."""
    num_cols = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees",
    ]
    return [
        ("MEANS",dict(df[num_cols].mean())),
        ("MEDIANS",dict(df[num_cols].median())),
        ("STDS",dict(df[num_cols].std())),
    ]


def missing_data(df: pd.DataFrame) -> list:
    """Count the percentage of NA values in each column."""
    missing_count = df.isna().sum()
    return ((missing_count / df.shape[0]) * 100).tolist()


def execution_time(
        input_folder_path,
        output_folder_path,
        output_model_folder,
        output_model_file="trainedmodel.pkl",
        log_name="ingestedfiles.txt",
        output_name="finaldata.csv") -> list:
    """Measure execution time for data ingestion and models training."""
    start_ingest = time.time()
    merged_df = merge_multiple_dataframe(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path,
        log_name=log_name,
        output_name=output_name)
    end_ingest = time.time()

    start_train = time.time()
    train_model(merged_df,
                output_folder=output_model_folder,
                output_file_name=output_model_file)
    end_train = time.time()

    return [end_ingest - start_ingest, end_train - start_train]


def outdated_packages_list():
    """List outdated Python packages."""
    result = subprocess.run(
        ["pip", "list", "--outdated"], capture_output=True, text=True
    )
    lines = result.stdout.strip().split("\n")[2:]
    data = [[line.split()[i] for i in [0, 1, 3]] for line in lines]
    return data


def run_diagnosis(
        df: pd.DataFrame,
        model: LogisticRegression,
        input_folder_path,
        output_folder_path,
        output_model_folder,
        output_model_file="trainedmodel.pkl",
        log_name="ingestedfiles.txt",
        output_name="finaldata.csv"
):
    preds = model_predictions(df, model)
    summary = dataframe_summary(df)
    missing = missing_data(df)
    exec_time = execution_time(input_folder_path=input_folder_path,
                   output_folder_path=output_folder_path,
                   output_model_folder=output_model_folder,
                   output_model_file=output_model_file,
                   log_name=log_name,
                   output_name=output_name)
    outdated_packages = outdated_packages_list()
    return (preds, summary, missing_data, exec_time, outdated_packages)


if __name__ == "__main__":
    df = pd.read_csv(f"{dataset_output_csv_path}/finaldata.csv",
                     index_col=False)
    with open(f"{prod_deploy_path}/trainedmodel.pkl", "rb") as f:
        model = pickle.load(f)
    diagnosis = run_diagnosis(
        df=df,
        model=model,
        input_folder_path=dataset_input_csv_path,
        output_folder_path=dataset_output_csv_path,
        output_model_folder=model_folder
    )
    print(diagnosis)
