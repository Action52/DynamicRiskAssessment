import json
import os
import shutil
import logging

"""
This script deploys a pre-trained models, its associated F1 score, 
and a list of ingested files to a production deployment directory.
"""


def load_config():
    """Load configuration from config.json file."""
    with open("config.json", "r") as f:
        return json.load(f)


config = load_config()
prod_deployment_path = os.path.join(config["prod_deployment_path"])
model_path = os.path.join(config["output_model_path"])
dataset_csv_path = os.path.join(config["output_folder_path"])


def store_model_into_pickle(prod_folder, model_folder, model_name, scores_name,
                            dataset_folder, dataset_log_name):
    """Copy the trained models, F1 score, and list of ingested files to the
    deployment directory."""
    logging.info(f"Moving trained model from {model_folder}/{model_name} to {prod_folder}/{model_name}")
    shutil.copy(
        f"{model_folder}/{model_name}",
        f"{prod_folder}/{model_name}",
    )
    logging.info(f"Moving latest scores from {model_folder}/{scores_name} to {prod_folder}/{scores_name}")
    shutil.copy(
        f"{model_folder}/{scores_name}",
        f"{prod_folder}/{scores_name}",
    )
    logging.info(f"Moving latest scores from {dataset_folder}/{dataset_log_name} to {prod_folder}/{dataset_log_name}")
    shutil.copy(
        f"{dataset_folder}/{dataset_log_name}",
        f"{prod_folder}/{dataset_log_name}",
    )


if __name__ == "__main__":
    store_model_into_pickle(prod_folder=prod_deployment_path,
                            model_folder=model_path,
                            model_name="trainedmodel.pkl",
                            scores_name="latestscore.txt",
                            dataset_folder=dataset_csv_path,
                            dataset_log_name="ingestedfiles.txt")
