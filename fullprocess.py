import json
import os

import pandas as pd

import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import apicalls
import pickle

# Read config.json and set up variables
with open("config.json", "r") as f:
    config = json.load(f)

prod_deployment_path = config["prod_deployment_path"]
input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
output_model_path = config["output_model_path"]

################## Check and read new data
# Read ingestedfiles.txt
with open(f"{prod_deployment_path}/ingestedfiles.txt", "r") as f:
    ingested_files = f.readlines()

# Check for new data files
new_files = []
for line in ingested_files:
    for f in os.listdir(input_folder_path):
        if f not in line:
            new_files.append(f)
            break


################## Deciding whether to proceed, part 1
if not new_files:
    print("No new data found. Exiting.")
    exit(0)
else:
    print("New files: ", new_files)

# Ingest new data if found
# Your ingestion.py code here
ingestion.merge_multiple_dataframe(
    input_folder_path=input_folder_path,
    output_folder_path=output_folder_path
)

# Latest dataset and model
df = pd.read_csv(f"{output_folder_path}/finaldata.csv")
with open(f"{prod_deployment_path}/trainedmodel.pkl", "rb") as model_file:
    model = pickle.load(model_file)

################## Checking for model drift
# Read the latest score
with open(f"{prod_deployment_path}/latestscore.txt", "r") as f:
    latest_score = float(f.read().strip())

# Get new score
diagnostics.model_predictions(df, model)
new_score = scoring.score_model(df, model, log_folder=output_model_path)
print("Latest score: ", latest_score)
print("New score: ", new_score)

################## Deciding whether to proceed, part 2
if new_score >= latest_score:
    print("No model drift detected. Exiting.")
    exit(0)

################## Re-training
# Train new model
training.train_model(df, output_folder=output_model_path,
                     output_file_name="trainedmodel.pkl")

################## Re-deployment
# Deploy new model
deployment.store_model_into_pickle(
    prod_folder=prod_deployment_path,
    model_folder=output_model_path,
    model_name="trainedmodel.pkl",
    scores_name="latestscore.txt",
    dataset_folder=output_folder_path,
    dataset_log_name="ingestedfiles.txt"
)

with open(f"{prod_deployment_path}/trainedmodel.pkl", "rb") as model_file:
    model = pickle.load(model_file)

################## Diagnostics and reporting
# Run diagnostics
diagnostics.run_diagnosis(
    df,
    model,
    input_folder_path,
    output_folder_path,
    output_model_path
)

# Run reporting and apicalls
reporting.score_model(
    df,
    model,
    output_model_path=prod_deployment_path,
    output_model_file="confusionmatrix2.png"
)
results = apicalls.run_api(output_name="apireturns2.txt")
print(results)
