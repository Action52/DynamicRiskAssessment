import shutil
import pandas as pd
import os
import json
from datetime import datetime
import logging

"""
This script merges multiple CSV files from a directory into a single CSV file.
It uses a configuration file 'config.json' to get the input and output folder 
paths.
"""

# Read configuration from 'config.json'
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


def merge_multiple_dataframe(input_folder_path=None,
                             output_folder_path=None,
                             log_name="ingestedfiles.txt",
                             output_name="finaldata.csv"):
    """
    Merges multiple CSV files from 'input_folder_path' into a single CSV file
    at 'output_folder_path'.
    It also creates an 'ingestedfiles.txt' that logs the datetime and file paths
    for each merged file.
    """
    logging.info("Starting ingestion.")

    # Create and write header for the log file.
    with open(f"{output_folder_path}/{log_name}", "w") as ingestedfiles:
        dfs = []
        # Loop through each file in the input folder
        for file in os.listdir(input_folder_path):
            logging.debug(f"Ingesting file {file}")
            input_path = os.path.join(input_folder_path, file)
            output_path = os.path.abspath(
                os.path.join(output_folder_path, output_name)
            )

            # Read and append each CSV file into 'finaldata.csv'
            df = pd.read_csv(input_path, index_col=False)
            dfs.append(df)

            # Log the datetime, origin and destination file paths
            ingestedfiles.write(
                f"{input_path}\n"
            )
        final_df = pd.concat(dfs, ignore_index=True)
        # Dedupe
        final_df.drop_duplicates(inplace=True)
        final_df.to_csv(output_path, index=False)
        logging.info(f"Saving joined dataframe to {output_path}")
        return final_df


if __name__ == "__main__":
    merge_multiple_dataframe(input_folder_path=input_folder_path,
                             output_folder_path=output_folder_path)
