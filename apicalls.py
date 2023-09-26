import requests
import json
import os


def run_api(output_name="apireturns.txt"):
    # Load config.json and set up variables
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config["output_folder_path"])
    model_path = os.path.join(config["output_model_path"])
    # Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1:8000/"

    # Prepare data for prediction endpoint
    payload_prediction = json.dumps({"file_location": f"testdata/testdata.csv"})

    # Call the Prediction API endpoint and store the response
    response1 = requests.post(f"{URL}prediction", headers={"Content-Type": "application/json"}, data=payload_prediction)

    # Call the Scoring API endpoint and store the response
    response2 = requests.get(f"{URL}scoring")

    # Call the Summary Statistics API endpoint and store the response
    response3 = requests.get(f"{URL}summarystats")

    # Call the Diagnostics API endpoint and store the response
    response4 = requests.get(f"{URL}diagnostics")

    # Combine all API responses
    responses = {
        "prediction": response1.json(),
        "scoring": response2.json(),
        "summary_stats": response3.json(),
        "diagnostics": response4.json(),
    }

    # Write to a text file
    with open(f"{model_path}/{output_name}", "w") as f:
        f.write(json.dumps(responses))

    return responses


if __name__ == "__main__":
    run_api()
