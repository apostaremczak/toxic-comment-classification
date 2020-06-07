"""
Script for submitting a CSV file with comments to the classifier
"""
import argparse
import json
import numpy as np
import pandas as pd
import requests
from typing import Tuple

from model.classifier import get_tokenizer
from utils.preprocessing import tokenize_comments

SERVER_URL = "http://104.248.37.249:8051"


def preprocess_input(input_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    comments = pd.read_csv(input_file_path, header=None)[0].values
    tokenizer = get_tokenizer()
    return tokenize_comments(comments, tokenizer)


def get_predictions(input_ids, mask_ids):
    # Convert input data into appropriate JSON objects
    data = {
        "signature_name": "predict",
        "inputs": {
            "input_token": input_ids.tolist(),
            "masked_token": mask_ids.tolist()
        }
    }

    prediction_url = f"{SERVER_URL}/v1/models/classifier"
    response = requests.post(prediction_url, json=data)

    if not response.ok:
        raise RuntimeError(f"Failed to submit data due to {response.reason}")

    return response.json()["outputs"]


def save_predictions(predictions, output_file_name):
    df = pd.DataFrame(data={"predictions": predictions})
    df.to_csv(output_file_name, sep=',', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Submit text data to a toxic comment classifier"
    )

    parser.add_argument("input_file", type=str,
                        help="Path to a CSV file with input data; "
                             "Each comment should be on a separate line, and "
                             "the file only one column with no header")

    parser.add_argument("output_file", type=str,
                        help="Path to a CSV file where the model's predictions"
                             " will be stored")

    args = parser.parse_args()

    tokenized = preprocess_input(args.input_file)
    predictions = get_predictions(*tokenized)
    save_predictions(predictions, args.output_file)
