import os
import requests
from loguru import logger
import argparse

DATASET_TO_URL = {
    "ETTh1": "https://github.com/zhouhaoyi/ETDataset/raw/11ab373cf9c9f5be7698e219a5a170e1b1c8a930/ETT-small/ETTh1.csv",
    "ETTh2": "https://github.com/zhouhaoyi/ETDataset/raw/11ab373cf9c9f5be7698e219a5a170e1b1c8a930/ETT-small/ETTh2.csv",
    "ETTm1": "https://github.com/zhouhaoyi/ETDataset/raw/11ab373cf9c9f5be7698e219a5a170e1b1c8a930/ETT-small/ETTm1.csv",
    "ETTm2": "https://github.com/zhouhaoyi/ETDataset/raw/11ab373cf9c9f5be7698e219a5a170e1b1c8a930/ETT-small/ETTm2.csv"
    }

def download_etdataset(url: str, download_directory: str):

    # Make sure the directory exists, create it if not
    if download_directory != "":
        os.makedirs(download_directory, exist_ok=True)

    # Extract the file name from the URL
    file_name = os.path.join(download_directory, os.path.basename(url))

    if os.path.exists(file_name):
        logger.info(f"File {file_name} already exists, skipping download")
        return
    logger.info(f"Downloading {dataset} to {args.dir} from {url}")

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the content of the response to the file
        with open(file_name, "wb") as file:
            file.write(response.content)
        logger.info(f"File downloaded and saved to {file_name}")
    else:
        logger.info(f"Failed to download the file. Status code: {response.status_code}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=False, default="data", help="Directory to save the downloaded files to")
    args = parser.parse_args()

    for dataset,url in DATASET_TO_URL.items():
        download_etdataset(url, download_directory=args.dir)