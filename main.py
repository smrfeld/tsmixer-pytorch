import argparse
from utils.load_etdataset import load_etdataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True, choices=["train"])
    args = parser.parse_args()

    load_etdataset("../pytorch-tsmixer-data/ETTh1.csv")