import argparse
from utils.load_etdataset import load_etdataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True, choices=["train"])
    parser.add_argument("--conf", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    load_etdataset("../pytorch-tsmixer-data/ETTh1.csv")