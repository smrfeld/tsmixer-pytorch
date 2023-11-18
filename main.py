import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True, choices=["train"])
    args = parser.parse_args()
