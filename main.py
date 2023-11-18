import argparse
from utils import TSMixer
import yaml

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True, choices=["train"])
    parser.add_argument("--conf", type=str, required=False, help="Path to the configuration file")
    args = parser.parse_args()

    if args.command == "train":
        assert args.conf is not None, "Must provide a configuration file"

        with open(args.conf, "r") as f:
            conf = TSMixer.Conf.from_dict(yaml.safe_load(f))

        tsmixer = TSMixer(conf)
        tsmixer.train()