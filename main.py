from utils import TSMixer, plot_preds, plot_loss


import argparse
import yaml


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True, choices=["train", "predict", "loss"])
    parser.add_argument("--conf", type=str, required=False, help="Path to the configuration file")
    parser.add_argument("--no-feats-plot", type=int, required=False, default=6, help="Number of features to plot")
    parser.add_argument("--show", action="store_true", required=False, help="Show plots")
    args = parser.parse_args()

    if args.command == "train":
        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixer.Conf.from_dict(yaml.safe_load(f))
        tsmixer = TSMixer(conf)
        tsmixer.train()

    elif args.command == "predict":

        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixer.Conf.from_dict(yaml.safe_load(f))
        tsmixer = TSMixer(conf)

        data = tsmixer.predict_val_dataset(max_samples=10)

        data_plt = data[0]
        assert args.no_feats_plot is not None, "Must provide number of features to plot"
        plot_preds(data_plt["pred"], data_plt["pred_gt"], no_feats_plot=args.no_feats_plot, show=args.show)

    elif args.command == "loss":

        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixer.Conf.from_dict(yaml.safe_load(f))
        tsmixer = TSMixer(conf)

        train_data = tsmixer.load_training_metadata_or_new()
        plot_loss(train_data, show=args.show)

    else:
        raise NotImplementedError(f"Command {args.command} not implemented")

