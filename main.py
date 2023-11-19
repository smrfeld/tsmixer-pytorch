from utils import TSMixer, plot_preds, plot_loss, TSMixerConf

import argparse
import yaml
import os


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
            conf = TSMixerConf.from_dict(yaml.safe_load(f))
        tsmixer = TSMixer(conf)
        tsmixer.train()

    elif args.command == "predict":

        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixerConf.from_dict(yaml.safe_load(f))

        # Load best checkpoint
        conf.initialize = TSMixerConf.Initialize.FROM_BEST_CHECKPOINT

        tsmixer = TSMixer(conf)

        data = tsmixer.predict_val_dataset(max_samples=10, save_inputs=False)

        data_plt = data[0]
        assert args.no_feats_plot is not None, "Must provide number of features to plot"
        plot_preds(data_plt["pred"], data_plt["pred_gt"], no_feats_plot=args.no_feats_plot, show=args.show, fname_save=os.path.join(conf.image_dir, "preds.png"))

    elif args.command == "loss":

        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixerConf.from_dict(yaml.safe_load(f))

        train_data = conf.load_training_metadata_or_new()
        plot_loss(train_data, show=args.show, fname_save=os.path.join(conf.image_dir, "loss.png"))

    else:
        raise NotImplementedError(f"Command {args.command} not implemented")

