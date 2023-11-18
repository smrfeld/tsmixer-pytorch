import argparse
from utils import TSMixer
import yaml
import json
from tqdm import tqdm
from loguru import logger
from typing import List, Tuple


def plot_preds(preds: List[List[float]], preds_gt: List[List[float]]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    no_feats = len(preds[0])
    no_feats = min(no_feats, 3)
    fig = make_subplots(rows=1, cols=no_feats)

    for ifeat in range(no_feats):
        fig.add_trace(go.Scatter(y=[pred[ifeat] for pred in preds], mode="lines", name=f"pred_{ifeat}"), row=1, col=ifeat+1)
        fig.add_trace(go.Scatter(y=[pred[ifeat] for pred in preds_gt], mode="lines", name=f"pred_gt_{ifeat}"), row=1, col=ifeat+1)

    fig.update_layout(
        height=400, 
        width=1200, 
        title_text="Predictions",
        font=dict(size=18),
        )

    fig.show()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True, choices=["train", "predict", "loss"])
    parser.add_argument("--conf", type=str, required=False, help="Path to the configuration file")
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

        # Change batch size to 1 and not shuffle data for consistency
        conf.batch_size = 1
        conf.shuffle = False

        tsmixer = TSMixer(conf)
        _, loader_val = tsmixer.create_data_loaders_train_val()
        
        data_json = []
        for _ in tqdm(range(10), desc="Predicting"):
            batch_input, batch_pred = next(iter(loader_val))
            batch_pred_hat = tsmixer.predict(batch_input)
            data_json.append({
                "input": batch_input.tolist()[0],
                "pred": batch_pred.tolist()[0],
                "pred_hat": batch_pred_hat.tolist()[0]
                })

        data_plt = data_json[0]
        plot_preds(data_plt["pred_hat"], data_plt["pred"])

        with open("data.json", "w") as f:
            json.dump(data_json, f)
            logger.info(f"Saved data to data.json")

    elif args.command == "loss":

        assert args.conf is not None, "Must provide a configuration file"
        with open(args.conf, "r") as f:
            conf = TSMixer.Conf.from_dict(yaml.safe_load(f))
        tsmixer = TSMixer(conf)

        # Plot loss
        train_data = tsmixer.load_training_metadata_or_new()

        import plotly.graph_objects as go
        fig = go.Figure()
        x = [ epoch for epoch in train_data.epoch_to_data.keys() ]
        y = [ data.val_loss for data in train_data.epoch_to_data.values() ]
        fig.add_trace(go.Scatter(x, y, mode="lines", name="Val. loss"))
        y = [ data.train_loss for data in train_data.epoch_to_data.values() ]
        fig.add_trace(go.Scatter(x, y, mode="lines", name="Train loss"))

        fig.update_layout(
            height=400, 
            width=1200, 
            title_text="Loss",
            font=dict(size=18),
            )
        fig.show()

    else:
        raise NotImplementedError(f"Command {args.command} not implemented")

