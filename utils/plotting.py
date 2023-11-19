from .tsmixer_conf import TrainingMetadata

from typing import List, Tuple, Optional
from loguru import logger


def plot_preds(preds: List[List[float]], preds_gt: List[List[float]], no_feats_plot: int, fname_save: Optional[str] = None, inputs: Optional[List[List[float]]] = None, show: bool = True):
    """Plot predictions

    Args:
        preds (List[List[float]]): Predictions of shape (no_samples, no_feats)
        preds_gt (List[List[float]]): Predictions of shape (no_samples, no_feats)
        no_feats_plot (int): Number of features to plot
        fname_save (Optional[str], optional): File name to save the plot. Defaults to None.
        inputs (Optional[List[List[float]]], optional): Input of shape (no_samples, no_feats)
        show (bool): Show the plot
    """    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    no_feats = len(preds[0])
    if no_feats_plot > no_feats:
        logger.warning(f"no_feats_plot ({no_feats_plot}) is larger than no_feats ({no_feats}). Setting no_feats_plot to no_feats")
        no_feats_plot = no_feats

    no_cols = 3
    no_rows = int(no_feats_plot / no_cols)
    if no_feats_plot % no_cols != 0:
        no_rows += 1

    fig = make_subplots(rows=no_rows, cols=no_cols, subplot_titles=[f"Feature {ifeat}" for ifeat in range(no_feats_plot)])

    no_inputs = len(inputs) if inputs is not None else 0
    x_preds = list(range(no_inputs, no_inputs + len(preds)))
    for ifeat in range(no_feats_plot):
        row = int(ifeat / no_cols) + 1
        col = (ifeat % no_cols) + 1

        if inputs is not None:
            x_inputs = list(range(len(inputs)))
            fig.add_trace(go.Scatter(x=x_inputs, y=[in_y[ifeat] for in_y in inputs], mode="lines", name=f"Inputs", line=dict(color="black"), showlegend=ifeat==0), row=row, col=col)

        fig.add_trace(go.Scatter(x=x_preds, y=[pred[ifeat] for pred in preds_gt], mode="lines", name=f"Ground truth", line=dict(color="red"), showlegend=ifeat==0), row=row, col=col)
        fig.add_trace(go.Scatter(x=x_preds, y=[pred[ifeat] for pred in preds], mode="lines", name=f"Model", line=dict(color="blue"), showlegend=ifeat==0), row=row, col=col)

    fig.update_layout(
        height=300*no_rows, 
        width=400*no_cols, 
        title_text="Predictions",
        font=dict(size=18),
        xaxis_title_text="Time",
        yaxis_title_text="Signal",
        )

    if fname_save is not None:
        fig.write_image(fname_save)
        logger.info(f"Saved plot to {fname_save}")

    if show:
        fig.show()

    return fig


def plot_loss(train_data: TrainingMetadata, fname_save: Optional[str] = None, show: bool = True):
    """Plot loss

    Args:
        train_data (TSMixer.TrainingMetadata): Training metadata
        fname_save (Optional[str], optional): File name to save the plot. Defaults to None.
        show (bool): Show the plot
    """    
    import plotly.graph_objects as go

    fig = go.Figure()
    x = [ epoch for epoch in train_data.epoch_to_data.keys() ]
    y = [ data.val_loss for data in train_data.epoch_to_data.values() ]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Val. loss"))
    y = [ data.train_loss for data in train_data.epoch_to_data.values() ]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Train loss"))

    fig.update_layout(
        height=500, 
        width=700, 
        title_text="Loss during training",
        xaxis_title_text="Epoch",
        yaxis_title_text="Loss",
        font=dict(size=18),
        )

    if fname_save is not None:
        fig.write_image(fname_save)
        logger.info(f"Saved plot to {fname_save}")

    if show:
        fig.show()

    return fig