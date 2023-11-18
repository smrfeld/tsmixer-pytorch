from .model import TSMixerModel

from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from enum import Enum
import os
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import torch
from loguru import logger
from tqdm import tqdm


class TSMixer:

    @dataclass
    class Conf(DataClassDictMixin):

        class Initialize(Enum):
            FROM_LATEST_CHECKPOINT = "from-latest-checkpoint"
            "Load the model from the latest checkpoint"

            FROM_BEST_CHECKPOINT = "from-best-checkpoint"
            "Load the model from the best checkpoint"

            FROM_SCRATCH = "from-scratch"
            "Initialize the model from scratch"

        class DataSrc(Enum):

            CSV_FILE = "csv-file"
            "Load the dataset from a CSV file"

        class ValidationSplit(Enum):
            
            TEMPORAL_HOLDOUT = "temporal-holdout"
            "Reserve the last portion (e.g., 10-20%) of your time-ordered data for validation, and use the remaining data for training. This is a simple and widely used approach."

        class Loss(Enum):

            MSE = "mse"
            "Mean squared error"

            MAE = "mae"
            "Mean absolute error"

        output_dir: str
        "Directory where to save checkpoints and generated images"

        input_length: int
        "Number of time steps to use as input"

        no_features: int
        "Number of features in the dataset"

        no_mixer_layers: int
        "Number of mixer layers"

        prediction_length: int
        "Number of time steps to predict"

        data_src: DataSrc = DataSrc.CSV_FILE
        "Where to load the dataset from"

        data_src_csv: Optional[str] = None
        "Path to the CSV file to load the dataset from. Only used if data_src is CSV_FILE"

        batch_size: int = 64
        "Batch size"

        num_epochs: int = 10
        "Number of epochs to train for"

        learning_rate: float = 0.001
        "Learning rate"

        optimizer: str = "Adam"
        "Optimizer to use"

        random_seed: int = 42
        "Random seed for reproducibility"

        validation_split: ValidationSplit = ValidationSplit.TEMPORAL_HOLDOUT
        "How to split the data into training and validation"

        validation_split_holdout: float = 0.2
        "Use the last X% of the data for validation. Only used for TEMPORAL_HOLDOUT"

        initialize: Initialize = Initialize.FROM_SCRATCH
        "How to initialize the model"

        loss: Loss = Loss.MSE
        "Loss function to use"

        @property
        def checkpoint_init(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "init.pth")

        @property
        def checkpoint_best(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "best.pth")

        @property
        def checkpoint_latest(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "latest.pth")

        def check_valid(self):
            assert 0 <= self.validation_split_holdout <= 1, "validation_split_holdout must be between 0 and 1"


    def __init__(self, conf: Conf):
        conf.check_valid()
        self.conf = conf

        # Create the model
        self.model = TSMixerModel(
            input_length=self.conf.input_length,
            forecast_length=self.conf.prediction_length,
            no_feats=self.conf.no_features,
            no_mixer_layers=self.conf.no_mixer_layers
            )


    def load_data_train_val(self) -> Tuple[DataLoader, DataLoader]:

        if self.conf.data_src == self.conf.DataSrc.CSV_FILE:
            assert self.conf.data_src_csv is not None, "data_src_csv must be set if data_src is CSV_FILE"

            from .load_etdataset import load_etdataset, ValidationSplit
            return load_etdataset(
                csv_file=self.conf.data_src_csv,
                batch_size=self.conf.batch_size,
                input_length=self.conf.input_length,
                prediction_length=self.conf.prediction_length,
                val_split=ValidationSplit(self.conf.validation_split.value),
                val_split_holdout=self.conf.validation_split_holdout
                )
        else:
            raise NotImplementedError(f"data_src {self.conf.data_src} not implemented")


    def train(self):

        # Create the optimizer
        optimizer_cls = getattr(torch.optim, self.conf.optimizer)
        optimizer = optimizer_cls(self.model.parameters(), lr=self.conf.learning_rate)

        # Create the loaders
        loader_train, loader_val = self.load_data_train_val()

        # Train
        for epoch in range(self.conf.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.conf.num_epochs}")

            for batch_input, batch_pred in tqdm(loader_train, desc="Training batches"):
                self._train_step(batch_input, batch_pred, optimizer)


    def _train_step(self, batch_input: torch.Tensor, batch_pred: torch.Tensor, optimizer: torch.optim.Optimizer):
        self.model.train()

        # Forward pass
        batch_pred_hat = self.model(batch_input)

        # Compute loss
        if self.conf.loss == self.conf.Loss.MSE:
            loss = torch.nn.functional.mse_loss(batch_pred_hat, batch_pred)
        elif self.conf.loss == self.conf.Loss.MAE:
            loss = torch.nn.functional.l1_loss(batch_pred_hat, batch_pred)
        else:
            raise NotImplementedError(f"Loss {self.conf.loss} not implemented")

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()