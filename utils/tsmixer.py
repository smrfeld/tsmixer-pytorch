from .model import TSMixerModel
from .load_csv import DataNormalization

from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from enum import Enum
import os
from typing import Optional, Tuple, Dict, List
from torch.utils.data import DataLoader
import torch
from loguru import logger
from tqdm import tqdm
import json
import time


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

        data_src: DataSrc
        "Where to load the dataset from"

        data_src_csv: Optional[str] = None
        "Path to the CSV file to load the dataset from. Only used if data_src is CSV_FILE"

        batch_size: int = 64
        "Batch size"

        shuffle: bool = True
        "Shuffle the data"

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

        @property
        def train_progress_json(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "loss.json")

        @property
        def pred_val_dataset_json(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "pred_val_dataset.json")

        @property
        def data_norm_json(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "data_norm.json")

        def check_valid(self):
            assert 0 <= self.validation_split_holdout <= 1, "validation_split_holdout must be between 0 and 1"


    @dataclass
    class TrainingMetadata(DataClassDictMixin):

        @dataclass
        class EpochData(DataClassDictMixin):
            epoch: int
            "Epoch number"

            train_loss: float
            "Training loss"

            val_loss: float
            "Validation loss"

            duration_seconds: float
            "Duration of the epoch in seconds"

        epoch_to_data: Dict[int, EpochData]
        "Mapping from epoch number to epoch data"


    def __init__(self, conf: Conf):
        """Constructor for TSMixer class

        Args:
            conf (Conf): Configuration
        """        
        conf.check_valid()
        self.conf = conf

        # Create the model
        self.model = TSMixerModel(
            input_length=self.conf.input_length,
            forecast_length=self.conf.prediction_length,
            no_feats=self.conf.no_features,
            no_mixer_layers=self.conf.no_mixer_layers
            )

        # Load the model
        if self.conf.initialize == self.conf.Initialize.FROM_LATEST_CHECKPOINT:
            self.load_checkpoint(fname=self.conf.checkpoint_latest)
        elif self.conf.initialize == self.conf.Initialize.FROM_BEST_CHECKPOINT:
            self.load_checkpoint(fname=self.conf.checkpoint_best)
        elif self.conf.initialize == self.conf.Initialize.FROM_SCRATCH:
            pass
        else:
            raise NotImplementedError(f"Initialize {self.conf.initialize} not implemented")


    def create_data_loaders_train_val(self, data_norm: Optional[DataNormalization]) -> Tuple[DataLoader, DataLoader, DataNormalization]:
        """Create the training and validation data loaders

        Returns:
            Tuple[DataLoader, DataLoader, DataNormalization]: Training and validation data loaders
        """        

        if self.conf.data_src == self.conf.DataSrc.CSV_FILE:
            assert self.conf.data_src_csv is not None, "data_src_csv must be set if data_src is CSV_FILE"

            from .load_csv import load_csv_dataset, ValidationSplit
            return load_csv_dataset(
                csv_file=self.conf.data_src_csv,
                batch_size=self.conf.batch_size,
                input_length=self.conf.input_length,
                prediction_length=self.conf.prediction_length,
                val_split=ValidationSplit(self.conf.validation_split.value),
                val_split_holdout=self.conf.validation_split_holdout,
                shuffle=self.conf.shuffle,
                data_norm=data_norm
                )
        else:
            raise NotImplementedError(f"data_src {self.conf.data_src} not implemented")


    def load_checkpoint(self, fname: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int,float]:
        """Load a checkpoint, optionally including the optimizer state

        Args:
            fname (str): File name
            optimizer (Optional[torch.optim.Optimizer], optional): Optimizer to update from checkpoint. Defaults to None.

        Returns:
            Tuple[int,float]: Epoch and loss
        """        
        logger.debug(f"Loading model weights from {fname}")
        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            logger.debug(f"Loading optimizer state from {fname}")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Loaded optimizer state from epoch {epoch} with loss {loss}")
        return epoch, loss


    def predict(self, batch_input: torch.Tensor) -> torch.Tensor:
        """Predict the output for a batch of input data

        Args:
            batch_input (torch.Tensor): Input data of shape (batch_size, input_length (time), no_features)

        Returns:
            torch.Tensor: Predicted output of shape (batch_size, prediction_length (time), no_features)
        """        
        self.model.eval()
        with torch.no_grad():
            batch_pred_hat = self.model(batch_input)
        return batch_pred_hat


    def _load_data_norm(self) -> Optional[DataNormalization]:
        if os.path.exists(self.conf.data_norm_json):
            logger.debug(f"Loading data normalization from {self.conf.data_norm_json}")
            with open(self.conf.data_norm_json, "r") as f:
                return DataNormalization.from_dict(json.load(f))
        else:
            return None


    def _write_data_norm(self, data_norm: DataNormalization):
        with open(self.conf.data_norm_json, "w") as f:
            json.dump(data_norm.to_dict(), f, indent=3)
            logger.debug(f"Saved data normalization to {f.name}")


    def predict_val_dataset(self, max_samples: Optional[int] = None, save_inputs: bool = False) -> List:

        # Change batch size to 1 and not shuffle data for consistency
        batch_size_save = self.conf.batch_size
        shuffle_save = self.conf.shuffle
        self.conf.batch_size = 1
        self.conf.shuffle = False

        # Load the data normalization if it exists and use it
        data_norm = self._load_data_norm()

        # Create the loaders
        _, loader_val, _ = self.create_data_loaders_train_val(data_norm)
        
        # Predict
        data_json = []
        for _ in tqdm(range(max_samples or len(loader_val)), desc="Predicting"):
            batch_input, batch_pred = next(iter(loader_val))
            batch_pred_hat = self.predict(batch_input)
            data_json.append({
                "pred_gt": batch_pred.tolist()[0],
                "pred": batch_pred_hat.tolist()[0]
                })
            if save_inputs:
                data_json[-1]["input"] = batch_input.tolist()[0]

        # Save data to json
        with open(self.conf.pred_val_dataset_json, "w") as f:
            json.dump(data_json, f)
            logger.info(f"Saved data to {f.name}")

        # Reset options
        self.conf.batch_size = batch_size_save
        self.conf.shuffle = shuffle_save

        return data_json


    def load_training_metadata_or_new(self, epoch_start: Optional[int] = None) -> TrainingMetadata:
        """Load the training progress from a JSON file, or create a new one

        Args:
            epoch_start (Optional[int], optional): Starting epoch - earlier epochs will be removed if not None. Defaults to None.

        Returns:
            TrainProgress: Training metadata
        """        
        if os.path.exists(self.conf.train_progress_json):
            with open(self.conf.train_progress_json, "r") as f:
                tp = self.TrainingMetadata.from_dict(json.load(f))

            # Remove epochs after epoch_start
            if epoch_start is not None:
                tp.epoch_to_data = { epoch: tp.epoch_to_data[epoch] for epoch in tp.epoch_to_data if epoch < epoch_start }
            
            return tp
        else:
            return self.TrainingMetadata(epoch_to_data={})


    def train(self):
        """Train the model
        """        

        # Create the optimizer
        optimizer_cls = getattr(torch.optim, self.conf.optimizer)
        optimizer = optimizer_cls(self.model.parameters(), lr=self.conf.learning_rate)

        # Load if needed
        if self.conf.initialize == self.conf.Initialize.FROM_LATEST_CHECKPOINT:
            epoch_start, val_loss_best = self.load_checkpoint(fname=self.conf.checkpoint_latest, optimizer=optimizer)
            data_norm = self._load_data_norm()
        elif self.conf.initialize == self.conf.Initialize.FROM_BEST_CHECKPOINT:
            epoch_start, val_loss_best = self.load_checkpoint(fname=self.conf.checkpoint_best, optimizer=optimizer)
            data_norm = self._load_data_norm()
        elif self.conf.initialize == self.conf.Initialize.FROM_SCRATCH:
            epoch_start, val_loss_best = 0, float("inf")
            # Save initial weights
            self._save_checkpoint(epoch=epoch_start, optimizer=optimizer, loss=val_loss_best, fname=self.conf.checkpoint_init)
            data_norm = None
        else:
            raise NotImplementedError(f"Initialize {self.conf.initialize} not implemented")
        train_data = self.load_training_metadata_or_new(epoch_start)

        # Create the loaders
        loader_train, loader_val, data_norm = self.create_data_loaders_train_val(data_norm)

        # Write data normalization
        self._write_data_norm(data_norm)

        # Train
        for epoch in range(epoch_start, self.conf.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.conf.num_epochs}")
            t0 = time.time()

            # Training
            train_loss = 0
            for batch_input, batch_pred in tqdm(loader_train, desc="Training batches"):
                train_loss += self._train_step(batch_input, batch_pred, optimizer)

            # Validation loss
            self.model.eval()
            val_loss = 0
            for batch_input, batch_pred in tqdm(loader_val, desc="Validation batches"):
                val_loss += self._compute_loss(batch_input, batch_pred).item()

            # Log
            train_loss /= len(loader_train)
            val_loss /= len(loader_val)
            dur = time.time() - t0
            logger.info(f"Training loss: {train_loss:.2f} val: {val_loss:.2f} duration: {dur:.2f}s")

            # Store metadata about training
            train_data.epoch_to_data[epoch] = self.TrainingMetadata.EpochData(epoch=epoch, train_loss=train_loss, val_loss=val_loss, duration_seconds=dur)

            # Save checkpoint
            if val_loss < val_loss_best:
                logger.info(f"New best validation loss: {val_loss:.2f}")
                self._save_checkpoint(epoch=epoch, optimizer=optimizer, loss=val_loss, fname=self.conf.checkpoint_best)
                val_loss_best = val_loss
            self._save_checkpoint(epoch=epoch, optimizer=optimizer, loss=val_loss, fname=self.conf.checkpoint_latest)
            self._write_training_metadata(train_data)


    def _save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, loss: float, fname: str):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, fname)


    def _write_training_metadata(self, train_data: TrainingMetadata):
        """Write the training progress to a JSON file

        Args:
            train_data (TrainingMetadata): _description_
        """        
        if os.path.dirname(self.conf.train_progress_json) != "":
            os.makedirs(os.path.dirname(self.conf.train_progress_json), exist_ok=True)
        with open(self.conf.train_progress_json, "w") as f:
            json.dump(train_data.to_dict(), f, indent=3)


    def _compute_loss(self, batch_input: torch.Tensor, batch_pred: torch.Tensor) -> torch.Tensor:
        # Forward pass
        batch_pred_hat = self.model(batch_input)

        # Compute loss
        loss = torch.nn.functional.mse_loss(batch_pred_hat, batch_pred)
        return loss


    def _train_step(self, batch_input: torch.Tensor, batch_pred: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()

        # Loss
        loss = self._compute_loss(batch_input, batch_pred)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        return loss.item()