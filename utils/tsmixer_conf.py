from .load_csv import DataNormalization

from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from enum import Enum
import os
from typing import Optional, Tuple, Dict, List
from torch.utils.data import DataLoader
from loguru import logger
import json


def makedirs(d: str):
    if d != "":
        os.makedirs(d, exist_ok=True)


@dataclass
class TSMixerConf(DataClassDictMixin):

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

    device: str = "mps"
    "Device to use for training"

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

    dropout: float = 0.5
    "Dropout"

    feat_mixing_hidden_channels: Optional[int] = None
    "Number of hidden channels in the feature mixing MLP. If None, uses same as input features."

    early_stopping_patience: Optional[int] = 5
    "Early stopping patience. If the validation loss does not improve over this many epochs, stop early. If None, no early stopping is used."

    @property
    def image_dir(self):
        makedirs(self.output_dir)
        makedirs(os.path.join(self.output_dir, "images"))
        return os.path.join(self.output_dir, "images")

    @property
    def checkpoint_init(self):
        makedirs(self.output_dir)
        return os.path.join(self.output_dir, "init.pth")

    @property
    def checkpoint_best(self):
        makedirs(self.output_dir)
        return os.path.join(self.output_dir, "best.pth")

    @property
    def checkpoint_latest(self):
        makedirs(self.output_dir)
        return os.path.join(self.output_dir, "latest.pth")

    @property
    def train_progress_json(self):
        makedirs(self.output_dir)
        return os.path.join(self.output_dir, "loss.json")

    @property
    def pred_val_dataset_json(self):
        makedirs(self.output_dir)
        return os.path.join(self.output_dir, "pred_val_dataset.json")

    @property
    def data_norm_json(self):
        makedirs(self.output_dir)
        return os.path.join(self.output_dir, "data_norm.json")

    def check_valid(self):
        assert 0 <= self.validation_split_holdout <= 1, "validation_split_holdout must be between 0 and 1"

        # Check device exists
        import torch
        assert self.device in ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "mps"], f"Device {self.device} not supported"
        if self.device == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available"
            assert torch.cuda.device_count() > 1, "Must have more than one CUDA device to use MPS"
        elif self.device == "mps":
            assert torch.backends.mps.is_available(), "MPS is not available"
    

    def load_training_metadata_or_new(self, epoch_start: Optional[int] = None) -> "TrainingMetadata":
        """Load the training progress from a JSON file, or create a new one

        Args:
            epoch_start (Optional[int], optional): Starting epoch - earlier epochs will be removed if not None. Defaults to None.

        Returns:
            TrainProgress: Training metadata
        """        
        if os.path.exists(self.train_progress_json):
            with open(self.train_progress_json, "r") as f:
                tp = TrainingMetadata.from_dict(json.load(f))

            # Remove epochs after epoch_start
            if epoch_start is not None:
                tp.epoch_to_data = { epoch: tp.epoch_to_data[epoch] for epoch in tp.epoch_to_data if epoch < epoch_start }
            
            return tp
        else:
            return TrainingMetadata(epoch_to_data={})


    def write_data_norm(self, data_norm: DataNormalization):
        """Write the data normalization to a JSON file

        Args:
            data_norm (DataNormalization): Data normalization
        """        
        with open(self.data_norm_json, "w") as f:
            json.dump(data_norm.to_dict(), f, indent=3)
            logger.debug(f"Saved data normalization to {f.name}")


    def write_training_metadata(self, train_data: "TrainingMetadata"):
        """Write the training progress to a JSON file

        Args:
            train_data (TrainingMetadata): _description_
        """        
        if os.path.dirname(self.train_progress_json) != "":
            makedirs(os.path.dirname(self.train_progress_json))
        with open(self.train_progress_json, "w") as f:
            json.dump(train_data.to_dict(), f, indent=3)


    def create_data_loaders_train_val(self, data_norm: Optional[DataNormalization] = None) -> Tuple[DataLoader, DataLoader, DataNormalization]:
        """Create the training and validation data loaders

        Args:
            data_norm (Optional[DataNormalization], optional): Data normalization to use, otherwise will be calculated. Defaults to None.

        Returns:
            Tuple[DataLoader, DataLoader, DataNormalization]: Training and validation data loaders
        """        

        if self.data_src == self.DataSrc.CSV_FILE:
            assert self.data_src_csv is not None, "data_src_csv must be set if data_src is CSV_FILE"

            from .load_csv import load_csv_dataset, ValidationSplit
            return load_csv_dataset(
                csv_file=self.data_src_csv,
                batch_size=self.batch_size,
                input_length=self.input_length,
                prediction_length=self.prediction_length,
                val_split=ValidationSplit(self.validation_split.value),
                val_split_holdout=self.validation_split_holdout,
                shuffle=self.shuffle,
                data_norm_exist=data_norm
                )
        else:
            raise NotImplementedError(f"data_src {self.data_src} not implemented")


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
