from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from enum import Enum
import os


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

        data_src: DataSrc = DataSrc.CSV_FILE
        "Where to load the dataset from"

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
        "Use the last X% of the data for validation"

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

        def check_valid(self):
            assert 0 <= self.validation_split_holdout <= 1, "validation_split_holdout must be between 0 and 1"
