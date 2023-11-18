import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from enum import Enum
import torch
from typing import Tuple


class ValidationSplit(Enum):
    
    TEMPORAL_HOLDOUT = "temporal-holdout"
    "Reserve the last portion (e.g., 10-20%) of your time-ordered data for validation, and use the remaining data for training. This is a simple and widely used approach."


class PdDataset(Dataset):

    def __init__(self, df: pd.DataFrame, window_size_input: int, window_size_predict: int):
        window_size_total = window_size_input + window_size_predict
        assert len(df) > window_size_total, f"Dataset length ({len(df)}) must be greater than window size ({window_size_total})"
        self.df = df
        self.window_size_input = window_size_input
        self.window_size_predict = window_size_predict

    def __len__(self):
        return len(self.df) - self.window_size_input - self.window_size_predict

    def get_sample(self, idx):
        # Check if the index plus window size exceeds the length of the dataset
        if idx + self.window_size_input + self.window_size_predict > len(self.df):
            raise IndexError(f"Index ({idx}) + window_size_input ({self.window_size_input}) + window_size_predict ({self.window_size_predict}) exceeds dataset length ({len(self.df)})")

        # Window the data
        sample_input = self.df.iloc[idx:idx + self.window_size_input, :]
        sample_pred = self.df.iloc[idx + self.window_size_input:idx + self.window_size_input + self.window_size_predict, :]

        # Convert to torch tensor
        sample_input = torch.tensor(sample_input.values, dtype=torch.float32)
        sample_pred = torch.tensor(sample_pred.values, dtype=torch.float32)

        return sample_input, sample_pred

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            # Handle a list of indices
            samples = [self.get_sample(i) for i in idx]
            return samples
        else:
            # Handle a single index
            return self.get_sample(idx)


def load_etdataset(csv_file: str, batch_size: int, input_length: int, prediction_length: int, val_split: ValidationSplit, val_split_holdout: float = 0.2, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, parse_dates=['date'])

    # Remove the date column, if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # Make dataset
    dataset = PdDataset(df, window_size_input=input_length, window_size_predict=prediction_length)
    no_pts = len(dataset)

    # Split the data into training and validation
    if val_split == ValidationSplit.TEMPORAL_HOLDOUT:
        idxs_train = list(range(int(no_pts * (1-val_split_holdout))))
        idxs_val = list(range(int(no_pts * (1-val_split_holdout)), no_pts))
        train_dataset = Subset(dataset, idxs_train)
        val_dataset = Subset(dataset, idxs_val)
    else:
        raise NotImplementedError(f"Validation split {val_split} not implemented")

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return loader_train, loader_val
