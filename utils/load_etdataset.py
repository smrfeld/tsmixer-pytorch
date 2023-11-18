import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from enum import Enum
import torch
from typing import Tuple


class ValidationSplit(Enum):
    
    TEMPORAL_HOLDOUT = "temporal-holdout"
    "Reserve the last portion (e.g., 10-20%) of your time-ordered data for validation, and use the remaining data for training. This is a simple and widely used approach."


class PdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int, transform=None):
        self.df = df
        self.transform = transform
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def get_sample(self, idx):
        # Check if the index plus window size exceeds the length of the dataset
        if idx + self.window_size > len(self.df):
            raise IndexError("Index + window_size exceeds dataset length")

        # Window the data
        sample = self.df.iloc[idx:idx + self.window_size, :]

        # Convert to torch tensor
        sample = torch.tensor(sample.values, dtype=torch.float32)

        # Transform as needed
        if self.transform:
            sample = self.transform(sample)

        return sample

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


def load_etdataset(csv_file: str, batch_size: int, input_length: int, val_split: ValidationSplit, val_split_holdout: float = 0.2) -> Tuple[DataLoader, DataLoader]:

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, parse_dates=['date'])
    no_pts = len(df)

    # Make dataset
    dataset = PdDataset(df, input_length)

    # Split the data into training and validation
    if val_split == ValidationSplit.TEMPORAL_HOLDOUT:
        idxs_train = list(range(int(no_pts * (1-val_split_holdout))))
        idxs_val = list(range(int(no_pts * (1-val_split_holdout)), no_pts))
        train_dataset = Subset(dataset, idxs_train)
        val_dataset = Subset(dataset, idxs_val)
    else:
        raise NotImplementedError(f"Validation split {val_split} not implemented")

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return loader_train, loader_val
