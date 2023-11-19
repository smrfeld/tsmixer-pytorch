import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
from enum import Enum
import torch
from typing import Tuple, Callable, Optional, List
from dataclasses import dataclass
from mashumaro import DataClassDictMixin
from loguru import logger


class ValidationSplit(Enum):
    
    TEMPORAL_HOLDOUT = "temporal-holdout"
    "Reserve the last portion (e.g., 10-20%) of your time-ordered data for validation, and use the remaining data for training. This is a simple and widely used approach."


class DataframeDataset(Dataset):
    """Dataset from a pandas dataframe
    """    

    def __init__(self, df: pd.DataFrame, window_size_input: int, window_size_predict: int, transform: Optional[Callable] = None):
        """Constructor

        Args:
            df (pd.DataFrame): Dataframe
            window_size_input (int): Input window size
            window_size_predict (int): Prediction window size
            transform (Optional[Callable], optional): Transforms such as normalization applied to time series. Defaults to None.
        """        
        window_size_total = window_size_input + window_size_predict
        assert len(df) > window_size_total, f"Dataset length ({len(df)}) must be greater than window size ({window_size_total})"
        self.df = df
        self.window_size_input = window_size_input
        self.window_size_predict = window_size_predict
        self.transform = transform

    def __len__(self):
        return len(self.df) - self.window_size_input - self.window_size_predict

    def get_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a window sample. Input from [idx, idx + window_size_input], prediction from [idx + window_size_input, idx + window_size_input + window_size_predict

        Args:
            idx (int): Index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and prediction tensors
        """

        # Check if the index plus window size exceeds the length of the dataset
        if idx + self.window_size_input + self.window_size_predict > len(self.df):
            raise IndexError(f"Index ({idx}) + window_size_input ({self.window_size_input}) + window_size_predict ({self.window_size_predict}) exceeds dataset length ({len(self.df)})")

        # Window the data
        sample_input = self.df.iloc[idx:idx + self.window_size_input, :]
        sample_pred = self.df.iloc[idx + self.window_size_input:idx + self.window_size_input + self.window_size_predict, :]

        # Convert to torch tensor
        sample_input = torch.tensor(sample_input.values, dtype=torch.float32)
        sample_pred = torch.tensor(sample_pred.values, dtype=torch.float32)

        # Apply transform
        if self.transform is not None:
            sample_input = self.transform(sample_input)
            sample_pred = self.transform(sample_pred)

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


@dataclass
class DataNormalization(DataClassDictMixin):
    mean_each_feature: Optional[List[float]] = None
    "Mean for each feature"

    std_each_feature: Optional[List[float]] = None
    "Std for each feature"


def load_csv_dataset(
    csv_file: str, 
    batch_size: int, 
    input_length: int, 
    prediction_length: int, 
    val_split: ValidationSplit, 
    val_split_holdout: float = 0.2, 
    shuffle: bool = True,
    normalize_each_feature: bool = True,
    data_norm_exist: Optional[DataNormalization] = None
    ) -> Tuple[DataLoader, DataLoader, DataNormalization]:
    """Load a CSV dataset

    Args:
        csv_file (str): CSV file path
        batch_size (int): Batch size
        input_length (int): Input length
        prediction_length (int): Prediction length
        val_split (ValidationSplit): Validation split method
        val_split_holdout (float, optional): Holdout fraction for validation (last X% of data) - only used for TEMPORAL_HOLDOUT. Defaults to 0.2.
        shuffle (bool, optional): True to shuffle data. Defaults to True.
        normalize_each_feature (bool, optional): Normalize each feature. Defaults to True.
        data_norm_exist (Optional[DataNormalization], optional): Existing normalization data - apply this instead of recalculating. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader, DataNormalization]: Training and validation data loaders, and normalization data
    """    

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file, parse_dates=['date'])        

    # Remove the date column, if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # Make dataset
    dataset = DataframeDataset(df, window_size_input=input_length, window_size_predict=prediction_length)
    no_pts = len(dataset)

    # Split the data into training and validation
    if val_split == ValidationSplit.TEMPORAL_HOLDOUT:
        idxs_train = list(range(int(no_pts * (1-val_split_holdout))))
        idxs_val = list(range(int(no_pts * (1-val_split_holdout)), no_pts))
    else:
        raise NotImplementedError(f"Validation split {val_split} not implemented")

    # Normalize each feature separately        
    if data_norm_exist is None:
        data_norm_exist = DataNormalization()

        # Compute mean and std on training data from pandas dataframe
        filtered_df = df.loc[idxs_train]
        data_norm_exist.mean_each_feature = list(filtered_df.mean().values)
        data_norm_exist.std_each_feature = list(filtered_df.std().values)
        logger.debug(f"Computed data mean for each feature: {data_norm_exist.mean_each_feature}")
        logger.debug(f"Computed data std for each feature: {data_norm_exist.std_each_feature}")

    if normalize_each_feature:
        assert data_norm_exist.mean_each_feature is not None, "Must provide data mean for each feature"
        assert data_norm_exist.std_each_feature is not None, "Must provide data std for each feature"

        # Create a normalization function
        transform = lambda x: (x - torch.Tensor(data_norm_exist.mean_each_feature)) / torch.Tensor(data_norm_exist.std_each_feature)

        # Apply the normalization function
        dataset.transform = transform

    # Splits
    train_dataset = Subset(dataset, idxs_train)
    val_dataset = Subset(dataset, idxs_val)

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return loader_train, loader_val, data_norm_exist
