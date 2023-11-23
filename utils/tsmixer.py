from .tsmixer_conf import TSMixerConf, TrainingMetadata, makedirs
from .model import TSMixerModel
from .load_csv import DataNormalization

import os
from typing import Optional, Tuple, Dict, List
import torch
from loguru import logger
from tqdm import tqdm
import json
import time
import shutil
from dataclasses import dataclass
from mashumaro import DataClassDictMixin
import yaml


class TSMixer:
    """TSMixer including training and prediction methods
    """    


    def __init__(self, conf: TSMixerConf):
        """Constructor for TSMixer class

        Args:
            conf (TSMixerConf): Configuration
        """        
        conf.check_valid()
        self.conf = conf

        # Create the model
        self.model = TSMixerModel(
            input_length=self.conf.input_length,
            forecast_length=self.conf.prediction_length,
            no_feats=self.conf.no_features,
            feat_mixing_hidden_channels=self.conf.feat_mixing_hidden_channels or self.conf.no_features,
            no_mixer_layers=self.conf.no_mixer_layers,
            dropout=self.conf.dropout
            )

        # Move to device
        self.model.to(self.conf.device)

        # Load the model
        if self.conf.initialize == self.conf.Initialize.FROM_LATEST_CHECKPOINT:
            self.load_checkpoint(fname=self.conf.checkpoint_latest)
        elif self.conf.initialize == self.conf.Initialize.FROM_BEST_CHECKPOINT:
            self.load_checkpoint(fname=self.conf.checkpoint_best)
        elif self.conf.initialize == self.conf.Initialize.FROM_SCRATCH:
            pass
        else:
            raise NotImplementedError(f"Initialize {self.conf.initialize} not implemented")


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

        # Check size
        assert batch_input.shape[1] == self.conf.input_length, f"Input length {batch_input.shape[1]} does not match configuration {self.conf.input_length}"
        assert batch_input.shape[2] == self.conf.no_features, f"Number of features {batch_input.shape[2]} does not match configuration {self.conf.no_features}"

        # Predict
        batch_input = batch_input.to(self.conf.device)
        with torch.no_grad():
            batch_pred_hat = self.model(batch_input)
        return batch_pred_hat


    def load_data_norm(self) -> Optional[DataNormalization]:
        """Load the data normalization from a JSON file

        Returns:
            Optional[DataNormalization]: Data normalization, or None if the file does not exist
        """        

        if os.path.exists(self.conf.data_norm_json):
            logger.debug(f"Loading data normalization from {self.conf.data_norm_json}")
            with open(self.conf.data_norm_json, "r") as f:
                return DataNormalization.from_dict(json.load(f))
        else:
            return None


    @dataclass
    class PredData(DataClassDictMixin):
        """Prediction data
        """        

        pred_gt: List[List[float]]
        "Ground truth prediction"

        pred: List[List[float]]
        "Model prediction"

        inputs: Optional[List[List[float]]] = None
        "Inputs"


    def predict_val_dataset(self, max_samples: Optional[int] = None, save_inputs: bool = False) -> List[PredData]:
        """Predict on the validation dataset

        Args:
            max_samples (Optional[int], optional): Maximum number of samples to predict from the validation dataset. Defaults to None.
            save_inputs (bool, optional): Save the inputs as well as the predictions. Defaults to False.

        Returns:
            List[PredData]: List of predictions
        """        

        # Change batch size to 1 and not shuffle data for consistency
        batch_size_save = self.conf.batch_size
        shuffle_save = self.conf.shuffle
        self.conf.batch_size = 1
        self.conf.shuffle = False

        # Load the data normalization if it exists and use it
        data_norm = self.load_data_norm()

        # Create the loaders
        _, loader_val, _ = self.conf.create_data_loaders_train_val(data_norm)
        
        # Predict
        data_list: List[TSMixer.PredData] = []
        for _ in tqdm(range(max_samples or len(loader_val)), desc="Predicting"):
            batch_input, batch_pred = next(iter(loader_val))
            batch_pred_hat = self.predict(batch_input)
            data = TSMixer.PredData(
                pred_gt=batch_pred.tolist()[0],
                pred=batch_pred_hat.tolist()[0],
                inputs=batch_input.tolist()[0] if save_inputs else None
                )
            data_list.append(data)            

        # Save data to json
        with open(self.conf.pred_val_dataset_json, "w") as f:
            json.dump([ d.to_dict() for d in data_list ], f)
            logger.info(f"Saved data to {f.name}")

        # Reset options
        self.conf.batch_size = batch_size_save
        self.conf.shuffle = shuffle_save

        return data_list


    def train(self):
        """Train the model
        """        

        # Create the optimizer
        optimizer_cls = getattr(torch.optim, self.conf.optimizer)
        optimizer = optimizer_cls(self.model.parameters(), lr=self.conf.learning_rate)

        # Load if needed
        if self.conf.initialize == self.conf.Initialize.FROM_LATEST_CHECKPOINT:
            epoch_start, val_loss_best = self.load_checkpoint(fname=self.conf.checkpoint_latest, optimizer=optimizer)
            data_norm = self.load_data_norm()
        elif self.conf.initialize == self.conf.Initialize.FROM_BEST_CHECKPOINT:
            epoch_start, val_loss_best = self.load_checkpoint(fname=self.conf.checkpoint_best, optimizer=optimizer)
            data_norm = self.load_data_norm()
        elif self.conf.initialize == self.conf.Initialize.FROM_SCRATCH:
            epoch_start, val_loss_best = 0, float("inf")

            # Clear the output directory
            if os.path.exists(self.conf.output_dir):
                logger.warning(f"Output directory {self.conf.output_dir} already exists. Deleting it to start over. You have 8 seconds.")
                for _ in range(8):
                    print(".", end="", flush=True)
                    time.sleep(1)
                print("")
                shutil.rmtree(self.conf.output_dir)
            makedirs(self.conf.output_dir)

            # Save initial weights
            self._save_checkpoint(epoch=epoch_start, optimizer=optimizer, loss=val_loss_best, fname=self.conf.checkpoint_init)
            data_norm = None

            # Copy the config to the output directory for reference
            fname_conf = os.path.join(self.conf.output_dir, "conf.yml")
            makedirs(self.conf.output_dir)
            with open(fname_conf, "w") as f:
                yaml.dump(self.conf.to_dict(), f, indent=3)
                logger.info(f"Saved configuration to {f.name}")
        
        else:
            raise NotImplementedError(f"Initialize {self.conf.initialize} not implemented")
        train_data = self.conf.load_training_metadata_or_new(epoch_start)

        # Create the loaders
        loader_train, loader_val, data_norm = self.conf.create_data_loaders_train_val(data_norm)

        # Write data normalization
        self.conf.write_data_norm(data_norm)

        # Train
        epoch_last_improvement = None
        for epoch in range(epoch_start, self.conf.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.conf.num_epochs}")
            t0 = time.time()

            # Training
            train_loss = 0
            for batch_input, batch_pred in tqdm(loader_train, desc="Training batches"):
                batch_input, batch_pred = batch_input.to(self.conf.device), batch_pred.to(self.conf.device)
                train_loss += self._train_step(batch_input, batch_pred, optimizer)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_input, batch_pred in tqdm(loader_val, desc="Validation batches"):
                    batch_input, batch_pred = batch_input.to(self.conf.device), batch_pred.to(self.conf.device)
                    val_loss += self._compute_loss(batch_input, batch_pred).item()

            # Log
            train_loss /= len(loader_train)
            val_loss /= len(loader_val)
            dur = time.time() - t0
            logger.info(f"Training loss: {train_loss:.5f} val: {val_loss:.5f} duration: {dur:.2f}s")

            # Store metadata about training
            train_data.epoch_to_data[epoch] = TrainingMetadata.EpochData(epoch=epoch, train_loss=train_loss, val_loss=val_loss, duration_seconds=dur)

            # Save checkpoint
            if val_loss < val_loss_best:
                logger.info(f"New best validation loss: {val_loss:.5f}")
                self._save_checkpoint(epoch=epoch, optimizer=optimizer, loss=val_loss, fname=self.conf.checkpoint_best)
                val_loss_best = val_loss
                epoch_last_improvement = epoch
            self._save_checkpoint(epoch=epoch, optimizer=optimizer, loss=val_loss, fname=self.conf.checkpoint_latest)
            self.conf.write_training_metadata(train_data)

            # Early stopping
            if epoch_last_improvement is not None and self.conf.early_stopping_patience is not None and epoch - epoch_last_improvement >= self.conf.early_stopping_patience:
                logger.info(f"Stopping early after {epoch - epoch_last_improvement} epochs without improvement in validation loss.")
                break


    def _save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, loss: float, fname: str):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, fname)


    def _compute_loss(self, batch_input: torch.Tensor, batch_pred: torch.Tensor) -> torch.Tensor:
        """Compute the loss

        Args:
            batch_input (torch.Tensor): Batch input of shape (batch_size, input_length (time), no_features)
            batch_pred (torch.Tensor): Batch prediction of shape (batch_size, prediction_length (time), no_features)

        Returns:
            torch.Tensor: Loss (MSE)
        """        

        # Forward pass
        batch_pred_hat = self.model(batch_input)

        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(batch_pred_hat, batch_pred)

        # Normalize the loss by the batch size
        # batch_size = batch_input.size(0)
        # loss /= batch_size

        return loss


    def _train_step(self, batch_input: torch.Tensor, batch_pred: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Training step

        Args:
            batch_input (torch.Tensor): Input data of shape (batch_size, input_length (time), no_features)
            batch_pred (torch.Tensor): Prediction data of shape (batch_size, prediction_length (time), no_features)
            optimizer (torch.optim.Optimizer): Optimizer

        Returns:
            float: Loss (MSE)
        """        
        optimizer.zero_grad()

        # Train mode
        self.model.train()

        # Loss
        loss = self._compute_loss(batch_input, batch_pred)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        return loss.item()