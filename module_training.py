"""Module related to training.

We provide two kinds of monitors for monitoring training procedure and
results (CSVLogger & WandbLogger). The main training model setup is 
created with `BinaryLitModel`, specially designed for binary 
classification tasks. The advantage of `L.LightningModule` is that it 
automatically handles routine jobs like moving data to cpu or gpu, loop 
optimizing batch data, or calculating gradients and updating model 
parameters. It can also be combined with `Wandb`, which monitoring
hardware information and training procedure.
"""

import os
import time
from typing import Callable, Union

import lightning as L
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from sklearn import metrics
import torch
import torch.nn as nn
from torch_geometric.data import Data
import wandb


def default_monitor(logger_config: dict, *args: dict):
    """Default logger using CSVLogger.

    Args:
        logger_config : dict
            `kwargs` dictionary for CSVLogger.
        *args : dict
            Other dictionary to be saved (e.g. hyperparameters).

    Returns:
        csv_logger : CSVLogger
            See `lightning.pytorch.loggers.CSVLogger`.
    """

    # It will automatically recursively create the save_dit path.
    save_dir = os.path.join(logger_config["save_dir"], "csv_logger")
    csv_logger = CSVLogger(save_dir=save_dir, name=logger_config["name"])
    csv_config = {}
    csv_config.update(logger_config)
    for config in args:
        csv_config.update(config)
    csv_logger.log_hyperparams(csv_config)
    return csv_logger


def wandb_monitor(model: nn.Module, logger_config: dict, *args: dict):
    """Wandb logger using WandbLogger.

    Args:
        model : nn.Module
            Model to be monitored during training.
        logger_config : dict
            `kwargs` dictionary for WandbLogger.
        *args : dict
            Other dictionary to be saved (e.g. hyperparameters).

    Returns:
        wandb_logger : WandbLogger
            See `lightning.pytorch.loggers.WandbLogger`.
    """

    wandb.login()
    wandb_logger = WandbLogger(**logger_config)
    wandb_config = {}
    wandb_config.update(logger_config)
    for config in args:
        wandb_config.update(config)
    wandb_logger.experiment.config.update(wandb_config, allow_val_change=True)
    wandb_logger.watch(model, log="all")
    return wandb_logger


class BinaryLitModel(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            lr: float,
            graph: bool,
            optimizer: torch.optim.Optimizer = None,
            loss_func: Callable = torch.nn.BCEWithLogitsLoss()
    ):
        """Lightning model for binary classification.

        When training with pytorch lightning module, the model has to be
        created with `L.LightningModule`. One should also specify the 
        optimizer and loss function. The `BinaryLitModel` is specially
        designed for training a binary classification task with graph
        data structure.

        Args:
            model : nn.Module
                PyTorch model based on nn.Module.
            lr : float
                Learning rate.
            graph : bool
                Whether the dataset is graph (in `torch_geometric.data.Data`
                format) or not (in `torch.Tensor` format).
            optimizer : torch.optim.Optimizer (default Adam)
                PyTorch optimizers.
            loss_func : Callable (default BCEWithLogitsLoss)
                PyTorch loss functions.
        """

        super().__init__()
        self.model = model
        self.graph = graph
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr) if optimizer is None else optimizer
        self.loss_func = loss_func

    def forward(self, data: Union[torch.tensor, Data], mode: str):
        """Feed data forward the model.

        Args:
            data : Union([torch.tensor, Data])
                The data type depends on argument `self.graph`.
            mode : str
                Whether the data is training, validation or testing.
                `mode` can be "train", "valid" or "test".
        """

        if self.graph == True:
            # `torch_geometric Data` type.
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.model(x, edge_index, batch)
            y_true = data.y
        else:
            # `torch.tensor` type.
            x, y_true = data
            x = self.model(x)
        x = x.squeeze(dim=-1)

        # Calculate loss and accuracy.
        y_pred = x > 0
        loss = self.loss_func(x, y_true.float())
        acc = (y_pred == y_true).float().mean()

        # Calculate the prediction score.
        y_true = y_true.detach().to("cpu")
        if isinstance(self.loss_func, torch.nn.BCEWithLogitsLoss):
            # Note BCEWithLogitsLoss use sigmoid when calling loss,
            # so the x has not go through a sigmoid or softmax yet.
            y_score = torch.sigmoid(x).detach().to("cpu")
        else:
            # We assume other loss functions have a sigmoid or softmax
            # at the last layer.
            y_score = x.detach().to("cpu")

        # Create buffers for each batch for calculating area under curve
        # (AUC). Note AUC will be calculated in last step (batch).
        if mode == "train":
            self.y_train_true_buffer = torch.cat(
                (self.y_train_true_buffer, y_true))
            self.y_train_score_buffer = torch.cat(
                (self.y_train_score_buffer, y_score))
        elif mode == "valid":
            self.y_valid_true_buffer = torch.cat(
                (self.y_valid_true_buffer, y_true))
            self.y_valid_score_buffer = torch.cat(
                (self.y_valid_score_buffer, y_score))
        elif mode == "test":
            self.y_test_true_buffer = torch.cat(
                (self.y_test_true_buffer, y_true))
            self.y_test_score_buffer = torch.cat(
                (self.y_test_score_buffer, y_score))
        return loss, acc

    def configure_optimizers(self):
        """Function for calling optimizers."""
        return self.optimizer

    # Training functions.
    def on_train_epoch_start(self):
        """Function called at the start of an epoch."""

        # Monitoring time for training an epoch.
        self.start_time = time.time()

        # Create buffer for calculating AUC.
        self.y_train_true_buffer = torch.tensor([])
        self.y_train_score_buffer = torch.tensor([])

    def on_train_epoch_end(self):
        """Function called at the end of an epoch."""

        # Monitoring time for training an epoch.
        self.end_time = time.time()
        delta_time = self.end_time - self.start_time
        self.log("epoch_time", delta_time, on_step=False, on_epoch=True)

        # Calculate AUC.
        roc_auc = metrics.roc_auc_score(
            self.y_train_true_buffer, self.y_train_score_buffer)
        self.log("train_roc_auc", roc_auc, on_step=False, on_epoch=True)

        # Clean up buffer at the end of each epochs.
        del self.y_train_true_buffer
        del self.y_train_score_buffer

    def training_step(self, data, batch_idx):
        """Function called during a step in an epoch."""

        # Calculate the number of data in the batch.
        batch_size = len(data.x) if self.graph is True else len(data[0])
        loss, acc = self.forward(data, mode="train")
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, batch_size=batch_size)
        self.log("train_acc", acc, on_step=True,
                 on_epoch=True, batch_size=batch_size)
        return loss

    # Validation functions (detail see training functions above).
    def on_validation_epoch_start(self):
        self.y_valid_true_buffer = torch.tensor([])
        self.y_valid_score_buffer = torch.tensor([])

    def on_validation_epoch_end(self):
        roc_auc = metrics.roc_auc_score(
            self.y_valid_true_buffer, self.y_valid_score_buffer)
        self.log("val_roc_auc", roc_auc, on_step=False, on_epoch=True)
        del self.y_valid_true_buffer
        del self.y_valid_score_buffer

    def validation_step(self, data, batch_idx):
        batch_size = len(data.x) if self.graph is True else len(data[0])
        _, acc = self.forward(data, mode="valid")
        self.log("valid_acc", acc, on_step=True,
                 on_epoch=True, batch_size=batch_size)

    # Testing functions (detail see training functions above)..
    def on_test_epoch_start(self):
        self.y_test_true_buffer = torch.tensor([])
        self.y_test_score_buffer = torch.tensor([])

    def on_test_epoch_end(self):
        roc_auc = metrics.roc_auc_score(
            self.y_test_true_buffer, self.y_test_score_buffer)
        self.log("test_roc_auc", roc_auc, on_step=False, on_epoch=True)
        del self.y_test_true_buffer
        del self.y_test_score_buffer

    def test_step(self, data, batch_idx):
        batch_size = len(data.x) if self.graph is True else len(data[0])
        _, acc = self.forward(data, mode="test")
        self.log("test_acc", acc, on_step=True,
                 on_epoch=True, batch_size=batch_size)
