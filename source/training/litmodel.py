"""Lightning Module for ML models."""

import time
from typing import Callable, Union

import lightning as L
from sklearn import metrics
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Order of lightning hooks https://pytorch-lightning.readthedocs.io/en/1.7.2/common/lightning_module.html#hooks

class TorchLightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, score_dim: int):
        """Lightning Module for PyTorch framework."""

        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.score_dim = score_dim

        if score_dim >= 2:
            # For output dimension >= 2
            self.loss_function = nn.CrossEntropyLoss()
            self.score_function = nn.functional.softmax
        else:
            # For output dimension == 1
            self.loss_function = nn.BCEWithLogitsLoss()
            self.score_function = nn.functional.sigmoid

        # Buffers for saving tensor result to calculate AUC.
        self.y_true_buffer = {
            'train': torch.tensor([]),
            'valid': torch.tensor([]),
            'test': torch.tensor([]),
        }

        self.y_score_buffer = {
            'train': torch.tensor([]),
            'valid': torch.tensor([]),
            'test': torch.tensor([]),
        }

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor], mode: str) -> float:
        """Return loss and save tensor buffers."""

        x, y_true = batch
        y = self.model(x)
        y_score = self.score_function(y, dim=-1)

        self.y_true_buffer[mode] = torch.cat((self.y_true_buffer[mode], y_true.detach().cpu()))
        self.y_score_buffer[mode] = torch.cat((self.y_score_buffer[mode], y_score.detach().cpu()))
        
        if mode != 'test':
            loss = self.loss_function(y, y_true)
            return loss

    def configure_optimizers(self):
        return self.optimizer

    def calculate_metrics(self, mode: str):
        
        y_true = self.y_true_buffer[mode]
        y_score = self.y_score_buffer[mode]


        # Defect of sklearn roc_auc_score.
        if self.score_dim == 2:
            y_score = y_score[:, 1]

        if self.score_dim <= 2:
            y_pred = (y_score > 0.5)
        else:
            y_pred = torch.argmax(y_score, dim=-1)

        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        auc_score = metrics.roc_auc_score(y_true=y_true.int(), y_score=y_score, average='macro', multi_class='ovo')
        
        return accuracy, auc_score
    
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
        self.y_true_buffer['train'] = torch.tensor([])
        self.y_score_buffer['train'] = torch.tensor([])

    def on_validation_epoch_start(self):
        self.valid_start_time = time.time()
        self.y_true_buffer['valid'] = torch.tensor([])
        self.y_score_buffer['valid'] = torch.tensor([])

    def on_test_epoch_start(self):
        self.y_true_buffer['test'] = torch.tensor([])
        self.y_score_buffer['test'] = torch.tensor([])

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self.forward(batch, mode='train')
        self.log('train_loss', loss, on_step=True)
        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self.forward(batch, mode='valid')
        self.log('valid_loss', loss, on_step=True)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        self.forward(batch, mode='test')

    def on_train_epoch_end(self):

        time_cost = time.time() - self.epoch_start_time
        accuracy, auc = self.calculate_metrics(mode='train')

        self.log('epoch_time', time_cost, on_epoch=True)
        self.log('train_accuracy', accuracy, on_epoch=True)
        self.log('train_auc', auc, on_epoch=True)

        print(f"\n - (Train) Epoch {self.current_epoch}: Accuracy = {accuracy} | AUC = {auc}\n")

    def on_validation_epoch_end(self):

        time_cost = time.time() - self.valid_start_time
        accuracy, auc = self.calculate_metrics(mode='valid')

        self.log('valid_time', time_cost, on_epoch=True)
        self.log('valid_accuracy', accuracy, on_epoch=True)
        self.log('valid_auc', auc, on_epoch=True)

        print(f"\n - (Valid) Epoch {self.current_epoch}: Accuracy = {accuracy} | AUC = {auc}\n")
    
    def on_test_epoch_end(self):

        accuracy, auc = self.calculate_metrics(mode='test')

        print(f"\n - (Test) Final metrics: Accuracy = {accuracy} | AUC = {auc}\n")



class GraphLightningModel(TorchLightningModule):
    """Torch geometric version for graph."""
    
    def forward(self, data: Data, mode: str) -> float:
        """Return loss and save tensor buffers."""

        y = self.model(data.x, data.edge_index, data.batch)
        y_true = data.y
        y_score = self.score_function(y, dim=-1)

        self.y_true_buffer[mode] = torch.cat((self.y_true_buffer[mode], y_true.detach().cpu()))
        self.y_score_buffer[mode] = torch.cat((self.y_score_buffer[mode], y_score.detach().cpu()))
        
        if mode != 'test':
            loss = self.loss_function(y, y_true)
            return loss

    def training_step(self, data: Data, batch_idx: int):
        batch_size = len(data.x)
        loss = self.forward(data, mode='train')
        self.log('train_loss', loss, on_step=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, data: Data, batch_idx: int):
        batch_size = len(data.x)
        loss = self.forward(data, mode='valid')
        self.log('valid_loss', loss, on_step=True, batch_size=batch_size)
    
    def test_step(self, data: Data, batch_idx: int):
        batch_size = len(data.x)
        self.forward(data, mode='test')