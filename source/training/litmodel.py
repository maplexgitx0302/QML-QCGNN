"""Lightning Module for ML models."""

import time

import lightning as L
from sklearn import metrics
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Order of lightning hooks https://pytorch-lightning.readthedocs.io/en/1.7.2/common/lightning_module.html#hooks

class TorchLightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, score_dim: int, print_log: bool = True):
        """Lightning Module for PyTorch framework."""

        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.score_dim = score_dim
        self.print_log = print_log

        if score_dim == 1:
            # For output dimension == 1
            self.loss_function = nn.BCEWithLogitsLoss()
            self.score_function = nn.functional.sigmoid
        else:
            # For output dimension >= 2
            self.loss_function = nn.CrossEntropyLoss()
            self.score_function = nn.functional.softmax

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

    def forward(self, batch: tuple[torch.Tensor, torch.Tensor], mode: str) -> torch.Tensor:
        """Return loss and save tensor buffers."""

        x, y_true = batch
        y = self.model(x)
        
        assert torch.isfinite(y).all(), "Model output contains NaN or Inf."

        if self.score_dim == 1:
            # Sigmoid.
            y = y.view(-1)
            y_score = self.score_function(y)
        else:
            # Softmax.
            y_score = self.score_function(y, dim=-1)

        self.y_true_buffer[mode] = torch.cat((self.y_true_buffer[mode], y_true.detach().cpu()))
        self.y_score_buffer[mode] = torch.cat((self.y_score_buffer[mode], y_score.detach().cpu()))
        
        if mode != 'test':
            if self.score_dim == 1:
                # BCEWithLogitsLoss needs to cast to float for y_true.
                loss = self.loss_function(y, y_true.float())
            else:
                # CrossEntropyLoss cannot use float for y_true.
                loss = self.loss_function(y, y_true)
            
            assert torch.isfinite(loss).all(), "Loss contains NaN or Inf."
            
            return loss

    def configure_optimizers(self):
        return self.optimizer

    def calculate_metrics(self, mode: str):
        
        y_true = self.y_true_buffer[mode]
        y_score = self.y_score_buffer[mode]


        if self.score_dim == 1:
            y_pred = (y_score > 0.5)

        elif self.score_dim == 2:
            # Defect of sklearn roc_auc_score.
            y_score = y_score[:, 1]
            y_pred = (y_score > 0.5)
            
        else:
            y_pred = torch.argmax(y_score, dim=-1)

        auc = metrics.roc_auc_score(y_true=y_true.int(), y_score=y_score, average='macro', multi_class='ovo')
        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)

        if self.score_dim >= 3:
            ovr_accuracies = [torch.mean(((y_true == i) == (y_pred == i)).float()) for i in range(self.score_dim)]
            accuracy = (accuracy, ovr_accuracies)
        
        return auc, accuracy
    
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
        # `training_step` logs will depend on `log_every_n_step`, see https://github.com/Lightning-AI/pytorch-lightning/issues/4479
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
        auc, accuracy = self.calculate_metrics(mode='train')

        # If score_dim >= 3, accuracy is tuple of (accuracy, ovr_accuracies).
        if self.score_dim >= 3:
            accuracy, ovr_accuracies = accuracy
            for i, ovr_accuracy in enumerate(ovr_accuracies):
                self.log(f'train_ovr_accuracy_{i}', ovr_accuracy, on_epoch=True)

        self.log('epoch_time', time_cost, on_epoch=True)
        self.log('train_auc', auc, on_epoch=True)
        self.log('train_accuracy', accuracy, on_epoch=True)


        if self.print_log:
            print(f"\n - (Train) Epoch {self.current_epoch}: Accuracy = {accuracy} | AUC = {auc}\n")

    def on_validation_epoch_end(self):

        time_cost = time.time() - self.valid_start_time
        auc, accuracy = self.calculate_metrics(mode='valid')

        # If score_dim >= 3, accuracy is tuple of (accuracy, ovr_accuracies).
        if self.score_dim >= 3:
            accuracy, ovr_accuracies = accuracy
            for i, ovr_accuracy in enumerate(ovr_accuracies):
                self.log(f'valid_ovr_accuracy_{i}', ovr_accuracy, on_epoch=True)

        self.log('valid_time', time_cost, on_epoch=True)
        self.log('valid_accuracy', accuracy, on_epoch=True)
        self.log('valid_auc', auc, on_epoch=True)

        if self.print_log:
            print(f"\n - (Valid) Epoch {self.current_epoch}: Accuracy = {accuracy} | AUC = {auc}\n")
    
    def on_test_epoch_end(self):

        auc, accuracy = self.calculate_metrics(mode='test')

        if self.print_log:
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