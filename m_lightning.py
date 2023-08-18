import time
import torch
import lightning as L
import lightning.pytorch as pl
from sklearn import metrics

# Binary classification of graph data
class BinaryLitModel(L.LightningModule):
    def __init__(self, model, lr=1E-3, optimizer=None, loss_func=None, graph=False):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])
        self.model     = model
        self.graph     = graph # whether the data input is a graph (using torch_geometric)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr) if optimizer is None else optimizer
        self.loss_func = torch.nn.BCEWithLogitsLoss() if loss_func is None else loss_func

    def forward(self, data):
        # predict y
        if self.graph == True:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.model(x, edge_index, batch)
        else:
            x = self.model(data)
        x = x.squeeze(dim=-1)

        # calculate loss and accuracy
        y_pred = x > 0
        y_true = data.y
        loss   = self.loss_func(x, y_true.float())
        acc    = (y_pred == data.y).float().mean()

        # calculate auc
        y_score = torch.sigmoid(x).detach() # because we use BCEWithLogitsLoss
        self.y_true_buffer  = torch.cat((self.y_true_buffer, y_true))
        self.y_score_buffer = torch.cat((self.y_score_buffer, y_score))
        return loss, acc

    def configure_optimizers(self):
        return self.optimizer

    def on_train_epoch_start(self):
        self.start_time     = time.time()
        self.y_true_buffer  = torch.tensor([])
        self.y_score_buffer = torch.tensor([])

    def on_train_epoch_end(self):
        self.end_time = time.time()
        delta_time    = self.end_time - self.start_time
        roc_auc       = metrics.roc_auc_score(self.y_true_buffer, self.y_score_buffer)
        self.log("epoch_time", delta_time, on_step=False, on_epoch=True)
        self.log("train_roc_auc", roc_auc, on_step=False, on_epoch=True)

    def training_step(self, data, batch_idx):
        loss, acc = self.forward(data)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=len(data.x))
        self.log("train_acc", acc, on_step=True, on_epoch=True, batch_size=len(data.x))
        return loss
    
    def on_validation_epoch_start(self):
        self.y_true_buffer  = torch.tensor([])
        self.y_score_buffer = torch.tensor([])

    def on_validation_epoch_end(self):
        roc_auc = metrics.roc_auc_score(self.y_true_buffer, self.y_score_buffer)
        self.log("val_roc_auc", roc_auc, on_step=False, on_epoch=True)

    def validation_step(self, data, batch_idx):
        _, acc = self.forward(data)
        self.log("valid_acc", acc, on_step=True, on_epoch=True, batch_size=len(data.x))

    def on_test_epoch_start(self):
        self.y_true_buffer  = torch.tensor([])
        self.y_score_buffer = torch.tensor([])

    def on_test_epoch_end(self):
        roc_auc = metrics.roc_auc_score(self.y_true_buffer, self.y_score_buffer)
        self.log("test_roc_auc", roc_auc, on_step=False, on_epoch=True)

    def test_step(self, data, batch_idx):
        _, acc = self.forward(data)
        self.log("test_acc", acc, on_step=True, on_epoch=True, batch_size=len(data.x))