# basic tools
import time

# ML tools
import torch
import lightning as L
from sklearn import metrics
from lightning.pytorch.loggers import CSVLogger

# wandb
import wandb
from lightning.pytorch.loggers import WandbLogger

# tensorboard logger
def default_monitor(logger_config, *args):
    csv_logger = CSVLogger(save_dir=logger_config["save_dir"], name=logger_config["name"])
    csv_config = {}
    csv_config.update(logger_config)
    for config in args:
        csv_config.update(config)
    csv_logger.log_hyperparams(csv_config)
    return csv_logger

# wandb monitor
def wandb_monitor(model, logger_config, *args):
    wandb.login()
    wandb_logger = WandbLogger(**logger_config)
    wandb_config = {}
    wandb_config.update(logger_config)
    for config in args:
        wandb_config.update(config)
    wandb_logger.experiment.config.update(wandb_config, allow_val_change=True)
    wandb_logger.watch(model, log="all")
    return wandb_logger

# Binary classification of graph data
class BinaryLitModel(L.LightningModule):
    def __init__(self, model, lr, graph, optimizer=None, loss_func=None):
        super().__init__()
        self.model     = model
        self.graph     = graph
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr) if optimizer is None else optimizer
        self.loss_func = torch.nn.BCEWithLogitsLoss() if loss_func is None else loss_func

    def forward(self, data, mode):
        if self.graph == True:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.model(x, edge_index, batch)
            y_true = data.y
        else:
            x, y_true = data
            x = self.model(x)
        x = x.squeeze(dim=-1)

        # calculate loss and accuracy
        y_pred = x > 0
        loss   = self.loss_func(x, y_true.float())
        acc    = (y_pred == y_true).float().mean()

        # calculate auc
        y_true  = y_true.detach().to("cpu")
        y_score = torch.sigmoid(x).detach().to("cpu") # because we use BCEWithLogitsLoss
        if mode == "train":
            self.y_train_true_buffer  = torch.cat((self.y_train_true_buffer, y_true))
            self.y_train_score_buffer = torch.cat((self.y_train_score_buffer, y_score))
        elif mode == "valid":
            self.y_valid_true_buffer  = torch.cat((self.y_valid_true_buffer, y_true))
            self.y_valid_score_buffer = torch.cat((self.y_valid_score_buffer, y_score))
        elif mode == "test":
            self.y_test_true_buffer  = torch.cat((self.y_test_true_buffer, y_true))
            self.y_test_score_buffer = torch.cat((self.y_test_score_buffer, y_score))
        return loss, acc

    def configure_optimizers(self):
        return self.optimizer

    # training
    def on_train_epoch_start(self):
        self.start_time = time.time()
        self.y_train_true_buffer  = torch.tensor([])
        self.y_train_score_buffer = torch.tensor([])

    def on_train_epoch_end(self):
        # time consumed
        self.end_time = time.time()
        delta_time    = self.end_time - self.start_time
        self.log("epoch_time", delta_time, on_step=False, on_epoch=True)
        
        roc_auc = metrics.roc_auc_score(self.y_train_true_buffer, self.y_train_score_buffer)
        self.log("train_roc_auc", roc_auc, on_step=False, on_epoch=True)
        del self.y_train_true_buffer
        del self.y_train_score_buffer

    def training_step(self, data, batch_idx):
        batch_size = len(data.x) if self.graph is True else len(data[0])
        loss, acc = self.forward(data, mode="train")
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
        return loss
    
    # validation
    def on_validation_epoch_start(self):
        self.y_valid_true_buffer  = torch.tensor([])
        self.y_valid_score_buffer = torch.tensor([])

    def on_validation_epoch_end(self):
        roc_auc = metrics.roc_auc_score(self.y_valid_true_buffer, self.y_valid_score_buffer)
        self.log("val_roc_auc", roc_auc, on_step=False, on_epoch=True)
        del self.y_valid_true_buffer
        del self.y_valid_score_buffer

    def validation_step(self, data, batch_idx):
        batch_size = len(data.x) if self.graph is True else len(data[0])
        _, acc = self.forward(data, mode="valid")
        self.log("valid_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)

    # testing
    def on_test_epoch_start(self):
        self.y_test_true_buffer  = torch.tensor([])
        self.y_test_score_buffer = torch.tensor([])

    def on_test_epoch_end(self):
        roc_auc = metrics.roc_auc_score(self.y_test_true_buffer, self.y_test_score_buffer)
        self.log("test_roc_auc", roc_auc, on_step=False, on_epoch=True)
        del self.y_test_true_buffer
        del self.y_test_score_buffer

    def test_step(self, data, batch_idx):
        batch_size = len(data.x) if self.graph is True else len(data[0])
        _, acc = self.forward(data, mode="test")
        self.log("test_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)