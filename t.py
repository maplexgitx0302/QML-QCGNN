# %%
"""
### Packages
"""

# %%
# basic
import os, time, itertools, subprocess

# qml
import pennylane as qml
from pennylane import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# pytorch_lightning
import lightning as L
import lightning.pytorch as pl

# wandb
import wandb
from lightning.pytorch.loggers import WandbLogger
wandb.login()

# reproducibility
L.seed_everything(3020616)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# faster calculation on GPU but less precision
torch.set_float32_matmul_precision("medium")

# current time
global_time = input("Input time if needed:")
if global_time == "":
    global_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# get gpu name
# if torch.cuda.is_available():
#     gpu_command = "nvidia-smi --query-gpu=name --format=csv,noheader".split()
#     gpu_name = subprocess.run(gpu_command, capture_output=True, text=True)
#     gpu_name = gpu_name.stdout
#     gpu_name = gpu_name.strip("\n")
#     gpu_name = gpu_name.replace(" ", "_")
gpu_name = "NVIDIA_RTX_4090"

# %%
"""
### QML NN Model
"""

# %%
class MLP(nn.Module):
    def __init__(self, q_device, q_diff, q_interface, num_qubits, num_reupload, num_qlayers, num_clayers, num_chidden):
        super().__init__()
    
        # create a quantum MLP
        @qml.qnode(qml.device(q_device, wires=num_qubits), diff_method=q_diff, interface=q_interface)
        def circuit(inputs, weights):
            for i in range(num_reupload):
                qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation='Y')
                qml.StronglyEntanglingLayers(weights=weights[i], wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
        
        # turn the quantum circuit into a torch layer
        weight_shapes = {"weights":(num_reupload, num_qlayers, num_qubits, 3)}
        q_net = [qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)]

        # classical mlp
        c_in, c_hidden, c_out = num_qubits, num_chidden, 1
        c_net = [nn.Linear(c_in, c_hidden), nn.ReLU()]
        for _ in range(num_clayers-2):
            c_net += [nn.Linear(c_hidden, c_hidden), nn.ReLU()]
        c_net += [nn.Linear(c_hidden, c_out)]

        # combine classical and quantum net
        self.net = nn.Sequential(*(q_net + c_net))

    def forward(self, x):
        return self.net(x)

# %%
"""
### Lit Model
"""

# %%
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, data):
        # predict y
        x, y_true = data
        x = self.model(x)
        x = x.squeeze(dim=-1)

        # calculate loss and accuracy
        y_pred = x > 0
        loss   = self.loss_function(x, y_true.float())
        acc    = (y_pred == y_true).float().mean()
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1E-3)
        return optimizer
    
    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        self.end_time = time.time()
        delta_time = self.end_time - self.start_time
        self.log("epoch_time", delta_time, on_step=False, on_epoch=True)

    def training_step(self, data, batch_idx):
        loss, acc = self.forward(data)
        # self.log("train_loss", loss, on_step=True, on_epoch=True)
        # self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

# %%
"""
### Dataset
"""

# %%
class RandomDataset(Dataset):
    def __init__(self, num_data, num_dim):
        super().__init__()
        self.x = torch.rand(num_data, num_dim)
        self.y = torch.cat((torch.ones(num_data//2), torch.zeros(num_data//2)))
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class RandomDataModule(pl.LightningDataModule):
    def __init__(self, num_data, num_dim, batch_size, num_workers):
        super().__init__()
        self.train_dataset  = RandomDataset(num_data, num_dim)
        self.batch_size     = batch_size
        self.num_workers    = num_workers
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

# %%
"""
### Training Procedure
"""

# %%
def test_time(wb, c_device, model_config, data_config):
    model       = MLP(**model_config)
    litmodel    = LitModel(model)
    data_module = RandomDataModule(**data_config)

    mcf = model_config
    dcf = data_config
    if wb == True:
        group    = f"{global_time}_{gpu_name}"
        job_type = f"{c_device}|{mcf['q_device']}|diff_{mcf['q_diff']}|interface_{mcf['q_interface']}"
        name     = f"{job_type}|batch{dcf['batch_size']}|worker{dcf['num_workers']}|dim{dcf['num_dim']}"
        id       = f"{group}|{name}"

        # wandb logger
        wandb_logger = WandbLogger(project="t_qml_time", group=group, job_type=job_type, name=name, id=id, save_dir=f"./result")
        wandb_logger.experiment.config.update(mcf)
        wandb_logger.experiment.config.update(dcf)
        wandb_logger.watch(model, log="all")
        logger = wandb_logger
    else:
        logger = None

    trainer  = L.Trainer(
        logger      = logger, 
        accelerator = c_device, 
        max_epochs  = 3,
        log_every_n_steps = 1,
        )

    print(f"Start testing {c_device}|{mcf['q_device']}|diff_{mcf['q_diff']}|interface_{mcf['q_interface']}|batch{dcf['batch_size']}|worker{dcf['num_workers']}|dim{dcf['num_dim']}")
    trainer.fit(litmodel, datamodule=data_module)

    if wb:
        wandb.finish()
        print("#" * 100)

# %%
real_test = True

if real_test:
    wb = True
    l_num_dim     = [20]
    l_cpu_batch   = [4, 8]
    l_gpu_batch   = [4, 8]
    l_num_workers = [0]
    l_cpu_product = list(itertools.product(["cpu"], l_num_dim, l_cpu_batch, l_num_workers))
    l_gpu_product = list(itertools.product(["gpu"], l_num_dim, l_gpu_batch, l_num_workers))
    l_product     = l_cpu_product + l_gpu_product

else:
    wb = False
    l_num_dim     = [20]
    l_cpu_batch   = [4, 8]
    l_gpu_batch   = [4, 8]
    l_num_workers = [0]
    l_cpu_product = list(itertools.product(["cpu"], l_num_dim, l_cpu_batch, l_num_workers))
    l_gpu_product = list(itertools.product(["gpu"], l_num_dim, l_gpu_batch, l_num_workers))
    l_product     = l_gpu_product

q_tuple = [
    # (q_device       , q_diff    , q_interface)
    ("default.qubit"  , "best"    , "auto"),
    ("default.qubit"  , "best"    , "torch"),
    ("lightning.qubit", "adjoint" , "auto"),
    ("lightning.qubit", "adjoint" , "torch"),
    ("lightning.gpu"  , "adjoint" , "auto"),
    ("lightning.gpu"  , "adjoint" , "torch"),
]

for c_device, num_dim, batch_size, num_workers in l_product:
    for q_device, q_diff, q_interface in q_tuple:
        model_config = {
            "q_device"     : q_device,
            "q_diff"       : q_diff,
            "q_interface"  : q_interface,
            "num_qubits"   : num_dim,
            "num_reupload" : 2,
            "num_qlayers"  : 1,
            "num_clayers"  : 2,
            "num_chidden"  : 4 * num_dim,
        }
        data_config = {
            "num_data"    : 16,
            "num_dim"     : num_dim,
            "batch_size"  : batch_size,
            "num_workers" : num_workers,
        }
        test_time(wb, c_device, model_config, data_config)