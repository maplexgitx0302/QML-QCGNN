# %%
"""
### Packages and Configurations
"""

# %%
# basic packages
import os, time, sys
import argparse
from itertools import product
import matplotlib.pyplot as plt

# model template
import m_nn
import m_lightning

# data
import d_mg5_data
import awkward as ak

# qml
import pennylane as qml
from pennylane import numpy as np

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

# pytorch_lightning
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar

# pytorch_geometric
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import MessagePassing

# wandb
import wandb
from lightning.pytorch.loggers import WandbLogger
wandb.login()

# reproducibility
L.seed_everything(3020616)

# faster calculation on GPU but less precision
torch.set_float32_matmul_precision("medium")

# directory for saving results
root_dir = f"./result"
if os.path.isdir(root_dir) == False:
    os.makedirs(root_dir)

# %%
# global settings
cf = {}
cf["time"]     = input("Type a date time or leave it space to use default date time: ")
cf["device"]   = input("Enter the device (3080, 4090, slurm, node, etc.): ")
cf["wandb"]    = True # <-----------------------------------------------
cf["project"]  = "g_main"

# training configuration
cf["lr"]                = None
cf["rnd_seed"]          = int(input("Set random seed = "))
cf["num_train_ratio"]   = 0.8
cf["num_bin_data"]      = 500 # <-----------------------------------------------
cf["batch_size"]        = 100 # <-----------------------------------------------
cf["num_workers"]       = 0
cf["max_epochs"]        = 30 # <-----------------------------------------------
cf["accelerator"]       = "cpu"
cf["fast_dev_run"]      = False
cf["log_every_n_steps"] = cf["batch_size"] // 2

# %%
"""
### Data
"""

# %%
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class JetDataModule(pl.LightningDataModule):
    def __init__(self, sig_events, bkg_events, graph=True):
        super().__init__()
        # whether transform to torch_geometric graph data
        self.graph = graph

        # jet events
        self.max_num_ptcs = max(
            max(ak.count(sig_events["fast_pt"], axis=1)),
            max(ak.count(bkg_events["fast_pt"], axis=1)))
        sig_events = self._preprocess(sig_events)
        bkg_events = self._preprocess(bkg_events)
        print(f"\nDataLog: Max number of particles = {self.max_num_ptcs}\n")

        # prepare dataset for dataloader
        train_idx = int(cf["num_train_ratio"] * len(sig_events))
        self.train_dataset = self._dataset(sig_events[:train_idx], 1) + self._dataset(bkg_events[:train_idx], 0)
        self.test_dataset  = self._dataset(sig_events[train_idx:], 1) + self._dataset(bkg_events[train_idx:], 0)

    def _preprocess(self, events):
        # "_" prefix means that it is a fastjet feature
        fatjet_radius = 0.8
        f1 = np.arctan(events["fast_pt"] / events["fatjet_pt"])
        f2 = events["fast_delta_eta"] / fatjet_radius * (np.pi/2)
        f3 = events["fast_delta_phi"] / fatjet_radius * (np.pi/2)
        arrays = ak.zip([f1, f2, f3])
        arrays = arrays.to_list()
        events = [torch.tensor(arrays[i], dtype=torch.float32, requires_grad=False) for i in range(len(arrays))]
        return events

    def _dataset(self, events, y):
        if self.graph == True:
            # create pytorch_geometric "Data" object
            dataset = []
            for i in range(len(events)):
                x = events[i]
                edge_index = list(product(range(len(x)), range(len(x))))
                edge_index = torch.tensor(edge_index, requires_grad=False).transpose(0, 1)
                dataset.append(Data(x=x, edge_index=edge_index, y=y))
        else:
            pad     = lambda x: torch.nn.functional.pad(x, (0,0,0,self.max_num_ptcs-len(x)), mode="constant", value=0)
            dataset = TorchDataset(x=[pad(events[i]) for i in range(len(events))], y=[y]*len(events))
        return dataset

    def train_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.train_dataset, batch_size=cf["batch_size"], shuffle=True)
        else:
            return TorchDataLoader(self.train_dataset, batch_size=cf["batch_size"], shuffle=True)

    def val_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.test_dataset, batch_size=cf["batch_size"], shuffle=False)
        else:
            return TorchDataLoader(self.test_dataset, batch_size=cf["batch_size"], shuffle=False)

    def test_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.test_dataset, batch_size=cf["batch_size"], shuffle=False)
        else:
            return TorchDataLoader(self.test_dataset, batch_size=cf["batch_size"], shuffle=False)

# %%
"""
### Classical GNN Model
"""

# %%
class MessagePassing(MessagePassing):
    def __init__(self, phi):
        super().__init__(aggr="add", flow="target_to_source")
        self.phi = phi
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.phi(torch.cat((x_i, x_j), dim=-1))
    def update(self, aggr_out, x):
        return aggr_out

class Graph2PCGNN(nn.Module):
    def __init__(self, phi, mlp):
        super().__init__()
        self.gnn = MessagePassing(phi)
        self.mlp = mlp
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = geom_nn.global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class Classical2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_in, gnn_out, gnn_hidden, gnn_layers, mlp_hidden=0, mlp_layers=0, **kwargs):
        phi = m_nn.ClassicalMLP(in_channel=gnn_in, out_channel=gnn_out, hidden_channel=gnn_hidden, num_layers=gnn_layers)
        mlp = m_nn.ClassicalMLP(in_channel=gnn_out, out_channel=1, hidden_channel=mlp_hidden, num_layers=mlp_layers)
        super().__init__(phi, mlp)

# %%
"""
### Quantum Fully Connected GNN Model
"""

# %%
class QuantumFCGNN(nn.Module):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, ctrl_enc_operator, **kwargs):
        super().__init__()
        self.phi = m_nn.QuantumDisorderedFCGraph(
                       num_idx_qubits    = gnn_idx_qubits, 
                       num_nn_qubits     = gnn_nn_qubits, 
                       num_layers        = gnn_layers, 
                       num_reupload      = gnn_reupload,
                       ctrl_enc_operator = ctrl_enc_operator,
                       **kwargs,
                       )
        self.mlp = m_nn.ClassicalMLP(in_channel=gnn_nn_qubits, out_channel=1, hidden_channel=0, num_layers=0)
    def forward(self, x):
        # inputs should be 1-dim for each data, otherwise it would be confused with batch shape
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = self.phi(x)
        x = self.mlp(x)
        return x

class QuantumRotFCGNN(QuantumFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, **kwargs):
        def ctrl_enc_operator(_input, control, control_values):
            for i in range(gnn_nn_qubits):
                ctrl_H = qml.ctrl(qml.Hadamard, control=control, control_values=control_values)
                ctrl_H(wires=gnn_idx_qubits+i)
                ctrl_R = qml.ctrl(qml.Rot, control=control, control_values=control_values)
                ctrl_R(theta=_input[0], phi=_input[1], omega=_input[2], wires=gnn_idx_qubits+i)
        super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, ctrl_enc_operator, **kwargs)

class QuantumWtRotFCGNN(QuantumRotFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, **kwargs):
        # Weighted QuantumRyFCGNN
        super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, **{"pt_weight":True})

# %%
"""
### Wandb Training
"""

# %%
def train_wandb(model, data_module, train_info, graph, suffix=""):
    # setup wandb logger
    wandb_info = {}
    if cf["wandb"]:
        wandb_info["project"]  = cf["project"]
        wandb_info["group"]    = f"{train_info['sig']}_{train_info['bkg']}"
        wandb_info["name"]     = f"{train_info['group_rnd']} | {cf['time']}_{cf['device']}_{train_info['rnd_seed']} {suffix}"
        wandb_info["id"]       = wandb_info["name"]
        wandb_info["save_dir"] = root_dir 
        wandb_logger = WandbLogger(**wandb_info)
        wandb_config = {}
        wandb_config.update(cf)
        wandb_config.update(train_info)
        wandb_config.update(wandb_info)
        wandb_logger.experiment.config.update(wandb_config, allow_val_change=True)
        wandb_logger.watch(model, log="all")

    # start lightning training
    logger  = wandb_logger if cf["wandb"] else None
    trainer = L.Trainer(
        logger               = logger, 
        accelerator          = cf["accelerator"],
        max_epochs           = cf["max_epochs"],
        fast_dev_run         = cf["fast_dev_run"],
        log_every_n_steps    = cf["log_every_n_steps"],
        num_sanity_val_steps = 0,
        )
    litmodel = m_lightning.BinaryLitModel(model, lr=cf["lr"], graph=graph)

    # load ckpt file if exists
    try:
        ckpt_dir = f"result/{cf['project']}/{wandb_info['id']}/checkpoints"
        for _file in os.listdir(ckpt_dir):
            if _file.endswith("ckpt"):
                ckpt_path = f"{ckpt_dir}/{_file}"
    except:
        ckpt_path = None

    # print information
    print("-------------------- Training information --------------------\n")
    print("model:", model.__class__.__name__, model, "")
    print("config:", cf, "")
    print("train_info:", train_info, "")
    print("wandb_info:", wandb_info, "")
    print("--------------------------------------------------------------\n")
    
    trainer.fit(litmodel, datamodule=data_module, ckpt_path=ckpt_path)
    trainer.test(litmodel, datamodule=data_module)

    # finish wandb monitoring
    if cf["wandb"]:
        wandb.finish()

# %%
"""
### Training
"""

# %%
data_info = {"sig": "VzToZhToVevebb", "bkg": "VzToQCD", "abbrev":"BB-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":cf["num_bin_data"], "num_ptcs_limit":None, "num_pt_ptcs":8}
# data_info = {"sig": "VzToTt", "bkg": "VzToQCD", "abbrev":"TT-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":cf["num_bin_data"], "num_ptcs_limit":None, "num_pt_ptcs":8}

sig_fatjet_events = d_mg5_data.FatJetEvents(channel=data_info["sig"], cut_pt=data_info["cut"], subjet_radius=data_info["subjet_radius"], num_pt_ptcs=data_info["num_pt_ptcs"])
bkg_fatjet_events = d_mg5_data.FatJetEvents(channel=data_info["bkg"], cut_pt=data_info["cut"], subjet_radius=data_info["subjet_radius"], num_pt_ptcs=data_info["num_pt_ptcs"])

L.seed_everything(cf["rnd_seed"])

sig_events  = sig_fatjet_events.generate_uniform_pt_events(bin=data_info["bin"], num_bin_data=data_info["num_bin_data"], num_ptcs_limit=data_info["num_ptcs_limit"])
bkg_events  = bkg_fatjet_events.generate_uniform_pt_events(bin=data_info["bin"], num_bin_data=data_info["num_bin_data"], num_ptcs_limit=data_info["num_ptcs_limit"])
data_suffix = f"{data_info['abbrev']}_cut{data_info['cut']}_ptc{data_info['num_pt_ptcs']}_bin{data_info['bin']}-{data_info['num_bin_data']}_R{data_info['subjet_radius']}"

def train_model(model_class, model_dict, model_suffix, graph, suffix=""):
    data_module = JetDataModule(sig_events, bkg_events, graph=graph)
    model       = model_class(**model_dict)
    train_info  = {"rnd_seed":cf["rnd_seed"], "model_name":model.__class__.__name__, "model_suffix":model_suffix}
    train_info["group_rnd"] = f"{model.__class__.__name__}_{model_suffix} | {data_suffix}"
    train_info.update(model_dict)
    train_info.update(data_info)
    train_wandb(model, data_module, train_info, graph=graph, suffix=suffix)

# # classical ML only
# for gnn_out, gnn_layers in product([3,6,9], [1,2,4,8,12]):
#     cf["lr"]     =  1E-3
#     model_dict   = {"gnn_in":6, "gnn_out":gnn_out, "gnn_hidden":gnn_out, "gnn_layers":gnn_layers, "mlp_hidden":0, "mlp_layers":0}
#     go, gh, gl   = model_dict["gnn_out"], model_dict["gnn_hidden"], model_dict["gnn_layers"]
#     mh, ml       = model_dict["mlp_hidden"], model_dict["mlp_layers"]
#     model_suffix = f"go{go}_gh{gh}_gl{gl}_mh{mh}_ml{ml}"
#     train_model(model_class=Classical2PCGNN, model_dict=model_dict, model_suffix=model_suffix, graph=True)

# QFCGNN
max_sig_ptcs    = max(ak.count(sig_events["fast_pt"], axis=1))
max_bkg_ptcs    = max(ak.count(bkg_events["fast_pt"], axis=1))
gnn_idx_qubits  = int(np.ceil(np.log2(max(max_sig_ptcs, max_bkg_ptcs))))
cf["lr"]        = 1E-4
model_dict      = {"gnn_idx_qubits":gnn_idx_qubits, "gnn_nn_qubits":1, "gnn_layers":0, "gnn_reupload":0}
qidx, qnn       = gnn_idx_qubits, model_dict["gnn_nn_qubits"]
gl, gr          = model_dict["gnn_layers"], model_dict["gnn_reupload"]
model_suffix    = f"qidx{qidx}_qnn{qnn}_gl{gl}_gr{gr}"
train_model(model_class=QuantumRotFCGNN, model_dict=model_dict, model_suffix=model_suffix, graph=False)