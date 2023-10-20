# %%
"""
### Packages and Configurations
"""

# %%
# basic packages
import os, time, sys
import argparse
from itertools import product
from collections import namedtuple
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

# argparser
use_parser = False
if use_parser:
    parser = argparse.ArgumentParser(description='Determine the structure of the quantum model.')
    parser.add_argument('--date_time', type=str, help='Date time in format Ymd_HMS')
    parser.add_argument('--q_gnn_layers', type=int, help='Quantum gnn layers')
    parser.add_argument('--q_gnn_reupload', type=int, help='Quantum gnn reupload')
    parser.add_argument('--rnd_seed', type=int, help='Random seed')
    parse_args = parser.parse_args()
else:
    parse_fields = ["date_time", "q_gnn_layers", "q_gnn_reupload", "q_gnn_num_qnn", "rnd_seed"]
    parse_tuple  = namedtuple('parse_tuple', " ".join(parse_fields))
    parse_args   = parse_tuple(
        date_time      = time.strftime("%Y%m%d_%H%M%S", time.localtime()),
        rnd_seed       = 0,
        q_gnn_layers   = 2,
        q_gnn_reupload = 0,
        q_gnn_num_qnn  = 1,
    )

# %%
# global settings
cf = {}
cf["time"]     = input("Type a date time or leave it space to use default date time: ") or parse_args.date_time
cf["wandb"]    = True # <-----------------------------------------------
cf["project"]  = "g_eflow_QFCGNN"

# training configuration
cf["lr"]                = 1E-4
cf["rnd_seed"]          = parse_args.rnd_seed
cf["num_train_ratio"]   = 0.8
cf["num_bin_data"]      = 500 # <-----------------------------------------------
cf["batch_size"]        = 50 # <-----------------------------------------------
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
    def __init__(self, sig_events, bkg_events, mode=None, graph=True):
        super().__init__()
        # whether transform to torch_geometric graph data
        self.graph = graph

        # jet events
        self.max_num_ptcs = max(
            max(ak.count(sig_events["fast_pt"], axis=1)),
            max(ak.count(bkg_events["fast_pt"], axis=1)))
        sig_events = self._preprocess(sig_events, mode)
        bkg_events = self._preprocess(bkg_events, mode)
        print(f"\nDataLog: Max number of particles = {self.max_num_ptcs}\n")

        # prepare dataset for dataloader
        train_idx = int(cf["num_train_ratio"] * len(sig_events))
        self.train_dataset = self._dataset(sig_events[:train_idx], 1) + self._dataset(bkg_events[:train_idx], 0)
        self.test_dataset  = self._dataset(sig_events[train_idx:], 1) + self._dataset(bkg_events[train_idx:], 0)

    def _preprocess(self, events, mode):
        # "_" prefix means that it is a fastjet feature
        if mode == "normalize":
            f1 = np.arctan(events["fast_pt"] / events["fatjet_pt"])
            f2 = events["fast_delta_eta"]
            f3 = events["fast_delta_phi"]
            arrays = ak.zip([f1, f2, f3])
        elif mode == "normalize_pi":
            fatjet_radius = 0.8
            f1 = np.arctan(events["fast_pt"] / events["fatjet_pt"])
            f2 = events["fast_delta_eta"] / fatjet_radius * (np.pi/2)
            f3 = events["fast_delta_phi"] / fatjet_radius * (np.pi/2)
            arrays = ak.zip([f1, f2, f3])
        elif mode == "tri_eflow":
            f1_s = np.sin(np.arctan(events["fast_pt"] / events["fatjet_pt"])/2)
            f1_c = np.cos(np.arctan(events["fast_pt"] / events["fatjet_pt"])/2)
            f2_s = np.sin(events["fast_delta_eta"]/2)
            f2_c = np.cos(events["fast_delta_eta"]/2)
            f3_s = np.sin(events["fast_delta_phi"]/2)
            f3_c = np.cos(events["fast_delta_phi"]/2)
            arrays = ak.zip([f1_s, f1_c, f2_s, f2_c, f3_s, f3_c])
        elif mode == "tri_eflow_pi":
            fatjet_radius = 0.8
            f1_s = np.sin(np.arctan(events["fast_pt"] / events["fatjet_pt"]) / 2)
            f1_c = np.cos(np.arctan(events["fast_pt"] / events["fatjet_pt"]) / 2)
            f2_s = np.sin((events["fast_delta_eta"] / fatjet_radius * (np.pi/2)) / 2)
            f2_c = np.cos((events["fast_delta_eta"] / fatjet_radius * (np.pi/2)) / 2)
            f3_s = np.sin((events["fast_delta_phi"] / fatjet_radius * (np.pi/2)) / 2)
            f3_c = np.cos((events["fast_delta_phi"] / fatjet_radius * (np.pi/2)) / 2)
            arrays = ak.zip([f1_s, f1_c, f2_s, f2_c, f3_s, f3_c])
        elif mode == "":
            f1 = events["fast_pt"]
            f2 = events["fast_delta_eta"]
            f3 = events["fast_delta_phi"]
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
### Quantum Trivial GNN Model
"""

# %%
class QuantumAngle2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_qubits, gnn_layers, gnn_reupload, gnn_measurements, **kwargs):
        phi = m_nn.QuantumMLP(num_qubits=gnn_qubits, num_layers=gnn_layers, num_reupload=gnn_reupload, measurements=gnn_measurements)
        mlp = m_nn.ClassicalMLP(in_channel=len(gnn_measurements), out_channel=1, hidden_channel=0, num_layers=0)
        super().__init__(phi, mlp)

# %%
"""
### Quantum Fully Connected GNN Model
"""

# %%
class QuantumFCGNN(nn.Module):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, ctrl_enc_operator, **kwargs):
        super().__init__()
        self.phi = nn.ModuleList([
            m_nn.QuantumDisorderedFCGraph(
                num_idx_qubits    = gnn_idx_qubits, 
                num_nn_qubits     = gnn_nn_qubits, 
                num_layers        = gnn_layers, 
                num_reupload      = gnn_reupload,
                ctrl_enc_operator = ctrl_enc_operator,
                **kwargs,
                ) for _ in range(gnn_num_qnn)
                ])
        self.mlp = m_nn.ClassicalMLP(in_channel=gnn_num_qnn*gnn_nn_qubits, out_channel=1, hidden_channel=0, num_layers=0)
    def forward(self, x):
        # inputs should be 1-dim for each data, otherwise it would be confused with batch shape
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = torch.cat([self.phi[i](x) for i in range(len(self.phi))], dim=-1)
        x = self.mlp(x)
        return x
    
class QuantumRyFCGNN(QuantumFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **kwargs):
        def ctrl_enc_operator(_input, control, control_values):
            ctrl = qml.ctrl(qml.AngleEmbedding, control=control, control_values=control_values)
            ctrl(features=_input, wires=range(gnn_idx_qubits, gnn_idx_qubits+3), rotation="Y")
        super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, ctrl_enc_operator, **kwargs)

class QuantumWtRyFCGNN(QuantumRyFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **kwargs):
        # Weighted QuantumRyFCGNN
        super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **{"pt_weight":True})
    
class QuantumRotFCGNN(QuantumFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **kwargs):
        def ctrl_enc_operator(_input, control, control_values):
            ctrl_H = qml.ctrl(qml.Hadamard, control=control, control_values=control_values)
            ctrl_H(wires=gnn_idx_qubits)
            ctrl_R = qml.ctrl(qml.Rot, control=control, control_values=control_values)
            ctrl_R(theta=_input[0], phi=_input[1], omega=_input[2], wires=gnn_idx_qubits)
        super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, ctrl_enc_operator, **kwargs)

class QuantumWtRotFCGNN(QuantumRotFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **kwargs):
        # Weighted QuantumRyFCGNN
        super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **{"pt_weight":True})

class QuantumMultiRotFCGNN(QuantumFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **kwargs):
        def ctrl_enc_operator(_input, control, control_values):
            for i in range(gnn_nn_qubits):
                ctrl_H = qml.ctrl(qml.Hadamard, control=control, control_values=control_values)
                ctrl_H(wires=gnn_idx_qubits+i)
                ctrl_R = qml.ctrl(qml.Rot, control=control, control_values=control_values)
                ctrl_R(theta=_input[0], phi=_input[1], omega=_input[2], wires=gnn_idx_qubits+i)
        super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, ctrl_enc_operator, **kwargs)

class QuantumWtMultiRotFCGNN(QuantumMultiRotFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **kwargs):
        # Weighted QuantumRyFCGNN
        super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **{"pt_weight":True})

# class QuantumAmpFCGNN(QuantumFCGNN):
#     def __init__(self, gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, **kwargs):
#         def ctrl_enc_operator(_input, control, control_values):
#             ctrl = qml.ctrl(qml.AmplitudeEmbedding, control=control, control_values=control_values)
#             ctrl(features=_input, wires=range(gnn_idx_qubits, gnn_idx_qubits+2), pad_with=np.pi/2, normalize=True)
#         super().__init__(gnn_idx_qubits, gnn_nn_qubits, gnn_layers, gnn_reupload, gnn_num_qnn, ctrl_enc_operator, **kwargs)

# %%
class QuantumSuperFCGNN(nn.Module):
    def __init__(self, gnn_idx_qubits, gnn_max_reupload, **kwargs):
        super().__init__()
        def angle_ctrl_enc_operator(_input, control, control_values):
            ctrl = qml.ctrl(qml.AngleEmbedding, control=control, control_values=control_values)
            ctrl(features=_input, wires=range(gnn_idx_qubits, gnn_idx_qubits+3), rotation="Y")
        def rot_ctrl_enc_operator(_input, control, control_values):
            ctrl_H = qml.ctrl(qml.Hadamard, control=control, control_values=control_values)
            ctrl_H(wires=gnn_idx_qubits)
            ctrl_R = qml.ctrl(qml.Rot, control=control, control_values=control_values)
            ctrl_R(theta=_input[0], phi=_input[1], omega=_input[2], wires=gnn_idx_qubits)
        module       = m_nn.QuantumDisorderedFCGraph
        angle_module = lambda r: module(gnn_idx_qubits, num_nn_qubits=3, num_layers=2, num_reupload=r, ctrl_enc_operator=angle_ctrl_enc_operator)
        rot_module   = lambda r: module(gnn_idx_qubits, num_nn_qubits=2, num_layers=2, num_reupload=r, ctrl_enc_operator=rot_ctrl_enc_operator)
        self.phi     = nn.ModuleList([angle_module(r) for r in range(gnn_max_reupload+1)])
        self.phi    += nn.ModuleList([rot_module(r) for r in range(gnn_max_reupload+1)])
        self.mlp     = m_nn.ClassicalMLP(in_channel=(3+2)*(gnn_max_reupload+1), out_channel=1, hidden_channel=0, num_layers=0)
    def forward(self, x):
        # inputs should be 1-dim for each data, otherwise it would be confused with batch shape
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = torch.cat([self.phi[i](x) for i in range(len(self.phi))], dim=-1)
        x = self.mlp(x)
        return x
    
class HybridSuperFCGNN(QuantumSuperFCGNN):
    def __init__(self, gnn_idx_qubits, gnn_max_reupload, **kwargs):
        super().__init__(gnn_idx_qubits, gnn_max_reupload, **kwargs)
        self.pre_mlp = nn.Linear(3, 3)
    def forward(self, x):
        # pre-linear
        x = self.pre_mlp(x)
        # inputs should be 1-dim for each data, otherwise it would be confused with batch shape
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = torch.cat([self.phi[i](x) for i in range(len(self.phi))], dim=-1)
        x = self.mlp(x)
        return x

# %%
"""
### Wandb Training
"""

# %%
def train(model, data_module, train_info, suffix="", graph=True):
    # setup wandb logger
    wandb_info = {}
    if cf["wandb"]:
        wandb_info["project"]  = cf["project"]
        wandb_info["group"]    = f"{train_info['sig']}_{train_info['bkg']}"
        wandb_info["name"]     = f"{train_info['group_rnd']} | {cf['time']}_{train_info['rnd_seed']}{suffix}"
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
for num_pt_ptcs in [8]:
    data_info = {"sig": "VzToZhToVevebb", "bkg": "VzToQCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":cf["num_bin_data"], "num_ptcs_limit":None, "num_pt_ptcs":num_pt_ptcs}
    # data_info = {"sig": "VzToTt", "bkg": "VzToQCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":cf["num_bin_data"], "num_ptcs_limit":None, "num_pt_ptcs":num_pt_ptcs}
    sig_fatjet_events = d_mg5_data.FatJetEvents(channel=data_info["sig"], cut_pt=data_info["cut"], subjet_radius=data_info["subjet_radius"], num_pt_ptcs=data_info["num_pt_ptcs"])
    bkg_fatjet_events = d_mg5_data.FatJetEvents(channel=data_info["bkg"], cut_pt=data_info["cut"], subjet_radius=data_info["subjet_radius"], num_pt_ptcs=data_info["num_pt_ptcs"])

    for rnd_seed in range(1):
        cf["rnd_seed"] = rnd_seed
        L.seed_everything(cf["rnd_seed"])

        sig_events  = sig_fatjet_events.generate_uniform_pt_events(bin=data_info["bin"], num_bin_data=data_info["num_bin_data"], num_ptcs_limit=data_info["num_ptcs_limit"])
        bkg_events  = bkg_fatjet_events.generate_uniform_pt_events(bin=data_info["bin"], num_bin_data=data_info["num_bin_data"], num_ptcs_limit=data_info["num_ptcs_limit"])
        data_suffix = f"cut{data_info['cut']}_ptc{data_info['num_pt_ptcs']}_bin{data_info['bin']}-{data_info['num_bin_data']}_R{data_info['subjet_radius']}"

        def train_classical(preprocess_mode, model_dict, suffix=""):
            data_module = JetDataModule(sig_events, bkg_events, preprocess_mode)
            model       = Classical2PCGNN(**model_dict)
            go, gh, gl  = model_dict['gnn_out'], model_dict['gnn_hidden'], model_dict['gnn_layers']
            mh, ml      = model_dict['mlp_hidden'], model_dict['mlp_layers']
            train_info  = {"rnd_seed":cf["rnd_seed"], "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            train_info["group_rnd"] = f"{model.__class__.__name__}_{preprocess_mode}_go{go}_gh{gh}_gl{gl}_mh{mh}_ml{ml} | {data_suffix}"
            train_info.update(model_dict)
            train_info.update(data_info)
            train(model, data_module, train_info, suffix=suffix)

        def train_qtrivial(preprocess_mode, model_dict, suffix=""):
            data_module = JetDataModule(sig_events, bkg_events, preprocess_mode)
            model       = QuantumAngle2PCGNN(**model_dict)
            gl, gr      = model_dict['gnn_layers'], model_dict['gnn_reupload']
            train_info  = {"rnd_seed":cf["rnd_seed"], "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            train_info["group_rnd"] = f"{model.__class__.__name__}_{preprocess_mode}_gr{gr}_gl{gl} | {data_suffix}"
            train_info.update(model_dict)
            train_info.update(data_info)
            train(model, data_module, train_info, suffix=suffix)

        def train_qfcgnn(preprocess_mode, model_class, model_dict, suffix=""):
            data_module = JetDataModule(sig_events, bkg_events, preprocess_mode, graph=False)
            model       = model_class(**model_dict)
            qidx, qnn   = model_dict['gnn_idx_qubits'], model_dict['gnn_nn_qubits']
            gl, gr      = model_dict['gnn_layers'], model_dict['gnn_reupload']
            num_qnn     = model_dict['gnn_num_qnn']
            train_info  = {"rnd_seed":cf["rnd_seed"], "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            train_info["group_rnd"]  = f"{model.__class__.__name__}_{preprocess_mode}_QNN{num_qnn}_qidx{qidx}_qnn{qnn}_gl{gl}_gr{gr} | {data_suffix}"
            train_info.update(model_dict)
            train_info.update(data_info)
            train(model, data_module, train_info, suffix=suffix, graph=False)

        def train_superqfcgnn(preprocess_mode, model_dict, suffix=""):
            data_module = JetDataModule(sig_events, bkg_events, preprocess_mode, graph=False)
            model       = HybridSuperFCGNN(**model_dict)
            train_info  = {"rnd_seed":cf["rnd_seed"], "model_name":model.__class__.__name__, "preprocess_mode":preprocess_mode}
            train_info["group_rnd"]  = f"{model.__class__.__name__}_{preprocess_mode}_qidx{model_dict['gnn_idx_qubits']}_maxgr{model_dict['gnn_max_reupload']} | {data_suffix}"
            train_info.update(model_dict)
            train_info.update(data_info)
            train(model, data_module, train_info, suffix=suffix, graph=False)

        # # classical ML only
        # for go, gh, gl in product([3,6], [16,64,256], [1,2,4]):
        #     model_dict = {
        #         "gnn_in":6, "gnn_out":go, "gnn_hidden":gh, "gnn_layers":gl, 
        #         "mlp_hidden":0, "mlp_layers":0
        #         }
        #     train_classical(preprocess_mode="normalize_pi", model_dict=model_dict)

        # # Trivial GNN
        # for gnn_layers, gnn_reupload in product((1,2), (0,1)):
        #     preprocess_mode  = "normalize_pi"
        #     # gnn_layers       = parse_args.q_gnn_layers
        #     # gnn_reupload     = parse_args.q_gnn_reupload
        #     gnn_qubits       = 6
        #     gnn_num_qnn      = parse_args.q_gnn_num_qnn
        #     model_dict       = {"gnn_layers":gnn_layers, "gnn_reupload":gnn_reupload, "gnn_qubits":gnn_qubits}
        #     gnn_measurements = list(product(range(gnn_qubits), ["Z"]))
        #     model_dict["gnn_measurements"] = gnn_measurements
        #     train_qtrivial(preprocess_mode, model_dict)

        # QFCGNN
        for gnn_layers, gnn_reupload, gnn_nn_qubits in product([2,4], [0,3,6], [3,6]):
            model_class     = QuantumMultiRotFCGNN
            gnn_idx_qubits  = int(np.ceil(np.log2(max(
                max(ak.count(sig_events["fast_pt"], axis=1)), 
                max(ak.count(bkg_events["fast_pt"], axis=1))))))
            preprocess_mode = "normalize_pi"
            # gnn_layers      = parse_args.q_gnn_layers
            # gnn_reupload    = parse_args.q_gnn_reupload
            gnn_num_qnn     = 1
            model_dict      = {"gnn_idx_qubits":gnn_idx_qubits, "gnn_nn_qubits":gnn_nn_qubits, "gnn_layers":gnn_layers, "gnn_reupload":gnn_reupload, "gnn_num_qnn":gnn_num_qnn}
            train_qfcgnn(preprocess_mode, model_class, model_dict)

        # # Super QFCGNN
        # for r in range(4):
        #     gnn_idx_qubits  = int(np.ceil(np.log2(max(
        #         max(ak.count(sig_events["fast_pt"], axis=1)), 
        #         max(ak.count(bkg_events["fast_pt"], axis=1))))))
        #     model_dict = {"gnn_idx_qubits":gnn_idx_qubits, "gnn_max_reupload":r}
        #     train_superqfcgnn("normalize_pi", model_dict)