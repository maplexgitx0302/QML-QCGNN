# %%
"""
### Packages and Configurations
"""

# %%
# basic packages
import os, time, sys, argparse
from itertools import product
import matplotlib.pyplot as plt

# model template
import module_model
import module_training

# data
import module_data
import awkward as ak

# qml
import pennylane as qml
from pennylane import numpy as np
print(f"\nPennylane default config path  = {qml.default_config.path}")
print(f"Pennylane default config setup = {qml.default_config}\n")

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader

# pytorch_lightning
import lightning as L
import lightning.pytorch as pl

# pytorch_geometric
import torch_geometric.nn as geomodule_model
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import MessagePassing

# reproducibility
L.seed_everything(3020616)

# faster calculation on GPU but less precision
torch.set_float32_matmul_precision("medium")

# directory for saving results
result_dir = "./result"
os.makedirs(result_dir, exist_ok=True)

# %%
# global settings
config = {}
config["wandb"]    = True # <-----------------------------------------------
# config["time"]     = input("Specify a datetime or leave empty (default current time): ") or time.strftime("%Y%m%d_%H%M%S", time.localtime())
# config["device"]   = input("Enter the computing device (4090, slurm, node, etc.)    : ")
# config["project"]  = "g_main"
# config["suffix"]   = ""
# config["rnd_seed"] = int(input("Set random seed = "))
# config["qdevice"]     = "default.qubit"
# config["qbackend"]    = ""
# config["diff_method"] = "best"

# test
config["time"]     = "20231123"
config["device"]   = "4090"
config["project"]  = "g_main"
config["suffix"]   = "qml_psr"
config["rnd_seed"] = 0
config["qdevice"]     = "default.qubit"
# config["qdevice"]     = "qiskit.aer"
config["qbackend"]    = None
config["diff_method"] = "parameter-shift"
config["use_qiskit_enc"] = True

# # parser (if needed)
# parser = argparse.ArgumentParser(description='argparse for slurm')
# parser.add_argument('--rnd_seed', type=int, help='random seed')
# parse_args = parser.parse_args()
# config["rnd_seed"] = parse_args.rnd_seed

# training configuration
config["num_train_ratio"]   = 0.8
config["num_bin_data"]      = 500
config["batch_size"]        = 100
config["num_workers"]       = 0
config["max_epochs"]        = 30 # <-----------------------------------------------
config["accelerator"]       = "cpu"
config["fast_dev_run"]      = False
config["log_every_n_steps"] = config["batch_size"] // 2

# %%
"""
### Data
"""

# %%
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
        train_idx = int(config["num_train_ratio"] * len(sig_events))
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
            dataset = module_training.TorchDataset(x=[pad(events[i]) for i in range(len(events))], y=[y]*len(events))
        return dataset

    def train_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.train_dataset, batch_size=config["batch_size"], shuffle=True)
        else:
            return TorchDataLoader(self.train_dataset, batch_size=config["batch_size"], shuffle=True)

    def val_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.test_dataset, batch_size=config["batch_size"], shuffle=False)
        else:
            return TorchDataLoader(self.test_dataset, batch_size=config["batch_size"], shuffle=False)

    def test_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.test_dataset, batch_size=config["batch_size"], shuffle=False)
        else:
            return TorchDataLoader(self.test_dataset, batch_size=config["batch_size"], shuffle=False)

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
        x = geomodule_model.global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class Classical2PCGNN(Graph2PCGNN):
    def __init__(self, gnn_in, gnn_out, gnn_hidden, gnn_layers, mlp_hidden=0, mlp_layers=0, **kwargs):
        phi = module_model.ClassicalMLP(in_channel=gnn_in, out_channel=gnn_out, hidden_channel=gnn_hidden, num_layers=gnn_layers)
        mlp = module_model.ClassicalMLP(in_channel=gnn_out, out_channel=1, hidden_channel=mlp_hidden, num_layers=mlp_layers)
        super().__init__(phi, mlp)

# %%
"""
### Quantum Complete Graph Neural Network (QCGNN)
"""

# %%
class QuantumRotQCGNN(nn.Module):
    def __init__(self, num_ir_qubits, num_nr_qubits, num_layers, num_reupload, device="default.qubit", backend="ibmq_qasm_simulator", diff_method="best", **kwargs):
        super().__init__()
        # rotation encoding on pennylane simulator
        def qml_encoding(_input, control_values):
            for i in range(num_nr_qubits):
                ctrl_H = qml.ctrl(qml.Hadamard, control=range(num_ir_qubits), control_values=control_values)
                ctrl_H(wires=num_ir_qubits+i)
                ctrl_R = qml.ctrl(qml.Rot, control=range(num_ir_qubits), control_values=control_values)
                ctrl_R(theta=_input[0], phi=_input[1], omega=_input[2], wires=num_ir_qubits+i)
        
        # rotation encoding on qiskit
        num_wk_qubits = num_ir_qubits - 1
        def qiskit_encoding(_input, control_values):
            theta, phi, omega = _input[0], _input[1], _input[2]
            # see N.C. page.184
            for i in range(num_nr_qubits):
                # control values
                for q in range(num_ir_qubits):
                    if control_values[q] == 0:
                        qml.PauliX(wires=q)
                # toffoli transformation
                if num_ir_qubits >= 2:
                    qml.Toffoli(wires=(0, 1, num_ir_qubits))
                for q in range(num_wk_qubits-1):
                    qml.Toffoli(wires=(2+q, num_ir_qubits+q, num_ir_qubits+1+q))
                # ctrl_H: decomposed by H = i Rx(pi) Ry(pi/2) (if complete graph with power of 2 nodes -> relative phase i becomes global)
                target_qubit = num_ir_qubits + num_wk_qubits + i
                qml.CRY(np.pi/2, wires=(num_ir_qubits + num_wk_qubits - 1, target_qubit))
                qml.CRX(np.pi, wires=(num_ir_qubits + num_wk_qubits - 1, target_qubit))
                # ctrl_R: Rot(phi, theta, omega) = Rz(omega) Ry(theta) Rz(phi)
                qml.CRZ(phi, wires=(num_ir_qubits + num_wk_qubits - 1, target_qubit))
                qml.CRY(theta, wires=(num_ir_qubits + num_wk_qubits - 1, target_qubit))
                qml.CRZ(omega, wires=(num_ir_qubits + num_wk_qubits - 1, target_qubit))
                # toffoli inverse transformation
                for q in reversed(range(num_wk_qubits-1)):
                    qml.Toffoli(wires=(2+q, num_ir_qubits+q, num_ir_qubits+1+q))
                if num_ir_qubits >= 2:
                    qml.Toffoli(wires=(0, 1, num_ir_qubits))
                # control values
                for q in range(num_ir_qubits):
                    if control_values[q] == 0:
                        qml.PauliX(wires=q)

        # constructing QCGNN like a MPGNN
        if "qiskit" in device or config["use_qiskit_enc"]:
            self.phi = module_model.QCGNN(num_ir_qubits, num_nr_qubits, num_layers, num_reupload, ctrl_enc=qiskit_encoding, device=device, backend=backend, diff_method=diff_method)
        else:
            self.phi = module_model.QCGNN(num_ir_qubits, num_nr_qubits, num_layers, num_reupload, ctrl_enc=qml_encoding, device=device, diff_method=diff_method)
        self.mlp = module_model.ClassicalMLP(in_channel=num_nr_qubits, out_channel=1, hidden_channel=0, num_layers=0)
    
    def forward(self, x):
        # inputs should be 1-dim for each data, otherwise it would be confused with batch shape
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = self.phi(x)
        x = self.mlp(x)
        return x

# %%
"""
### Training
"""

# %%
def train(model, model_config, data_module, data_config, graph, suffix=""):
    # use wandb monitoring if needed
    if config["wandb"] == True:
        model_config["model_name"] = model.__class__.__name__
        model_config["group_rnd"]  = f"{model_config['model_name']}_{model_config['model_suffix']} | {data_config['data_suffix']}"
        logger_config = {}
        logger_config["project"]  = config["project"]
        logger_config["group"]    = f"{data_config['sig']}_{data_config['bkg']}"
        if "qiskit" in config['qdevice']:
            logger_config["name"]     = f"{model_config['group_rnd']} | {config['time']}_{config['qdevice']}_{config['rnd_seed']} {suffix}"
        else:
            logger_config["name"]     = f"{model_config['group_rnd']} | {config['time']}_{config['device']}_{config['rnd_seed']} {suffix}"
        logger_config["id"]       = logger_config["name"]
        logger_config["save_dir"] = result_dir
        logger = module_training.wandb_monitor(model, logger_config, config, model_config, data_config)
    else:
        logger = None

    # training information
    print("-------------------- Training information --------------------\n")
    print("config:", config, "")
    print("data_config:", data_config, "")
    print("model_config:", model_config, "")
    if config["wandb"] == True:
        print("logger_config:", logger_config, "")
    print("--------------------------------------------------------------\n")
    
    # pytorch lightning setup
    trainer = L.Trainer(
        logger               = logger, 
        accelerator          = config["accelerator"],
        max_epochs           = config["max_epochs"],
        fast_dev_run         = config["fast_dev_run"],
        log_every_n_steps    = config["log_every_n_steps"],
        num_sanity_val_steps = 0,
        )
    litmodel = module_training.BinaryLitModel(model, lr=model_config["lr"], graph=graph)
    trainer.fit(litmodel, datamodule=data_module)
    trainer.test(litmodel, datamodule=data_module)

    # finish wandb monitoring
    if config["wandb"] == True:
        module_training.wandb_finish()

# %%
# data_config = {"sig": "VzToZhToVevebb", "bkg": "VzToQCD", "abbrev":"BB-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":8}
data_config = {"sig": "VzToTt", "bkg": "VzToQCD", "abbrev":"TT-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":8}
# data_config = {"sig": "VzToTt", "bkg": "VzToQCD", "abbrev":"TT-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":2}

sig_fatjet_events = module_data.FatJetEvents(channel=data_config["sig"], cut_pt=data_config["cut"], subjet_radius=data_config["subjet_radius"], num_pt_ptcs=data_config["num_pt_ptcs"])
bkg_fatjet_events = module_data.FatJetEvents(channel=data_config["bkg"], cut_pt=data_config["cut"], subjet_radius=data_config["subjet_radius"], num_pt_ptcs=data_config["num_pt_ptcs"])

for rnd_seed in range(1):
    config["rnd_seed"] = rnd_seed
    L.seed_everything(config["rnd_seed"])
    sig_events  = sig_fatjet_events.generate_uniform_pt_events(bin=data_config["bin"], num_bin_data=data_config["num_bin_data"])
    bkg_events  = bkg_fatjet_events.generate_uniform_pt_events(bin=data_config["bin"], num_bin_data=data_config["num_bin_data"])
    data_suffix = f"{data_config['abbrev']}_cut{data_config['cut']}_ptc{data_config['num_pt_ptcs']}_bin{data_config['bin']}-{data_config['num_bin_data']}_R{data_config['subjet_radius']}"
    data_config["data_suffix"] = data_suffix

    # # classical ML
    # data_module  = JetDataModule(sig_events, bkg_events, graph=True)
    # go, gh, gl   = 3, 3, 3
    # mh, ml       = 0, 0
    # model_suffix = f"go{go}_gh{gh}_gl{gl}_mh{mh}_ml{ml}"
    # model_config = {"gnn_in":6, "gnn_out":go, "gnn_hidden":gh, "gnn_layers":gl, "mlp_hidden":mh, "mlp_layers":ml, "lr":1E-3, "model_suffix":model_suffix}
    # model        = Classical2PCGNN(**model_config) 
    # train(model, model_config, data_module, data_config, graph=True, suffix=config["suffix"])

    # QFCGNN
    data_module   = JetDataModule(sig_events, bkg_events, graph=False)
    qidx, qnn     = int(np.ceil(np.log2(data_config["num_pt_ptcs"]))), 3
    gl, gr        = 1, 3
    model_suffix  = f"qidx{qidx}_qnn{qnn}_gl{gl}_gr{gr}"
    model_config  = {"gnn_idx_qubits":qidx, "gnn_nn_qubits":qnn, "gnn_layers":gl, "gnn_reupload":gr, "lr":1E-2, "model_suffix":model_suffix}
    model         = QuantumRotQCGNN(num_ir_qubits=qidx, num_nr_qubits=qnn, num_layers=gl, num_reupload=gr, device=config["qdevice"], backend=config["qbackend"], diff_method=config["diff_method"])
    train(model, model_config, data_module, data_config, graph=False, suffix=config["suffix"])