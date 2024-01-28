# %%
"""
# Main script

This notebook script is about the main training procedure of MPGNN and QCGNN, with some functions and classes are written in other python scripts `module_*.py`. The following codes contain:
- Setup for the classical MPGNN.
- Data encoding ansatz of QCGNN (on simulator or IBMQ).
- Training and prediction workflow.

About the modules `module_*.py`:
- `module_data.py`: Reading data and constructing data modules.
- `module_model.py`: Some detail construction of classical and quantum models.
- `module_training.py`: Related to routine training procedure.
- `module_gmail.py`: Not used in this script (for gmail notification).
"""

# %%
"""
## Initialization
"""

# %%
"""
### Import packages

- `lightning`: tools for simplified training procedure.
- `pennylane`: used for quantum machine learning.
- `torch`: machine learning backend (for both classical and quantum).
- `torch_geometric`: used for classical graph neural network.
"""

# %%
import argparse
from itertools import product
import json
import time

import lightning as L
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import MessagePassing

import module_data
import module_model
import module_training

# Faster calculation on GPU but less precision.
# torch.set_float32_matmul_precision("medium")

# Choose QCGNN base model.
QCGNN = module_model.QCGNN_IX

# %%
"""
### Manual Settings

Most of the configuration setup can be done in `./config.json`, some explanation of the arguments in `./config.json` is listed below:
- Set `quick_test` to true to test whether the code can run in your environment.
- Set `computing_platform` to "slurm" if running in *slurm* (optional).
- Set `use_wandb` to true if you want to monitor the trainning procedure through [Wandb](https://wandb.ai/site), otherwise the training result will be saved as a `csv` file.
"""

# %%
# The logging function.
def _log(message: str) -> None:
    """Printing function for log."""
    print(f"# MainLog: {message}")

# Read json configuration file.
with open("config.json", "r") as json_file:
    # This json config might be loaded for other use in other python scripts.
    json_config = json.load(json_file)
    # Copy another config since it will be modified in this code.
    general_config = json_config.copy()

# This notebook script will be converted to python script for other usage, the
# following parts are intended not to be imported for other usage.
if __name__ == "__main__":
    # Settings for running in `slurm` or other platforms that are inconvenient
    # to pass inputs directly.
    if general_config["computing_platform"] == "slurm":
        # Pass arguments in shell script (e.g. python a.py --rnd_seed 0).
        parser = argparse.ArgumentParser(description='argparse for slurm')
        # Add `rnd_seed` argument to specify the random seed.
        parser.add_argument('--rnd_seed', type=int, help='random seed')
        parse_args = parser.parse_args()
        general_config["rnd_seed"] = parse_args.rnd_seed
    else:
        # Inputs for additional training information manually.
        # Record current time.
        general_config["ctime"]  = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # Specifying a datetime if needed (to compatible with other exp results).
        general_config["time"]   = input("Datetime (default current time): ") \
            or general_config["ctime"]
        # This will recording the classical device for simulation (optional).
        general_config["device"] = input("Classical computing device     : ")
        # The suffix of training result (optional).
        general_config["suffix"] = input("Suffix for this training exp   : ")

    # Whether using Wandb to monitor the training procedure.
    use_wandb = general_config["use_wandb"]
    if use_wandb:
        import wandb
        # Api can be used to fetch training results.
        api = wandb.Api()

    # Configuration of quantum devices (PennyLane simulation or IBMQ real devices).
    quantum_config = {"qdevice": "default.qubit", "qbackend": ""}

    # Test whether the code can run for at least 1 epoch.
    if general_config["quick_test"] == True:
        general_config["max_epochs"] = 1
        general_config["num_bin_data"] = 1

# %%
"""
### Dataset
"""

# %%
"""
##### Data configuration settings

The dictionary that contains the information of dataset. The channels will be determined in the training loop below.
"""

# %%
def generate_data_config(sig, bkg, abbrev, general_config):
    """Set up dataset configurations."""

    bin = general_config["bin"]
    cut_pt = (general_config["cut_pt_low"], general_config["cut_pt_high"])
    num_bin_data = general_config["num_bin_data"]
    max_num_ptcs = general_config["max_num_ptcs"]
    pad_num_ptcs = general_config["pad_num_ptcs"]
    pt_threshold = general_config["pt_threshold"]
    subjet_radius = general_config["subjet_radius"]
    num_ptcs_range = general_config["num_ptcs_range"]

    data_config = {
        "sig": sig,  # Signal channel.
        "bkg": bkg,  # Background channel.
        "abbrev": abbrev,  # Abbreviation name for the classification.
        "cut_pt": cut_pt,  # Select jets with pt in this range.
        "subjet_radius": subjet_radius,  # The radius for subjet reclustering.
        "bin": bin,  # Number of bins uniform distributed in range `cut_pt`.
        "num_bin_data": num_bin_data,  # Number of data in each bin.
        "max_num_ptcs": max_num_ptcs,  # Drop particles if exceed this number.
        "pad_num_ptcs": pad_num_ptcs,  # Pad to this number with zero pt.
        "pt_threshold": pt_threshold,  # Drop particles less than this threshold.
        "num_ptcs_range": num_ptcs_range,  # This is used for running on real device.
        "print_log": True,  # Whether print random sampling message.
    }
    
    data_config["data_suffix"] = (
        f"{abbrev}_ptc{max_num_ptcs}_thres{pt_threshold}"
        f"-nb{num_bin_data}_R{subjet_radius}"
    )
    
    return data_config

# %%
def generate_datamodule(
        data_config: dict,
        data_ratio: float,
        batch_size: int,
        graph: bool
    ):
    """Generate datamodule for `execute` usage
    
    Args:
        data_config : dict
            See code below for detail usage.
        data_ratio : float
            Ratio of (# of training) / (# of training + testing).
        batch_size : int
            Number of data per batch.
        graph : bool
            Whether the dataset is generated in graph structure or not.
    """

    # Generate uniform pt events.
    fatjet_events = lambda channel: module_data.FatJetEvents(
        channel=channel,
        cut_pt=data_config["cut_pt"],
        subjet_radius=data_config["subjet_radius"],
        max_num_ptcs=data_config["max_num_ptcs"],
        pt_threshold=data_config["pt_threshold"],
    )
    sig_fatjet_events = fatjet_events(data_config["sig"])
    bkg_fatjet_events = fatjet_events(data_config["bkg"])
    sig_events = sig_fatjet_events.generate_uniform_pt_events(
        bin=data_config["bin"],
        num_bin_data=data_config["num_bin_data"],
        num_ptcs_range=data_config["num_ptcs_range"],
        print_log=data_config["print_log"],
    )
    bkg_events = bkg_fatjet_events.generate_uniform_pt_events(
        bin=data_config["bin"],
        num_bin_data=data_config["num_bin_data"],
        num_ptcs_range=data_config["num_ptcs_range"],
        print_log=data_config["print_log"],
    )

    return module_data.JetDataModule(
        sig_events=sig_events,
        bkg_events=bkg_events,
        data_ratio=data_ratio,
        batch_size=batch_size,
        graph=graph,
        pad_num_ptcs=data_config["pad_num_ptcs"],
    )

# %%
"""
## Classical and Quantum Models
"""

# %%
"""
### Classical Message Passing Graph Neural Network (MPGNN)

This classical model is built with GNN structure followed by a simple shallow fully connected linear network. The GNN is constructed with package `torch_geometric`, see ["PyTorch Geometric"](https://pytorch-geometric.readthedocs.io) for futher details. The tutorial for creating a `MessagePassing` class can be found at ["Creating Message Passing Networks"](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html).
"""

# %%
class MessagePassing(MessagePassing):
    def __init__(self, phi, aggr):
        """Undirected message passing model.
        
        Args:
            phi : message passing function
                See the paper or the "Creating Message Passing Networks"
                for futher details.
        """

        super().__init__(aggr=aggr, flow="target_to_source")
        self.phi = phi
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        return self.phi(torch.cat((x_i, x_j), dim=-1))
    
    def update(self, aggr_out, x):
        return aggr_out


class GraphMPGNN(nn.Module):
    def __init__(self, phi, mlp, aggr):
        """MPGNN with SUM as the aggregation function.
        
        Instead of combining this part with the class `ClassicalMPGNN` 
        below, we seperate out for the possibility of other design of 
        mlp.

        Args:
            phi : message passing function
                See `MessagePassing` above.
            mlp : Multi-layer perceptrons
                Basically just a simple shallow fully connected linear
                model for transforming dimensions.
        """

        super().__init__()
        self.aggr = aggr
        self.gnn = MessagePassing(phi, aggr)
        self.mlp = mlp

    def forward(self, x, edge_index, batch):
        # Graph neural network.
        x = self.gnn(x, edge_index)

        # Graph aggregation.
        if self.aggr == "add":
            x = torch_geometric.nn.global_add_pool(x, batch)
        elif self.aggr == "mean":
            x = x = torch_geometric.nn.global_mean_pool(x, batch)

        # Shallow linear model.
        x = self.mlp(x)

        return x


class ClassicalMPGNN(GraphMPGNN):
    def __init__(
            self,
            gnn_in: int,
            gnn_out: int,
            gnn_hidden: int,
            gnn_layers: int,
            aggregation: str,
            mlp_hidden:int = 0,
            mlp_layers:int = 0,
            **kwargs
    ):
        """Classical model for benchmarking

        Arguments with prefix "gnn" are related to the message passing
        function `phi`, which is constructed with a classical MLP.

        Arguments with prefix "mlp" are related to the shallow linear
        model after graph aggregation, which is also constructed with a 
        classical MLP.

        The shallow linear model at the last step is designed for simply
        transforming the dimension of GNN outputs to 1 dimension, then
        the final 1 dimensional output represents the prediction of the 
        binary classification task, so the number of hidden neurons and
        hidden layers are default as 0.
        
        Args:
            gnn_in : int
                The input channel dimension of `phi` in MPGNN.
            gnn_out : int
                The output channel dimension of `phi` in MPGNN.
            gnn_hidden : int
                Number of hidden neurons of `phi` in MPGNN.
            gnn_layers : int
                Number of hidden layers of `phi` in MPGNN.
            mlp_hidden : int (default 0)
                Number of hidden neurons of the shallow linear model.
            mlp_layers : int (default 0)
                Number of hidden layers of the shallow linear model.
        """
        
        # See `GraphMPGNN` above.
        phi = module_model.ClassicalMLP(
            in_channel=gnn_in,
            out_channel=gnn_out,
            hidden_channel=gnn_hidden,
            num_layers=gnn_layers
            )
        mlp = module_model.ClassicalMLP(
            in_channel=gnn_out,
            out_channel=1,
            hidden_channel=mlp_hidden,
            num_layers=mlp_layers
            )

        super().__init__(phi, mlp, aggr=aggregation)

# %%
"""
### Quantum Complete Graph Neural Network (QCGNN)

The main structure of QCGNN is written in the `module_model.py` script, and the following codes focus on how data is encoded to the quantum circuit (corresponding to the arguement `ctrl_enc` in `module_model.QCGNN_IX` or `module_model.QCGNN_0`). In this paper, we test the encoding ansatz constructed through angle encoding.

Note that there are two encoding functions below (`pennylane_encoding` and `qiskit_encoding`), both are equivalent but run in different quantum devices.
- `pennylane_encoding`: for simulation using PennyLane (such as "default.qubit").
- `qiskit_encoding`: The multi-qubit gates in `pennylane_encoding` are decomposed to single-qubit and two-qubit gates. The decomposition method can be found at [Quantum Computation and Quantum Information](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview) section 4.3.
"""

# %%
def pennylane_encoding(ptc_input: torch.tensor, control_values: list[int], num_ir_qubits: int, num_nr_qubits: int):
    """Angle encoding ansatz for PennyLane

    Args:
        ptc_input : torch.tensor
            One particle information of the jet.
        control_values : list[int]
            Control values of the multi-controlled gates.
        num_ir_qubits : int
            Number of IR qubits.
        num_nr_qubits : int
            Number of NR qubits.
    """

    # Check input shape due to version `pennylane==0.31.0` above.
    if len(ptc_input.shape) > 1:
        # The shape of `ptc_input` is (batch, 3)
        theta, phi, omega = ptc_input[:,0], ptc_input[:,1], ptc_input[:,2]
    else:
        # The shape of `ptc_input` is (3,)
        theta, phi, omega = ptc_input[0], ptc_input[1], ptc_input[2]
    
    # Encode data on NR qubits.
    nr_wires = range(num_ir_qubits, num_ir_qubits + num_nr_qubits)
    for wires in nr_wires:
        # Add a Hadamard gate before the rotation gate.
        ctrl_H = qml.ctrl(qml.Hadamard, control=range(num_ir_qubits), control_values=control_values)
        ctrl_H(wires=wires)
        # Appl general rotation gate.
        ctrl_R = qml.ctrl(qml.Rot, control=range(num_ir_qubits), control_values=control_values)
        ctrl_R(theta=theta, phi=phi, omega=omega, wires=wires)


def qiskit_encoding(ptc_input: torch.tensor, control_values: list[int], num_ir_qubits: int, num_nr_qubits: int):
    """Angle encoding ansatz for IBMQ.

    Args:
        See `pennylane_encoding` above.
    """

    # Decomposition of multi-qubit gates needs working qubits.
    num_wk_qubits = num_ir_qubits - 1

    # Check input shape due to version `pennylane==0.31.0` above.
    if len(ptc_input.shape) > 1:
        # The shape of `ptc_input` is (batch, 3)
        theta, phi, omega = ptc_input[:,0], ptc_input[:,1], ptc_input[:,2]
    else:
        # The shape of `ptc_input` is (3,)
        theta, phi, omega = ptc_input[0], ptc_input[1], ptc_input[2]
    
    # Target wires to be encoded.
    nr_wires = range(num_ir_qubits + num_wk_qubits, 
                     num_ir_qubits + num_wk_qubits + num_nr_qubits)
    
    def control_condition_transform(control_values: list[int]):
        """Turn ctrl-0 to ctrl-1
        
        If control values == 0, use X-gate for transforming to 1.
        """

        for i in range(len(control_values)):
            # `i` also corresponds to the i-th qubit in IR.
            bit = control_values[i]
            if bit == 0:
                qml.PauliX(wires=i)

    def toffoli_tranformation(inverse: bool = False):
        """Decomposition of multi-contolled gates
        
        Use Toffoli transformation for decomposition of multi-controlled gates.

        Args:
            inverse : bool
                Whether to apply inverse transformation or not.
        """

        if (not inverse) and (num_ir_qubits > 1):
            wk_qubit_t = num_ir_qubits # target qubit, also first working qubit
            qml.Toffoli(wires=(0, 1, wk_qubit_t))
        
        if inverse:
            toffoli_range = reversed(range(num_wk_qubits-1))
        else:
            toffoli_range = range(num_wk_qubits-1)
        for i in toffoli_range:
            ir_qubit_c = 2 + i # control qubit
            wk_qubit_c = num_ir_qubits + i # control qubit
            wk_qubit_t = num_ir_qubits + i + 1 # target qubit
            qml.Toffoli(wires=(ir_qubit_c, wk_qubit_c, wk_qubit_t))

        if inverse and (num_ir_qubits > 1):
            wk_qubit_t = num_ir_qubits # target qubit, also first working qubit
            qml.Toffoli(wires=(0, 1, wk_qubit_t))

    # See "Quantum Computation and Quantum Information" section 4.3.
    for wires in nr_wires:
        control_condition_transform(control_values)
        toffoli_tranformation()
        
        # The last working qubit becomes the control qubit.
        wk_qubit_c = num_ir_qubits + num_wk_qubits - 1
        # ctrl_H: decomposed by H = i Rx(pi) Ry(pi/2) up to a global phase.
        qml.CRY(np.pi/2, wires=(wk_qubit_c, wires))
        qml.CRX(np.pi, wires=(wk_qubit_c, wires))
        # ctrl_R: Rot(phi, theta, omega) = Rz(omega) Ry(theta) Rz(phi).
        qml.CRZ(phi, wires=(wk_qubit_c, wires))
        qml.CRY(theta, wires=(wk_qubit_c, wires))
        qml.CRZ(omega, wires=(wk_qubit_c, wires))
        
        toffoli_tranformation(inverse=True)
        control_condition_transform(control_values)


class QuantumRotQCGNN(nn.Module):
    def __init__(
            self,
            num_ir_qubits: int,
            num_nr_qubits: int,
            num_layers: int,
            num_reupload: int,
            quantum_config: dict,
            aggregation: str = "SUM",
            return_meas: bool = False,
    ):
        """The quantum model that will be mainly used.

        This quantum model is based on the QCGNN_IX in `modeul_model.py`.

        Args:
            num_ir_qubits : int
                Number of qubits in IR.
            num_nr_qubits : int
                Number of qubits in NR.
            num_layers : int
                Number of VQC layers (per data re-uploading).
            num_reupload : int
                Number of data re-uploading.
            quantum_config : dict
                Arguments related in creating quantum device.
            aggregation : str (default "SUM")
                Aggregation method, either "SUM" or "MEAN".
            return_meas : bool (default False)
                Whether record the measurement output (for IBMQ).
        """
        
        super().__init__()
        self.aggregation = aggregation
        self.return_meas = return_meas
        
        # Detemine the encoding method due to the quantum device.
        qdevice = quantum_config["qdevice"]
        qbackend = quantum_config["qbackend"]

        # Real device (qiskit / IBMQ).
        if ("qiskit" in qdevice) or ("qiskit" in qbackend):
            ctrl_enc = lambda ptc_input, control_values: \
                qiskit_encoding(ptc_input, control_values, num_ir_qubits, num_nr_qubits)
        
        # Quantum simulators.
        else:
            ctrl_enc = lambda ptc_input, control_values: \
                pennylane_encoding(ptc_input, control_values, num_ir_qubits, num_nr_qubits)
        
        self.ctrl_enc = ctrl_enc
        
        # Constructing `phi` and `mlp` just like MPGNN.
        self.phi = QCGNN(
            num_ir_qubits=num_ir_qubits,
            num_nr_qubits=num_nr_qubits,
            num_layers=num_layers,
            num_reupload=num_reupload,
            ctrl_enc=ctrl_enc,
            aggregation=aggregation,
            return_meas=return_meas,
            **quantum_config
        )

        self.mlp = module_model.ClassicalMLP(
            in_channel=num_nr_qubits,
            out_channel=1,
            hidden_channel=0,
            num_layers=0
        )
    
    def forward(self, x):
        # Run on real devices (will return measurement outputs).
        if self.return_meas:
            x, meas = self.phi(x)
            x = self.mlp(x)
            return x, meas
        
        # Run on quantum simulators.
        else:
            x = self.phi(x)
            x = self.mlp(x)
            return x

# %%
"""
## Training Workflow
"""

# %%
"""
### Classical and quantum workflow

- The main workflow is written in function `execute`, consists of
  - Determine the name, id or other information for a training.
  - Monitoring the training procedure, either with `csv` only or `wandb`.
  - Record all the configurations (e.g. training, model, settings, .etc).
  - Two modes can be specified, either `mode="train"` or `mode="predict"`
    - `mode="train"` will create a new training, with results saving at `result_dir` in `./config.json`.
    - `mode="predict"` will load `ckpt` files from `ckpt_dir` in `./config.json`.
"""

# %%
"""
##### General `execute` function -> Main workflow
"""

# %%
def execute(
        model: nn.Module,
        general_config: dict,
        model_config: dict,
        data_config: dict,
        graph: bool,
        mode:str,
        suffix:str = ""
    ):
    """General training procedure.
    
    Workflow:
    1. Initialize the training monitor.
    2. Create data module and lightning model.
    3. Train or predict, depending on `mode`.
    4. Return training ID and summary.

    Args:
        model : nn.Module
            PyTorch format model.
        general_config : dict
            Dictionary of general settings.
        model_config : dict
            Used for recording model information.
        data_config : dict
            Used for generating data module.
        graph : bool
            Whether the training dataset is in graph structure or not.
        mode : str
            Either "train" or "predict" only.
        suffix : str
            The training suffix.
    
    Returns:
        1st argument : str
            The training ID (can be used to restore wandb runs).
        2nd argument : dict
            Summary of the training result.
    """
    
    # Seed everything.
    L.seed_everything(general_config["rnd_seed"])

    # Record the total training time.
    general_config["ctime"] = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    time_start = time.time()

    # Use suffix for the training if needed.
    if suffix != "":
        suffix = "_" + suffix

    # Additional model information (used for wandb filter).
    model_config["model_name"] = model.__class__.__name__
    model_config["group_rnd"] = f"{model_config['base_model']}_{model_config['model_suffix']}_{model_config['aggregation']}-{data_config['data_suffix']}"
    model_config["group_rnd_full"] = f"{model_config['model_name']}-{model_config['group_rnd']}"

    # Monitor (either CSVLogger or WandbLogger) configuration and setup.
    logger_config = {"resume":True}
    logger_config["project"] = general_config["wandb_project"]
    logger_config["group"] = f"{data_config['sig']}_{data_config['bkg']}"
    logger_config["name"] = f"{model_config['group_rnd']}-{general_config['time']}_{general_config['device']}{suffix}_{general_config['rnd_seed']}"
    logger_config["id"] = logger_config["name"]
    logger_config["save_dir"] = general_config["result_dir"]
    if use_wandb:
        try:
            account = general_config["wandb_account"]
            project = general_config["wandb_project"]
            run_id = logger_config["id"]
            run = api.run(f"{account}/{project}/{run_id}")
            if run.summary["epoch"] == general_config["max_epochs"]:
                _log(f"{run_id} has been successfully trained, ignored.\n")
                return run_id, run.summary
        except ValueError:
            _log(f"{run_id} does not exist, start training now.")
        except:
            _log(f"{run_id} exists but not trained successfully, retraining now.")
        logger = module_training.wandb_monitor(model, logger_config, general_config, model_config, data_config)   
    else:
        logger = module_training.default_monitor(logger_config, general_config, model_config, data_config)

    # Print out the training information.
    print("\n-------------------- Training information --------------------")
    print("| * general_config:", general_config, "\n")
    print("| * data_config:", data_config, "\n")
    print("| * model_config:", model_config, "\n")
    print("| * logger_config:", logger_config, "\n")
    print("--------------------------------------------------------------\n")
    
    # pytorch-lightning trainer.
    trainer = L.Trainer(
        logger=logger, 
        accelerator=general_config["accelerator"],
        max_epochs=general_config["max_epochs"],
        fast_dev_run=general_config["fast_dev_run"],
        log_every_n_steps=general_config["log_every_n_steps"],
        num_sanity_val_steps=0,
    )
    
    # pytorch-lightning data module.
    data_module = generate_datamodule(
        data_config=data_config, 
        data_ratio=general_config["data_ratio"],
        batch_size=general_config["batch_size"],
        graph=graph,
    )
    
    # pytorch-lightning model.
    litmodel = module_training.BinaryLitModel(
        model=model,
        lr=model_config["lr"],
        graph=graph,
    )
    
    # Training or prediction.
    if mode == "train":
        trainer.fit(litmodel, datamodule=data_module)
        train_summary = trainer.test(
            model=litmodel,
            dataloaders=data_module.train_dataloader()
        )[0] # The output is like [summary_object], so use [0] to get the item.
        test_summary = trainer.test(
            model=litmodel,
            dataloaders=data_module.test_dataloader()
        )[0] # The output is like [summary_object], so use [0] to get the item.
    
    elif mode == "predict":
        # The ckpt key helped for finding correct checkpoints file.
        ckpt_key = model_config["group_rnd"]
        # Train 16 particles only, and use the parameters to test other number 
        # of particles.
        if model_config['model_name'] == QuantumRotQCGNN.__name__:
            gnn_idx_qubits = model_config['gnn_idx_qubits']
            ckpt_key = ckpt_key.replace(f"qidx{gnn_idx_qubits}", "qidx4")
        # Get the correct checkpoints file path.
        ckpt_path = module_training.get_ckpt_path(ckpt_key, general_config["rnd_seed"])
        train_summary = trainer.test(
            model=litmodel,
            dataloaders=data_module.train_dataloader(),
            ckpt_path=ckpt_path
        )[0] # The output is like [summary_object], so use [0] to get the item.
        test_summary = trainer.test(
            model=litmodel,
            dataloaders=data_module.test_dataloader(),
            ckpt_path=ckpt_path
        )[0] # The output is like [summary_object], so use [0] to get the item.

    # Finish wandb monitoring if using WandbLogger.
    if use_wandb:
        wandb.finish()

    # Summary of training results.
    train_summary.update({"data_mode": "train"})
    test_summary.update({"data_mode": "test"})
    for _summary in [train_summary, test_summary]:
        _summary.update(general_config)
        _summary.update(data_config)
        _summary.update(model_config)
    summary = pd.DataFrame([train_summary, test_summary])
    
    # Calculating total training time.
    time_end = time.time()
    _log(f"Time = {(time_end - time_start) / 60} minutes")
    print("\n", 100 * "@", "\n")
    
    return logger_config["id"], summary

# %%
"""
##### Specific execution functions for MPGNN and QCGNN
"""

# %%
def execute_classical(
        general_config: dict,
        data_config: dict,
        go: int,
        gh: int,
        gl: int,
        lr: float,
        mode: str
    ):
    """Default setup for classical execution.
    
    Create with `ClassicalMPGNN` model.

    Args:
        general_config : dict
            Dictionary of general settings.
        data_config : dict
            Data configuration for generating data module.
        go : int
            Dimension of GNN output channel.
        gh : int
            Number of hidden neurons in GNN.
        gl : int
            Number of hidden layers in GNN
        lr : float
            Learning rate.
        mode : str
            Either "train" for "predict".
    """
    
    # Suffix used for wandb filter and training result.
    model_suffix = f"go{go}_gh{gh}_gl{gl}_mh0_ml0"

    # Determine the aggregation function
    if general_config["aggregation"] == "SUM":
        aggregation = "add"
    elif general_config["aggregation"] == "MEAN":
        aggregation = "mean"
    
    # Configurations for constructing MPGNN.
    model_config = {
        "gnn_in": 6, "gnn_out": go, "gnn_hidden": gh, "gnn_layers": gl,
        "mlp_hidden": 0, "mlp_layers": 0, "aggregation": aggregation,
        "lr": lr, "model_suffix": model_suffix,
        "base_model": module_model.ClassicalMLP.__name__,
    }
    model = ClassicalMPGNN(**model_config)

    # Execute training for classical models.
    general_config["accelerator"] = json_config["accelerator"]
    run_id, summary = execute(
        model=model,
        general_config=general_config,
        model_config=model_config,
        data_config=data_config,
        graph=True,
        mode=mode,
        suffix=general_config["suffix"],
    )

    # Check whether the model is trained successfully above threshold.
    test_acc = summary["test_acc_epoch"]
    acc_threshold = general_config["retrain_threshold"]
    try:
        below_threshold = (test_acc < acc_threshold).any()
    except:
        below_threshold = (test_acc < acc_threshold)
    
    # Too small model might not trainable, we retrain with random seed += 10.
    if (mode == "train") and general_config["retrain"] and below_threshold:
        # Delete the failed training on Wandb.
        if use_wandb:
            account = general_config["wandb_account"]
            project = general_config["wandb_project"]
            run = api.run(f"{account}/{project}/{run_id}")
            run.delete()
        
        # Increase random seed by 10, since we train 10 different seeds.
        old_rnd_ssed = general_config["rnd_seed"]
        new_rnd_seed = old_rnd_ssed + general_config["retrain_cycle"]
        general_config["rnd_seed"] = new_rnd_seed
        _log(f"Reinitialize model with new random seed = {new_rnd_seed}")

        # Retrain a new round.
        run_id, summary = execute_classical(general_config, data_config,
                                            go, gh, gl, lr, mode)
    
    return run_id, summary

def execute_quantum(
        general_config: dict,
        data_config: dict,
        qnn: int,
        gl: int,
        gr: int,
        lr: float,
        mode: str
    ):
    """Default setup for quantum execution.
    
    Create with `QuantumRotQCGNN` model.

    Args:
        general_config : dict
            Dictionary of general settings.
        data_config : dict
            Data configuration for generating data module.
        qnn : int
            Number of NR qubits.
        gl : int
            Number of strongly entangling layers of a VQC.
        gr : int
            Number of data reuploading times.
        lr : float
            Learning rate.
        mode : str
            Either "train" for "predict".
    """

    # Number of IR qubits
    qidx = int(np.ceil(np.log2(data_config["max_num_ptcs"])))

    # Suffix used for wandb filter and training result.
    model_suffix = f"qidx{qidx}_qnn{qnn}_gl{gl}_gr{gr}"

    # Configurations for constructing QCGNN.
    model_config = {
        "gnn_idx_qubits": qidx,
        "gnn_nn_qubits": qnn,
        "gnn_layers": gl,
        "gnn_reupload": gr,
        "lr": lr,
        "model_suffix": model_suffix,
        "base_model": QCGNN.__name__,
        "aggregation": general_config["aggregation"],
    }
    model = QuantumRotQCGNN(
        num_ir_qubits=qidx,
        num_nr_qubits=qnn,
        num_layers=gl,
        num_reupload=gr,
        quantum_config=quantum_config,
        aggregation=general_config["aggregation"],
    )
    
    # Execute training for quantum models.
    general_config["accelerator"] = "cpu"
    run_id, summary = execute(
        model=model,
        general_config=general_config,
        model_config=model_config,
        data_config=data_config,
        graph=False,
        mode=mode,
        suffix=general_config["suffix"],
    )
    
    return run_id, summary

# %%
"""
## Training and Prediction

Jet dataset using 3 features $p_T$, $\Delta\eta$, $\Delta\phi$.
- 2-prong v.s. 1-prong
    - Signal: VzToZhToVevebb
    - Background: VzToQCD
- 3-prong v.s. 1-prong
    - Signal: VzToTt
    - Background: VzToQCD
"""

# %%
"""
### Data Cell
"""

# %%
data_config_list = [
    # 2-prong v.s. 1-prong.
    generate_data_config(
        sig="VzToZhToVevebb",
        bkg="VzToQCD",
        abbrev="BB-QCD",
        general_config=general_config,
    ),
    
    # 3-prong v.s. 1-prong.
    generate_data_config(
        sig="VzToTt",
        bkg="VzToQCD",
        abbrev="TT-QCD",
        general_config=general_config,
    ),
]

# %%
"""
### Training Cell
"""

# %%
# Uncomment the model you want to train.
for rnd_seed, data_config in product(range(10), data_config_list):
    general_config["rnd_seed"] = rnd_seed
    
    # # Classical MPGNN with hidden neurons {3, 5, 7} and 2 layers.
    # for g_dim in [3, 5, 7]:
    #     execute_classical(general_config, data_config, go=g_dim, gh=g_dim, gl=2, lr=1E-3, mode="train")

    # # Best classical MPGNN with hidden neurons 1024 and 4 layers.
    # execute_classical(general_config, data_config, go=1024, gh=1024, gl=4, lr=1E-3, mode="train")

    # # Quantum QCGNN with NR qubits = reuploads = {3, 5, 7}.
    # for q in [3, 5, 7]:
    #     execute_quantum(general_config, data_config, qnn=q, gl=1, gr=q, lr=1E-3, mode="train")