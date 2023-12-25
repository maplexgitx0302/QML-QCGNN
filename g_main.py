# %%
"""
### Packages and Configurations
"""

# %%
# basic packages
import pandas as pd
import os, time, json, argparse
from itertools import product

# modules
import module_data
import module_model
import module_training

# qml tools
import pennylane as qml
from pennylane import numpy as np

# ml tools
import torch
import torch.nn as nn
import lightning as L
import torch_geometric.nn as geomodule_model
from torch_geometric.nn import MessagePassing

# faster calculation on GPU but less precision
torch.set_float32_matmul_precision("medium")

# %%
"""
### Manual Settings
"""

# %%
with open("config.json", "r") as json_file:
    # read JSON config file
    json_config = json.load(json_file)

    # directory to store result
    os.makedirs(json_config["result_dir"], exist_ok=True)
    os.makedirs(json_config["ckpt_dir"], exist_ok=True)

    # default configurations
    config = json_config["config"]

if __name__ == "__main__":
    # general configuration and setup
    config["ctime"]  = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    config["time"]   = input("Specify a datetime or leave empty (default current time): ") or config["ctime"]
    config["device"] = input("Enter the computing device (4090, slurm, node, etc.)    : ")
    config["suffix"] = input("Suffix (the format looks like 'MODEL_SUFFIX_DATE')      : ")
    if config["device"] == "slurm":
        # pass arguments when running through slurm
        parser = argparse.ArgumentParser(description='argparse for slurm')
        parser.add_argument('--rnd_seed', type=int, help='random seed')
        parse_args = parser.parse_args()
        config["rnd_seed"] = parse_args.rnd_seed

    # whether using wandb to monitor the training procedure
    use_wandb = json_config["use_wandb"]
    if use_wandb:
        import wandb
        api = wandb.Api()

    # quantum config
    quantum_config = json_config[config["quantum_config"]]

    # manual settings
    if config["quick_test"] == True:
        config["max_epochs"]   = 1
        config["num_bin_data"] = 10

# %%
"""
### Classical Message Passing Graph Neural Network (MPGNN)
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

class GraphMPGNN(nn.Module):
    def __init__(self, phi, mlp):
        super().__init__()
        self.gnn = MessagePassing(phi)
        self.mlp = mlp
    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = geomodule_model.global_add_pool(x, batch)
        x = self.mlp(x)
        return x
    
class ClassicalMPGNN(GraphMPGNN):
    def __init__(self, gnn_in, gnn_out, gnn_hidden, gnn_layers, mlp_hidden=0, mlp_layers=0, **kwargs):
        phi = module_model.ClassicalMLP(in_channel=gnn_in, out_channel=gnn_out, hidden_channel=gnn_hidden, num_layers=gnn_layers)
        mlp = module_model.ClassicalMLP(in_channel=gnn_out, out_channel=1, hidden_channel=mlp_hidden, num_layers=mlp_layers)
        super().__init__(phi, mlp)

# %%
"""
### Quantum Complete Graph Neural Network (QCGNN)
"""

# %%
# rotation encoding on pennylane simulator
def pennylane_encoding(_input, control_values, num_ir_qubits, num_nr_qubits):
    for i in range(num_nr_qubits):
        ctrl_H = qml.ctrl(qml.Hadamard, control=range(num_ir_qubits), control_values=control_values)
        ctrl_H(wires=num_ir_qubits+i)
        ctrl_R = qml.ctrl(qml.Rot, control=range(num_ir_qubits), control_values=control_values)
        # batch data
        if len(_input.shape) > 1:
            ctrl_R(theta=_input[:,0], phi=_input[:,1], omega=_input[:,2], wires=num_ir_qubits+i)
        # single data
        else:
            ctrl_R(theta=_input[0], phi=_input[1], omega=_input[2], wires=num_ir_qubits+i)

# rotation encoding on qiskit
def qiskit_encoding(_input, control_values, num_ir_qubits, num_nr_qubits):
    num_wk_qubits = num_ir_qubits - 1
    # batch data
    if len(_input.shape) > 1:
        theta, phi, omega = _input[:,0], _input[:,1], _input[:,2]
    # single data
    else:
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

class QuantumRotQCGNN(nn.Module):
    def __init__(self, num_ir_qubits, num_nr_qubits, num_layers, num_reupload, quantum_config):
        super().__init__()
        # constructing QCGNN like a MPGNN
        if "qiskit" in quantum_config["qdevice"]:
            ctrl_enc = lambda _input, control_values: qiskit_encoding(_input, control_values, num_ir_qubits, num_nr_qubits)
        else:
            ctrl_enc = lambda _input, control_values: pennylane_encoding(_input, control_values, num_ir_qubits, num_nr_qubits)
        self.phi = module_model.QCGNN(num_ir_qubits, num_nr_qubits, num_layers, num_reupload, ctrl_enc=ctrl_enc, **quantum_config)
        self.mlp = module_model.ClassicalMLP(in_channel=num_nr_qubits, out_channel=1, hidden_channel=0, num_layers=0)
    
    def forward(self, x):
        # inputs should be 1-dim for each data, otherwise it would be confused with batch shape
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = self.phi(x)
        x = self.mlp(x)
        return x

# %%
"""
### Workflow
"""

# %%
def get_ckpt(ckpt_key):
    ckpt_dir = json_config["ckpt_dir"]
    for f in os.listdir(ckpt_dir):
        if ckpt_key in f and int(f[-1]) == config["rnd_seed"]:
            ckpt_path = os.path.join(ckpt_dir, f, "checkpoints")
            ckpt_path = os.path.join(ckpt_path, os.listdir(ckpt_path)[0])
            break
    else:
        raise ValueError(f"# ModelLog: ckpt NOT found in {ckpt_dir} -> {ckpt_key}")
    print(f"# ModelLog: ckpt found at {ckpt_path}")
    return ckpt_path

def execute(model, model_config, data_module, data_config, graph, mode, suffix=""):
    time_start = time.time()
    if suffix != "":
        suffix = "_" + suffix

    # additional model information
    model_config["model_name"] = model.__class__.__name__
    model_config["group_rnd"]  = f"{model.__class__.__name__}_{model_config['model_suffix']} | {data_config['data_suffix']}"

    # use wandb monitoring if needed
    logger_config = {}
    logger_config["project"]    = json_config["project"]
    logger_config["group"]      = f"{data_config['sig']}_{data_config['bkg']}"
    if "qiskit" in quantum_config["qdevice"]:
        logger_config["name"] = f"{model_config['group_rnd']} | {config['time']}_{quantum_config['qdevice']}{suffix}_{config['rnd_seed']}"
    else:
        logger_config["name"] = f"{model_config['group_rnd']} | {config['time']}_{config['device']}{suffix}_{config['rnd_seed']}"
    logger_config["id"]       = logger_config["name"]
    logger_config["save_dir"] = json_config["result_dir"]
    if use_wandb:
        logger = module_training.wandb_monitor(model, logger_config, config, model_config, data_config)
    else:
        logger = module_training.default_monitor(logger_config, config, model_config, data_config)

    # training information
    print("\n-------------------- Training information --------------------")
    print("| * config:", config)
    print("| * data_config:", data_config)
    print("| * model_config:", model_config)
    print("| * logger_config:", logger_config)
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
    
    if mode == "train":
        trainer.fit(litmodel, datamodule=data_module)
        train_summary = trainer.test(litmodel, dataloaders=data_module.train_dataloader())[0]
        test_summary  = trainer.test(litmodel, dataloaders=data_module.test_dataloader())[0]
    elif mode == "predict":
        ckpt_key = f"{model_config['model_name']}_{model_config['model_suffix']} | {data_config['abbrev']}_cut{data_config['cut']}"
        if model_config['model_name'] == QuantumRotQCGNN.__name__:
            ckpt_key = ckpt_key.replace(f"qidx{model_config['gnn_idx_qubits']}", "qidx3")
        ckpt_path = get_ckpt(ckpt_key)
        train_summary = trainer.test(litmodel, dataloaders=data_module.train_dataloader(), ckpt_path=ckpt_path)[0]
        test_summary  = trainer.test(litmodel, dataloaders=data_module.test_dataloader(), ckpt_path=ckpt_path)[0]

    # finish wandb monitoring
    if use_wandb:
        wandb.finish()

    # summary
    train_summary.update({"data_mode":"train"})
    test_summary.update({"data_mode":"test"})
    for _summary in [train_summary,test_summary]:
        _summary.update(config)
        _summary.update(data_config)
        _summary.update(model_config)
    summary = pd.DataFrame([train_summary, test_summary])

    time_end = time.time()
    print(f"# ModelLog: Time = {(time_end - time_start) / 60} minutes")
    print("\n", 100 * "@", "\n")
    return logger_config["id"], summary

# %%
def generate_datamodule(data_config, graph):
    sig_fatjet_events = module_data.FatJetEvents(channel=data_config["sig"], cut_pt=data_config["cut"], subjet_radius=data_config["subjet_radius"], num_pt_ptcs=data_config["num_pt_ptcs"])
    bkg_fatjet_events = module_data.FatJetEvents(channel=data_config["bkg"], cut_pt=data_config["cut"], subjet_radius=data_config["subjet_radius"], num_pt_ptcs=data_config["num_pt_ptcs"])
    sig_events  = sig_fatjet_events.generate_uniform_pt_events(bin=data_config["bin"], num_bin_data=data_config["num_bin_data"])
    bkg_events  = bkg_fatjet_events.generate_uniform_pt_events(bin=data_config["bin"], num_bin_data=data_config["num_bin_data"])
    data_suffix = f"{data_config['abbrev']}_cut{data_config['cut']}_ptc{data_config['num_pt_ptcs']}_bin{data_config['bin']}-{data_config['num_bin_data']}_R{data_config['subjet_radius']}"
    data_config["data_suffix"] = data_suffix
    return module_data.JetDataModule(sig_events, bkg_events, data_ratio=config["data_ratio"], batch_size=config["batch_size"], graph=graph)

def execute_classical(data_config, go, gh, gl, lr, mode):
    '''
        go -> dimension of gnn output
        gh -> dimension of gnn hidden neurons
        gl -> number of gnn hidden layers
    '''
    data_module  = generate_datamodule(data_config, graph=True)
    model_suffix = f"go{go}_gh{gh}_gl{gl}_mh0_ml0"
    model_config = {"gnn_in":6, "gnn_out":go, "gnn_hidden":gh, "gnn_layers":gl, "mlp_hidden":0, "mlp_layers":0, "lr":lr, "model_suffix":model_suffix}
    model        = ClassicalMPGNN(**model_config)
    run_id, summary = execute(model, model_config, data_module, data_config, graph=True, mode=mode, suffix=config["suffix"])
    if mode == "train":
        while (summary["test_acc_epoch"] < json_config["retrain_threshold"]).any() and json_config["retrain"]:
            if use_wandb:
                run = api.run(f"{json_config['wandb_id']}/{json_config['project']}/{run_id}")
                run.delete()
            config["rnd_seed"] += json_config["retrain_cycle"]
            L.seed_everything(config["rnd_seed"])
            print(f"\n # ModelLog: Reinitialize model with new rnd_seed = {config['rnd_seed']}\n")
            model = ClassicalMPGNN(**model_config)
            run_id, summary = execute(model, model_config, data_module, data_config, graph=True, mode=mode, suffix=config["suffix"])
    return run_id, summary

def execute_quantum(data_config, qnn, gl, gr, lr, mode):
    '''
        qnn -> number of NR qubits
        gl  -> number of strongly entangling layers
        gr  -> number of data reuploading
    '''
    data_module     = generate_datamodule(data_config, graph=False)
    qidx            = int(np.ceil(np.log2(data_config["num_pt_ptcs"])))
    model_suffix    = f"qidx{qidx}_qnn{qnn}_gl{gl}_gr{gr}"
    model_config    = {"gnn_idx_qubits":qidx, "gnn_nn_qubits":qnn, "gnn_layers":gl, "gnn_reupload":gr, "lr":lr, "model_suffix":model_suffix}
    model           = QuantumRotQCGNN(num_ir_qubits=qidx, num_nr_qubits=qnn, num_layers=gl, num_reupload=gr, quantum_config=quantum_config)
    run_id, summary = execute(model, model_config, data_module, data_config, graph=False, mode=mode, suffix=config["suffix"])
    return run_id, summary

# %%
"""
### Training
"""

# %%
# data_configs = [
#     {"sig": "VzToZhToVevebb", "bkg": "VzToQCD", "abbrev":"BB-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":8},
#     {"sig": "VzToTt", "bkg": "VzToQCD", "abbrev":"TT-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":8},
# ]

# # training
# for data_config, rnd_seed in product(data_configs, range(10)):
#     config["rnd_seed"] = rnd_seed
#     L.seed_everything(config["rnd_seed"])
    
#     # # classical
#     # for g_dim in [3,6,9]:
#     #     execute_classical(data_config, go=g_dim, gh=g_dim, gl=2, lr=1E-3, mode="train")

#     # # best classical
#     # execute_classical(data_config, go=1024, gh=1024, gl=4, lr=1E-3, mode="train")

#     # quantum
#     for q in [3,6,9]:
#         execute_quantum(data_config, qnn=q, gl=1, gr=q, lr=1E-3, mode="train")

# %%
"""
### Prediction
"""

# %%
# os.makedirs("./csv", exist_ok=True)

# data_configs = [
#     {"sig": "VzToZhToVevebb", "bkg": "VzToQCD", "abbrev":"BB-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":None},
#     {"sig": "VzToTt", "bkg": "VzToQCD", "abbrev":"TT-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":None},
# ]

# c_df, b_df, q_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
# num_ptcs_range = range(2, 16+1, 2)
# for rnd_seed, num_pt_ptcs in product(range(3), num_ptcs_range):
#     for data_config in data_configs:
#         data_config["num_pt_ptcs"] = num_pt_ptcs
#         config["rnd_seed"] = rnd_seed
#         L.seed_everything(config["rnd_seed"])

#         # # classical
#         # for g_dim in [3,6,9]:
#         #     _, summary = execute_classical(data_config, go=g_dim, gh=g_dim, gl=2, lr=1E-3, mode="predict")
#         #     c_df = pd.concat((c_df, summary))
#         #     c_df.to_csv(f"csv/classical-{config['num_bin_data']}_{rnd_seed}.csv", index=False)

#         # # best classical
#         # _, summary = execute_classical(data_config, go=1024, gh=1024, gl=4, lr=1E-3, mode="predict")
#         # b_df = pd.concat((b_df, summary))
#         # b_df.to_csv(f"csv/classical-{config['num_bin_data']}_best.csv", index=False)

#         # quantum
#         for q in [3,6,9]:
#             _, summary = execute_quantum(data_config, qnn=q, gl=1, gr=q, lr=1E-3, mode="predict")
#             q_df = pd.concat((q_df, summary))
#         q_df.to_csv(f"csv/qnn{q}_gl1_gr{q}_ptc({num_ptcs_range[0]},{num_ptcs_range[-1]})-{config['num_bin_data']}_{rnd_seed}.csv", index=False)

# %%
"""
### Qiskit device
"""

# %%
# data_configs = [
#     {"sig": "VzToZhToVevebb", "bkg": "VzToQCD", "abbrev":"BB-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":8},
#     {"sig": "VzToTt", "bkg": "VzToQCD", "abbrev":"TT-QCD", "cut": (800, 1000), "bin":10, "subjet_radius":0, "num_bin_data":config["num_bin_data"], "num_pt_ptcs":8},
# ]

# q_df = pd.DataFrame()
# for rnd_seed in range(3):
#     for data_config in data_configs:
#         config["rnd_seed"] = rnd_seed
#         L.seed_everything(config["rnd_seed"])

#         # quantum
#         q = 3
#         config["batch_size"] = 300
#         _, summary = execute_quantum(data_config, qnn=q, gl=1, gr=q, lr=1E-3, mode="predict")
#         q_df = pd.concat((q_df, summary))
#         q_df.to_csv(f"csv/{quantum_config['qbackend']}_{q}_gl1_gr{q}_ptc{config['num_bin_data']}-{config['num_pt_ptcs']}_{rnd_seed}.csv", index=False)