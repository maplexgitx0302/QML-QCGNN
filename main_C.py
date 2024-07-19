# %%
import argparse
import sys
import time
import yaml

import awkward as ak
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import pennylane as qml
import torch
import torch.nn as nn
import wandb

from source.data.datamodule import JetTorchDataModule, JetGraphDataModule
from source.data.opendata import JetNetEvents, TopQuarkEvents
from source.models.mpgnn import ClassicalMPGNN
from source.models.part import ParticleTransformer
from source.models.pfn import ParticleFlowNetwork
from source.models.pnet import ParticleNet
from source.models.qcgnn import QuantumRotQCGNN
from source.training.litmodel import TorchLightningModule, GraphLightningModel
from source.training.loggers import csv_logger, wandb_logger
from source.training.result import plot_metrics

# %%
if 'ipykernel' in sys.argv[0]:
    # Jupyter notebook (.ipynb)
    # dataset = 'TopQCD'
    dataset = 'JetNet'
    random_seed = 42
    parser_args = None

else:
    # Python script (.py)
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset (default: None)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--num_train', type=int, default=None, help='Number of training data (default: None)')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix (default: None)')
    parser_args = parser.parse_args()

    dataset = parser_args.dataset
    random_seed = parser_args.random_seed

with open(f"configs/config.yaml", 'r') as file:
    
    # Configuration of training.
    config = yaml.safe_load(file)
    config['date'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config['dataset'] = dataset

    # Whether change default number of data
    if parser_args is not None:
        if parser_args.num_train is not None:
            config['Data']['num_train'] = parser_args.num_train
            config['Data']['num_val'] = int(0.1 * parser_args.num_train)
            config['Data']['num_test'] = int(0.1 * parser_args.num_train)
        if parser_args.suffix is not None:
            config['Settings']['suffix'] = parser_args.suffix

    # Determine the dimension of the score.
    if config[dataset]['num_classes'] <= 2:
        # Will use `BCELossWithLogits`
        score_dim = 1
    else:
        # Will use `CrossEntropyLoss`
        score_dim = config[dataset]['num_classes']

# %%
def create_data_module(graph: bool, pi_scale: bool = False) -> L.LightningDataModule:
    """Randomly create a data module."""
    
    # Read jet data (reading is not affected by random seed).
    num_train = config['Data']['num_train']
    num_valid = config['Data']['num_valid']
    num_test = config['Data']['num_test']
    num_events = num_train + num_valid + num_test

    # Dataset settings.
    dataset_config = {}
    dataset_config.update(config['Data'])
    dataset_config.update(config[dataset])

    # JetNet dataset (multi-class classification).
    if dataset == 'JetNet':
        channels = ['q', 'g', 't', 'w', 'z']
        events = [JetNetEvents(channel=channel, **dataset_config) for channel in channels]
        events = [_events.generate_events(num_events) for _events in events]

    # Top quark tagging dataset (background light quark QCD).
    elif dataset == 'TopQCD':
        events = []
        for y, channel in enumerate(['Top', 'QCD']):
            events_train = TopQuarkEvents(mode='train', is_signal_new=y, **dataset_config).generate_events(num_train)
            events_valid = TopQuarkEvents(mode='valid', is_signal_new=y, **dataset_config).generate_events(num_valid)
            events_test  = TopQuarkEvents(mode='test',  is_signal_new=y, **dataset_config).generate_events(num_test)
            events.append(ak.concatenate([events_train, events_valid, events_test], axis=0))
    
    # Turn into data module (for lightning training module).
    if graph:
        data_module = JetGraphDataModule(events, pi_scale=pi_scale, **dataset_config)
    else:
        data_module = JetTorchDataModule(events, **dataset_config)

    return data_module


def create_training_info(model: nn.Module, model_description: str, model_hparams: dict, lr: float) -> dict:
    """Create a training information dictionary that will be recorded."""

    # Data information.
    data_description = f"{dataset}_P{config['Data']['max_num_ptcs']}_N{config['Data']['num_train']}"

    # The description used for grouping different result of random seeds.
    group_rnd = '-'.join([model.__class__.__name__, model_description, data_description])
    
    # Name for this particular training (including random seed).
    name = group_rnd
    if config['Settings']['suffix'] != '':
        name += '-' + config['Settings']['suffix']
    name += '-' + str(random_seed)

    # Hyperparameters and configurations that will be recorded.
    training_info = config.copy()
    training_info.update(model_hparams)
    training_info.update({
        'lr': lr,
        'name': name,
        'date': config['date'],
        'model': model.__class__.__name__,
        'group_rnd': group_rnd,
        'random_seed': random_seed,
        'data_description': data_description,
        'model_description': model_description,
    })

    return training_info

    
def create_lightning_model(model: nn.Module, graph: bool, lr: float) -> L.LightningModule:
    """Create a lightning model for trainer."""

    # Optimizer.
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)

    # Create lightning model depends on graph or not.
    print_log = config['Settings']['print_log']

    # Graph is for PFN.
    if graph:
        return GraphLightningModel(model, optimizer=optimizer, score_dim=score_dim, print_log=print_log)
    else:
        return TorchLightningModule(model, optimizer=optimizer, score_dim=score_dim, print_log=print_log)


def create_trainer(model: nn.Module, training_info: dict, accelerator: str) -> L.Trainer:
    """Create lightning trainer for training."""

    # Create logger for monitoring the training.
    if config['Settings']['use_wandb']:
        wandb.login()
        logger = wandb_logger(training_info)
        logger.watch(model)
        loggers = [logger, csv_logger(training_info)]
    else:
        loggers = csv_logger(training_info)
    
    # Return the lightning trainer.
    return L.Trainer(
        logger=loggers,
        accelerator=accelerator,
        max_epochs=config['Train']['max_epochs'],
        log_every_n_steps=config['Train']['log_every_n_steps'],
        num_sanity_val_steps=config['Train']['num_sanity_val_steps'],
        callbacks=[ModelCheckpoint(
            monitor=config['Train']['ckpt_monitor'],
            mode=config['Train']['ckpt_mode'],
            save_top_k=config['Train']['ckpt_top_k'],
            save_last=True,
            filename='{epoch}-{valid_auc:.3f}-{valid_accuracy:.3f}',
        )],
    )

# %%
def train(
        model: nn.Module, model_description: str, model_hparams: dict,
        accelerator: str, lr: float, graph: bool, pi_scale: bool = False
    ):
    
    # Fix all random stuff.
    L.seed_everything(random_seed)

    # Traditional training procedure.
    data_module = create_data_module(graph=graph, pi_scale=pi_scale)
    lightning_model = create_lightning_model(model=model, graph=graph, lr=lr)
    training_info = create_training_info(model, model_description, model_hparams, lr=lr)
    trainer = create_trainer(model, training_info, accelerator)
    
    # Training and validation.
    if eval(config['Settings']['mode'][0]):
        if eval(config['Settings']['mode'][1]):
            trainer.fit(lightning_model, datamodule=data_module)
        else:
            trainer.fit(lightning_model, train_dataloaders=data_module.train_dataloader())

    # Testing.
    if eval(config['Settings']['mode'][2]):
        trainer.test(lightning_model, datamodule=data_module, ckpt_path='best')

    # Finish wandb if used.
    if config['Settings']['use_wandb']:
        wandb.finish()

    return training_info['name']

def train_quantum(model_class: nn.Module, model_hparams: dict, pi_scale: bool, lr: float):

    model = model_class(score_dim=score_dim, **model_hparams)

    num_ir_qubits = model_hparams['num_ir_qubits']
    num_nr_qubits = model_hparams['num_nr_qubits']
    num_layers = model_hparams['num_layers']
    num_reupload = model_hparams['num_reupload']
    dropout = model_hparams['dropout']
    model_description = f"nI{num_ir_qubits}_nQ{num_nr_qubits}_L{num_layers}_R{num_reupload}_D{dropout:.2f}"

    accelerator = 'cpu'
    name = train(model, model_description, model_hparams, accelerator, lr=lr, graph=False, pi_scale=pi_scale)

    return name

def train_mpgnn(model_hparams: dict, lr: float):
    
    model = ClassicalMPGNN(score_dim=score_dim, **model_hparams)

    phi_out = model_hparams['phi_out']
    phi_hidden = model_hparams['phi_hidden']
    phi_layers = model_hparams['phi_layers']
    dropout = model_hparams['dropout']
    model_description = f"O{phi_out}_H{phi_hidden}_L{phi_layers}_D{dropout:.2f}"

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    name = train(model, model_description, model_hparams, accelerator, lr=lr, graph=True)
    
    return name

def train_benchmark(model_class: nn.Module, lr: float):
    with open('configs/benchmark.yaml', 'r') as file:
        hparams = yaml.safe_load(file)[model_class.__name__]
        model_description = ''

    model = model_class(score_dim=score_dim, parameters=hparams)
    
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    if model_class == ParticleFlowNetwork:
        name = train(model, model_description, hparams, accelerator, lr=lr, graph=True)
    else:
        name = train(model, model_description, hparams, accelerator, lr=lr, graph=False)
    
    return name

# %%
# # QCGNN
# for Q in [3, 6]:
#     print(f"\n* Train QCGNN n_Q = {Q}.\n")
#     qcgnn_hparams = {'num_ir_qubits': 4, 'num_nr_qubits': Q, 'num_layers': Q // 3, 'num_reupload': 2, 'dropout': 0.0, 'vqc_ansatz': qml.StronglyEntanglingLayers}
#     name = train_quantum(model_class=QuantumRotQCGNN, model_hparams=qcgnn_hparams, pi_scale=True, lr=1e-3)

# %%
# MPGNN
for phi_dim in [3, 6, 64]:
    pfn_hparams = {'phi_in': 3 + 3, 'phi_out': phi_dim, 'phi_layers': 2, 'phi_hidden': phi_dim, 'dropout': 0.0}
    name = train_mpgnn(model_hparams=pfn_hparams, lr=1e-3)

# Particle Flow Network.
print(f"\n* Train Particle Flow Network.\n")
name = train_benchmark(model_class=ParticleFlowNetwork, lr=1e-3)

# Particle Transformer.
print(f"\n* Train Particle Transformer.\n")
name = train_benchmark(model_class=ParticleTransformer, lr=1e-3)

# Particle Net.
print(f"\n* Train Particle Net.\n")
name = train_benchmark(model_class=ParticleNet, lr=1e-3)