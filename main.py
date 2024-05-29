# %%
import argparse
import sys
import time
import yaml

import awkward as ak
import lightning as L
import torch
import torch.nn as nn
import wandb

from source.data.datamodule import JetTorchDataModule, JetGraphDataModule
from source.data.opendata import JetNetEvents, TopQuarkEvents
from source.models.qcgnn import QuantumRotQCGNN, HybridQCGNN
from source.models.mpgnn import ClassicalMPGNN
from source.models.part import ParticleTransformer
from source.models.pnet import ParticleNet
from source.training.litmodel import TorchLightningModule, GraphLightningModel
from source.training.loggers import csv_logger, wandb_logger
from source.training.result import plot_metrics

# %%
if 'ipykernel' in sys.argv[0]:
    # Jupyter notebook (.ipynb)
    # dataset = 'TopQCD'
    dataset = 'JetNet'
    random_seed = 42

else:
    # Python script (.py)
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset (default: None)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed (default: 42)')
    parser_args = parser.parse_args()

    dataset = parser_args.dataset
    random_seed = parser_args.random_seed

with open(f"configs/config.yaml", 'r') as file:
    
    # Configuration of training.
    config = yaml.safe_load(file)
    config['date'] = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config['dataset'] = dataset

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


def create_training_info(model: nn.Module, model_description: str, model_hparams: dict) -> dict:
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
        'name': name,
        'date': config['date'],
        'model': model.__class__.__name__,
        'group_rnd': group_rnd,
        'random_seed': random_seed,
        'data_description': data_description,
        'model_description': model_description,
    })

    return training_info

    
def create_lightning_model(model: nn.Module, graph: bool) -> L.LightningModule:
    """Create a lightning model for trainer."""

    # Optimizer.
    lr = eval(config['Train']['lr'])
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)

    # Create lightning model depends on graph or not.
    print_log = config['Settings']['print_log']

    # Graph is for MPGNN.
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
        
    )

# %%
def train(
        model: nn.Module, model_description: str, model_hparams: dict,
        accelerator: str, graph: bool, pi_scale: bool = False,
    ):
    
    # Fix all random stuff.
    L.seed_everything(random_seed)

    # Traditional training procedure.
    data_module = create_data_module(graph=graph, pi_scale=pi_scale)
    lightning_model = create_lightning_model(model=model, graph=graph)
    training_info = create_training_info(model, model_description, model_hparams)
    trainer = create_trainer(model, training_info, accelerator)
    
    # Training and validation.
    if eval(config['Settings']['mode'][0]):
        if eval(config['Settings']['mode'][1]):
            trainer.fit(lightning_model, datamodule=data_module)
        else:
            trainer.fit(lightning_model, train_dataloaders=data_module.train_dataloader())

    # Testing.
    if eval(config['Settings']['mode'][2]):
        trainer.test(lightning_model, datamodule=data_module)

    # Finish wandb if used.
    if config['Settings']['use_wandb']:
        wandb.finish()

    return training_info['name']

def train_quantum(model_class: nn.Module, model_hparams: dict, pi_scale: bool):

    model = model_class(score_dim=score_dim, **model_hparams)

    num_ir_qubits = model_hparams['num_ir_qubits']
    num_nr_qubits = model_hparams['num_nr_qubits']
    num_layers = model_hparams['num_layers']
    num_reupload = model_hparams['num_reupload']
    model_description = f"nI{num_ir_qubits}_nQ{num_nr_qubits}_l{num_layers}_r{num_reupload}"

    accelerator = 'cpu'
    name = train(model, model_description, model_hparams, accelerator, graph=False, pi_scale=pi_scale)

    return name

def train_mpgnn(model_hparams: dict):
    
    model = ClassicalMPGNN(score_dim=score_dim, **model_hparams)

    gnn_out = model_hparams['gnn_out']
    gnn_hidden = model_hparams['gnn_hidden']
    gnn_layers = model_hparams['gnn_layers']
    model_description = f"go{gnn_out}_gh{gnn_hidden}_gl{gnn_layers}"

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    name = train(model, model_description, model_hparams, accelerator, graph=True)
    
    return name

def train_benchmark(model_class: nn.Module, lite: bool = False):
    if lite:
        with open('configs/benchmark_lite.yaml', 'r') as file:
            hparams = yaml.safe_load(file)[model_class.__name__]
            model_description = 'lite'
    else:
        with open('configs/benchmark.yaml', 'r') as file:
            hparams = yaml.safe_load(file)[model_class.__name__]
            model_description = ''

    model = model_class(score_dim=score_dim, parameters=hparams)
    
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    name = train(model, model_description, hparams, accelerator, graph=False)
    
    return name

# %%
# # QCGNN
# for num_nr_qubits in [3, 5, 7]:
#     print(f"\n* Train QCGNN {num_nr_qubits}.\n")
#     qcgnn_hparams = {'num_ir_qubits': 4, 'num_nr_qubits': num_nr_qubits, 'num_layers': 1, 'num_reupload': num_nr_qubits}
#     name = train_quantum(model_class=QuantumRotQCGNN, model_hparams=qcgnn_hparams, pi_scale=True)

# %%
# # Hybrid
# for num_nr_qubits in [3, 5, 7]:
#     print(f"\n* Train Hybrid {num_nr_qubits}.\n")
#     qcgnn_hparams = {'num_ir_qubits': 4, 'num_nr_qubits': num_nr_qubits, 'num_layers': 1, 'num_reupload': num_nr_qubits}
#     name = train_quantum(model_class=HybridQCGNN, model_hparams=qcgnn_hparams, pi_scale=False)dddddsdasdasdasdasd

# %%
# MPGNN

for gnn_dim in [3, 5, 7, 64]:
    print(f"\n* Train MPGNN {gnn_dim}.\n")
    mpgnn_hparams = {'gnn_in': 6, 'gnn_out': gnn_dim, 'gnn_layers': 2, 'gnn_hidden': gnn_dim}
    name = train_mpgnn(model_hparams=mpgnn_hparams)

# %%
# Particle Transformer (without interaction).

print(f"\n* Train Particle Transformer.\n")
name = train_benchmark(model_class=ParticleTransformer, lite=False)
name = train_benchmark(model_class=ParticleTransformer, lite=True)

# %%
# Particle Net.

print(f"\n* Train Particle Net.\n")
name = train_benchmark(model_class=ParticleNet, lite=False)
name = train_benchmark(model_class=ParticleNet, lite=True)