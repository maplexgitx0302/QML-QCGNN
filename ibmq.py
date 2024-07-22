# %%
"""
# Test pre-trained QCGNN model on IBMQ

1. Must have an existing [IBMQ](https://quantum.ibm.com) account.
2. Create a local `./config.toml` file for PennyLane to link to the IBMQ, see [PennyLane configuration file](https://docs.pennylane.ai/en/latest/introduction/configuration.html#format) for further detail.
"""

# %%
import argparse
import os
import sys
import time
import yaml

import lightning as L
from lightning.pytorch.loggers import CSVLogger
import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.accounts import AccountAlreadyExistsError
import torch

from source.data.datamodule import JetTorchDataModule
from source.data.opendata import TopQuarkEvents
from source.models.qcgnn import QuantumRotQCGNN
from source.training.litmodel import TorchLightningModule
from source.utils.gmail import send_email

# %%
dataset = 'TopQCD' # 'JetNet', or 'TopQCD'

if 'ipykernel' in sys.argv[0]:
    mode = 'IBMQ' # 'IBMQ', or 'Simulator', or 'Noise'
    backend = ''
else:
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('--mode', type=str, help='Mode: IBMQ, Simulator, or Noise')
    parser.add_argument('--backend', type=str, default=None, help='Backend (default: None)')
    parser_args = parser.parse_args()

    mode = parser_args.mode
    backend = parser_args.backend

with open(f"configs/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

with open(f"configs/ibmq.yaml", 'r') as file:
    ibmq_config = yaml.safe_load(file)
    save_dir = ibmq_config['Result']['output_dir']
    os.makedirs(save_dir, exist_ok=True)

if mode == 'IBMQ':
    ibmq_config['Device']['IBMQ']['qbackend'] = backend
    try:
        print('Test IBMQ Authentication...')
        service = QiskitRuntimeService(instance="ibm-q-hub-ntu/ntu-internal/default")
        ibmq_backends = service.backends()
        for ibmq_backend in ibmq_backends:
            num_jobs =  ibmq_backend.status().pending_jobs
            print(f"IBMQ backend {ibmq_backend} has {num_jobs} pending jobs.")
    except AccountAlreadyExistsError:
        token = input('Enter IBMQ token: ')
        QiskitRuntimeService.save_account(token=token, overwrite=True)

# %%
num_data = ibmq_config['Data']['num_data']
num_ptcs = ibmq_config['Data']['num_ptcs']
dataset_config = {}
dataset_config.update(config['Data'])
dataset_config.update(config[dataset])
dataset_config['min_num_ptcs'] = num_ptcs
dataset_config['max_num_ptcs'] = num_ptcs

# Get 'TopQCD' events.
events = []
for y, channel in enumerate(['Top', 'QCD']):
    y_true = [y] * num_data
    top_qcd_events = TopQuarkEvents(mode='test', is_signal_new=y, **dataset_config)
    events.append(top_qcd_events.generate_events(num_data))
    print(f"{channel} has {len(top_qcd_events.events)} events -> selected = {num_data}\n")

# Turn into data-module.    
data_module = JetTorchDataModule(
    events=events,
    num_train=0,
    num_valid=0,
    num_test=num_data * config[dataset]['num_classes'],
    batch_size=ibmq_config['Data']['batch_size'],
    max_num_ptcs=num_ptcs,
    pi_scale=True
)

# %%
# Create QCGNN model.
n_I = ibmq_config['Pretrain']['n_I']
n_Q = ibmq_config['Pretrain']['n_Q']
model = QuantumRotQCGNN(
    num_ir_qubits=n_I,
    num_nr_qubits=n_Q,
    num_layers=n_Q//3,
    num_reupload=2,
    vqc_ansatz=qml.StronglyEntanglingLayers,
    score_dim=1,
    **ibmq_config['Device'][mode],
)

# Load pre-trained checkpoint.
last_ckpt = torch.load(ibmq_config['Pretrain']['ckpt_path']) # Last checkpoint.
best_ckpt = list(last_ckpt['callbacks'].values())[0]['best_model_path'] # Get the best checkpoint path.
best_state_dict = torch.load(best_ckpt)['state_dict'] # Load the best state dict.
best_state_dict = {k.replace('model.', ''): v for k, v in best_state_dict.items()} # Remove the 'model.' prefix.
model.load_state_dict(best_state_dict)

# Turn into a lightning model.
model.eval()
lit_model = TorchLightningModule(model=model, optimizer=None, score_dim=1, print_log=False)

# Start testing.
time_start = time.time()
logger = CSVLogger(save_dir=save_dir, name=f"{mode}-{backend}-{ibmq_config['Data']['rnd_seed']}")
trainer = L.Trainer(accelerator='cpu', default_root_dir=save_dir, logger=logger)

try:
    status = 'Success'
    message = ''
    trainer.test(model=lit_model, datamodule=data_module)
except Exception as e:
    status = 'Failed'
    message = e
finally:
    if mode == 'IBMQ':
        send_email(
            subject=f"IBMQ-{backend}: {status}",
            message=message,
            config={},
        )

time_end = time.time()

print(f"Time elapsed on IBMQ ({ibmq_config['Device'][mode]['qbackend']}): {time_end - time_start:.2f}s")