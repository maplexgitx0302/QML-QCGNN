"""Loggers for logging training procedure."""

import os
import yaml

from lightning.pytorch.loggers import WandbLogger, CSVLogger
import torch

from source.utils.device import get_cpu_name, get_gpu_name
from source.utils.path import root_path

training_logs_dir = 'result/training_logs'

device_info = {
    'cpu': get_cpu_name().replace(' ', '_').replace('-', '_'),
    'gpu': get_gpu_name().replace(' ', '_').replace('-', '_'),
    'num_threads': torch.get_num_threads(),
    'num_cpu_cores': os.cpu_count(),
}


def csv_logger(training_info: dict):
    """Create a `CSVLogger` for default."""
    
    save_dir = os.path.join(root_path, training_logs_dir, 'CSVLogger')
    os.makedirs(save_dir, exist_ok=True)

    logger = CSVLogger(save_dir=save_dir, name=training_info['name'])
    
    log_info = training_info.copy()
    log_info.update(device_info)
    logger.log_hyperparams(log_info)

    return logger


def wandb_logger(training_info: dict):
    """Create a `WandbLogger` for default."""
    
    save_dir = os.path.join(root_path, training_logs_dir, 'WandbLogger')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(root_path, 'configs', 'wandb.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
    
    logger = WandbLogger(
        name=training_info['name'],
        id=training_info['name'],
        project=wandb_config['project'],
        group=wandb_config['group'],
        save_dir=save_dir,
    )

    log_info = training_info.copy()
    log_info.update(device_info)
    logger.experiment.config.update(log_info, allow_val_change=True)

    return logger