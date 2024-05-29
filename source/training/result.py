"""Modules for presenting the training results."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from source.utils.path import root_path

sns.set_theme()

def read_csv(name: str, num_classes: int = 2):
    csv_dir = os.path.join(root_path, 'training_logs', 'CSVLogger')
    version = os.listdir(os.path.join(csv_dir, name))[-1]
    csv_path = os.path.join(csv_dir, name, version, 'metrics.csv')

    df = pd.read_csv(csv_path)

    logs = [
        'train_accuracy', 'valid_accuracy', 'train_auc', 'valid_auc',
        'epoch_time', 'train_loss', 'valid_loss_step', 'valid_loss_epoch'
    ]

    if num_classes >= 3:
        for i in range(num_classes):
            logs.append(f'train_ovr_accuracy_{i}')
            logs.append(f'valid_ovr_accuracy_{i}')

    logs = list(set(logs) & set(df.columns))

    metrics = {}
    for log in logs:
        metrics[log] = df.dropna(subset=[log])[log].to_numpy()
    
    return metrics


def plot_metrics(name:str, num_classes: int = 2):
    metrics = read_csv(name, num_classes)

    num_cols = 5 if num_classes >= 3 else 3
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

    def normalized_plot(ax, metric):
        if metric in metrics:
            x = np.linspace(0, 1, len(metrics[metric]))
            ax.plot(x, metrics[metric], label=metric, marker='.')
            ax.legend()
        else:
            print(f" * {metric} not found in the metrics.")

    ax_auc = 0
    ax_accuracy = 1
    ax_loss = 2
    ax_train_ovr_accuracy = 3
    ax_valid_ovr_accuracy = 4

    axes[ax_auc].set_title('AUC')
    axes[ax_accuracy].set_title('Accuracy')
    axes[ax_loss].set_title('Loss')
    
    normalized_plot(axes[ax_auc], 'train_auc')
    normalized_plot(axes[ax_accuracy], 'train_accuracy')
    normalized_plot(axes[ax_loss], 'train_loss')

    normalized_plot(axes[ax_auc], 'valid_auc')
    normalized_plot(axes[ax_accuracy], 'valid_accuracy')
    normalized_plot(axes[ax_loss], 'valid_loss_epoch')

    if num_classes >= 3:
        axes[ax_train_ovr_accuracy].set_title('Train_OVR_Arrcuracy')
        axes[ax_valid_ovr_accuracy].set_title('Valid_OVR_Arrcuracy')
        for i in range(num_classes):
            normalized_plot(axes[ax_train_ovr_accuracy], f"train_ovr_accuracy_{i}")
            normalized_plot(axes[ax_valid_ovr_accuracy], f"valid_ovr_accuracy_{i}")
        
    plt.show()