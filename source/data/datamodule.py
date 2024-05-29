"""Lightning data module"""

import functools
import itertools
from typing import Optional

import awkward as ak
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.tensor, y: torch.tensor):
        """Torch format dataset.

        Args:
            x : torch.tensor
                Feature data.
            y : torch.tensor
                Label of data.
        """
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class JetTorchDataModule(L.LightningDataModule):
    def __init__(
            self,
            events: list[ak.Array],
            num_train: int,
            num_valid: int,
            num_test: int,
            batch_size: int,
            max_num_ptcs: Optional[int] = None,
            pi_scale: Optional[bool] = False,
            **kwargs
        ):
        """Pytorch Lightning Data Module for jet.

        Similar to PyTorch DataLoader, but pytorch-lightning DataModule
        provides more concrete structures. Note that we choose the 
        validation data to be the same as testing data, since we just 
        want to monitor the behavior during training.

        Args:
            events : list[ak.Array]
                List of FatJetEvents, label will be the sequence order.
            num_train / num_valid / num_test : int
                Number of training / validation / testing data.
            batch_size : int
                Batch size for data loaders.
            max_num_ptcs : int (default None)
                Pad number of particles within jets, used for non-graph
                data, i.e., `graph == False`.
        """

        super().__init__()
        self.batch_size = batch_size

        # Determine maximum number of particles within jets
        if max_num_ptcs is None:
            max_num_ptcs = max([max(ak.count(_events['pt'], axis=1)) for _events in events])
        self.max_num_ptcs = max_num_ptcs

        # Preprocess the events.
        events = [self._preprocess(_events, pi_scale) for _events in events]

        # Prepare dataset for dataloaders.
        train_events = [_events[:num_train] for _events in events]
        valid_events = [_events[num_train : num_train + num_valid] for _events in events]
        test_events  = [_events[num_train + num_valid : num_train + num_valid + num_test] for _events in events]

        train_events = [self._dataset(_events, i) for i, _events in enumerate(train_events)]
        valid_events = [self._dataset(_events, i) for i, _events in enumerate(valid_events)]
        test_events  = [self._dataset(_events, i) for i, _events in enumerate(test_events)]

        self.train_dataset = functools.reduce(lambda x, y: x + y, train_events)
        self.valid_dataset = functools.reduce(lambda x, y: x + y, valid_events)
        self.test_dataset  = functools.reduce(lambda x, y: x + y, test_events)
        
    def _preprocess(self, events: ak.Array, pi_scale: bool) -> list[torch.tensor]:
        """Function for preprocessing events."""
        
        if pi_scale: # Rescale features in [-pi / 2, pi / 2]
            R = 0.8
            f1 = np.arctan(events['pt'] / events['fatjet_pt'])
            f2 = events['delta_eta'] / R * (np.pi / 2)
            f3 = events['delta_phi'] / R * (np.pi / 2)

        else:
            f1 = events['pt'] / events['fatjet_pt']
            f2 = events['delta_eta']
            f3 = events['delta_phi']
        
        # Since the data size is zig-zag, use list.
        arrays = ak.zip([f1, f2, f3])
        arrays = arrays.to_list()
        events = [torch.tensor(arrays[i]).float() for i in range(len(arrays))]
        
        return events

    def _dataset(self, events: torch.Tensor, y: int) -> TorchDataset:
        
        # Create padded tensor dataset
        pad_function = lambda x: nn.functional.pad(
            input=x,
            pad=(0, 0, 0, self.max_num_ptcs - len(x)),
            mode="constant",
            value=float('nan')
        )
        
        x = list(map(pad_function, events))
        y = torch.full((len(events), ), y)
        dataset = TorchDataset(x=x, y=y)
        
        return dataset

    def train_dataloader(self):
        """Training data loader"""
        return TorchDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Validation data loader"""
        return TorchDataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Testing data loader"""
        return TorchDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class JetGraphDataModule(JetTorchDataModule):
    """Data module for torch_geometric.data.Data."""
    
    def _dataset(self, events: torch.Tensor, y: int) -> list:

        # Create graph structure dataset
        dataset = []

        for i in range(len(events)):
            x = events[i]

            # Use fully-connected edges (including self loop).
            edge_index = list(itertools.product(range(len(x)), range(len(x))))
            edge_index = torch.tensor(edge_index, requires_grad=False)
            edge_index = edge_index.transpose(0, 1).contiguous()

            # Turn into pytorch_geometric "Data" object
            dataset.append(Data(x=x, edge_index=edge_index, y=y))

        return dataset

    def train_dataloader(self):
        return GeoDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return GeoDataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return GeoDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)