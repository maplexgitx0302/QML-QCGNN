"""Lightning data module"""

import itertools

import awkward as ak
import lightning as L
import numpy as np
import torch
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
            sig_events: ak.Array,
            bkg_events: ak.Array,
            num_train: int,
            num_valid: int,
            batch_size: int,
            pad_num_ptcs: int = None,
            **kwargs
        ):
        """Pytorch Lightning Data Module for jet.

        Similar to PyTorch DataLoader, but pytorch-lightning DataModule
        provides more concrete structures. Note that we choose the 
        validation data to be the same as testing data, since we just 
        want to monitor the behavior during training.

        Args:
            sig_events : ak.Array
                Typically just FatJetEvents.events of signal channel.
            bkg_events : ak.Array
                Typically just FatJetEvents.events of backgound channel.
            num_train : int
                Number of training data.
            num_valid : int
                Number of validation data.
            batch_size : int
                Batch size for data loaders.
            pad_num_ptcs : int (default None)
                Pad number of particles within jets, used for non-graph
                data, i.e., `graph == False`.
        """

        super().__init__()
        self.batch_size = batch_size

        # Determine maximum number of particles within jets
        if pad_num_ptcs is None:
            pad_num_ptcs = max(
                max(ak.count(sig_events['pt'], axis=1)),
                max(ak.count(bkg_events['pt'], axis=1)),
            )
        self.pad_num_ptcs = pad_num_ptcs

        # Preprocess the events.
        sig_events = self._preprocess(sig_events)
        bkg_events = self._preprocess(bkg_events)

        # Prepare dataset for dataloaders.
        sig_train = sig_events[:num_train]
        bkg_train = bkg_events[:num_train]
        sig_valid = sig_events[num_train : num_train + num_valid]
        bkg_valid = bkg_events[num_train : num_train + num_valid]

        self.train_dataset = self._dataset(sig_train, 1) + self._dataset(bkg_train, 0)
        self.valid_dataset = self._dataset(sig_valid, 1) + self._dataset(bkg_valid, 0)

    def _preprocess(self, events: ak.Array) -> list[torch.tensor]:
        """Function for preprocessing events."""
        
        # Jet (fatjet) radius used in MadGraph5.
        R = 0.8
        
        # Use pt, eta, phi as features f1, f2, f3
        f1 = np.arctan(events['pt'] / events['fatjet_pt'])
        f2 = events['delta_eta'] / R * (np.pi / 2)
        f3 = events['delta_phi'] / R * (np.pi / 2)
        
        # Since the data size is zig-zag, use list.
        arrays = ak.zip([f1, f2, f3])
        arrays = arrays.to_list()
        events = [torch.tensor(arrays[i]).float() for i in range(len(arrays))]
        
        return events

    def _dataset(self, events, y) -> TorchDataset:
        
        # Create padded tensor dataset
        pad_function = lambda x: torch.nn.functional.pad(
            input=x,
            pad=(0, 0, 0, self.pad_num_ptcs - len(x)),
            mode="constant",
            value=0
        )
        
        x = list(map(pad_function, events))
        y = torch.full((len(events), ), y)
        dataset = TorchDataset(x=x, y=y)
        
        return dataset

    def train_dataloader(self):
        """Training data loader"""
        return TorchDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Validation data loader (same as testing data)"""
        return TorchDataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Testing data loader (same as validation data)"""
        return TorchDataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)


class JetGraphDataModule(JetTorchDataModule):
    """Data module for torch_geometric.data.Data."""
    
    def _dataset(self, events, y) -> list:

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
        return GeoDataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)