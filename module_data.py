"""Module related to reading jet data.

When using the package `fastjet`, the warning related to citation will  
pop up, so we initialize a simple `fastjet.ClusterSequence` to make it  
pop up in the first place.

To read data from MadGraph5 (mg5), use the class `FatJetEvents`. We 
assume the mg5 data is generated in directory `./jet_dataset`.

This module also provides classes to transform data into formats of 
`torch` or `pytorch-lightning`.

Loading dataset from raw mg5 data may be time consuming, we provide
functions to save and load the energy information with `hdf5` files.
"""

import h5py
from itertools import product
import json
import os
import random
from typing import Union

import awkward as ak
import fastjet
import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
import uproot


# PID table.
pdgid_table = {
    "electron": 11, 
    "muon": 13, 
    "gamma": 22, 
    "ch_hadron": 211, 
    "neu_hadron": 130, 
    "HF_hadron": 1, 
    "HF_em": 2,
    }


# Run fastjet 1 time then it won't show up citation warning.
_jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
_array   = ak.Array([{"px": 0.1, "py": 0.2, "pz": 0.3, "E": 0.4},])
fastjet.ClusterSequence(_array, _jet_def)


# logging function
def _log(message: str) -> None:
    """Printing function for log."""
    print(f"\n# DataLog: {message}\n")


class FatJetEvents:
    """Create fatjet events.
    
    The most important thing is to get the attribute `self.events`. We
    then generate data through:
    - generate_fastjet_events
    - generate_uniform_pt_events
    - generate_max_pt_events
    """
    def __init__(
            self,
            channel: str,
            cut_pt: tuple[float, float] = None, 
            subjet_radius: float = 0, 
            num_pt_ptcs: int = "Full",
            use_hdf5: bool = True
            ):
        """Read MadGraph5 raw data.

        Note the default mg5 dataset is assumed to be loaded from 
        directory `./jet_dataset`.

        Args:
            channel : str
                The mg5 dataset directory name in `./jet_dataset`.
            cut_pt : tuple[float, float] (default None)
                The range of transverse momentum pt to be cut.
            subjet_radius : float (default 0)
                Radius for reclustering jet particles into subjets.
            num_pt_ptcs : int (default "Full")
                Number of particles to be selected within a jet. The 
                selection criteria is by sorting with pt.
            use_hdf5 : bool (default True)
                Whether to check if there is hdf5 file available, 
                loading hdf5 is much more faster than mg5 war data.
        """

        self.channel = channel
        self.cut_pt = cut_pt
        self.subjet_radius = subjet_radius
        self.num_pt_ptcs = num_pt_ptcs
        
        if use_hdf5:
            # If hdf5 file exists, just load it.
            data_info = f"c{cut_pt[0]}_{cut_pt[1]}_r{subjet_radius}"
            self.events = load_hdf5(channel, data_info)
            _log(f"Load {channel} {data_info} hdf5 file.")
        else:
            # Read original MadGraph5 root file through 'uproot'.
            dir_path = f"{os.path.dirname(__file__)}/jet_dataset"
            mg5_path = f"{dir_path}/{channel}/Events/run_01"
            root_file = "tag_1_delphes_events.root"
            root_path = os.path.join(dir_path, mg5_path, root_file)
            events = uproot.open(root_path + ":Delphes;1")

            # Set aliases of branches for uproot reading data
            aliases = {
                'fatjet_pt': 'FatJet/FatJet.PT',
                'fatjet_eta': 'FatJet/FatJet.Eta',
                'fatjet_phi': 'FatJet/FatJet.Phi',
                'fatjet_ptcs': 'FatJet/FatJet.Particles',
                'ptcs_pid': 'Particle/Particle.PID',
                'ptcs_e': 'Particle/Particle.E',
                'ptcs_pt': 'Particle/Particle.PT',
                'ptcs_eta': 'Particle/Particle.Eta',
                'ptcs_phi': 'Particle/Particle.Phi',
                'ptcs_px': 'Particle/Particle.Px',
                'ptcs_py': 'Particle/Particle.Py',
                'ptcs_pz': 'Particle/Particle.Pz',
            }
            events = events.arrays(
                expressions=aliases.keys(), 
                cut=None, 
                aliases=aliases,
                )
            
            # Select events with at least one fatjet.
            events = events[(ak.num(events['fatjet_pt']) > 0)]

            # Get the jet with highest pt in each events.
            jet_arg_sort = ak.argsort(events[f'fatjet_pt'], ascending=False)
            jet_index = ak.firsts(jet_arg_sort, axis=1)
            jet_index = ak.unflatten(jet_index, counts=ak.ones_like(jet_index))
            events['fatjet_pt'] = ak.flatten(events['fatjet_pt'][jet_index])
            events['fatjet_eta'] = ak.flatten(events['fatjet_eta'][jet_index])
            events['fatjet_phi'] = ak.flatten(events['fatjet_phi'][jet_index])

            # Get daughters information of each fatjets.
            refs = events['fatjet_ptcs'][jet_index].refs[:, 0] - 1
            events['fatjet_daughter_e'] = events['ptcs_e'][refs]
            events['fatjet_daughter_pt'] = events['ptcs_pt'][refs]
            events['fatjet_daughter_eta'] = events['ptcs_eta'][refs]
            events['fatjet_daughter_phi'] = events['ptcs_phi'][refs]
            events['fatjet_daughter_pid'] = events['ptcs_pid'][refs]
            events['fatjet_daughter_px'] = events['ptcs_px'][refs]
            events['fatjet_daughter_py'] = events['ptcs_py'][refs]
            events['fatjet_daughter_pz'] = events['ptcs_pz'][refs]

            # Remove unnecessary records which contain 'ptcs' in field name,
            # since ptcs records all jets information but we only need the jets
            # with highest transverse momentum pt.
            fields = [field for field in events.fields if 'ptcs' not in field]
            events = events[fields]

            # Cut events that not in pt range by `cut_pt``.
            if cut_pt is not None:
                low, high = min(cut_pt), max(cut_pt)
                pt = events['fatjet_pt']
                cut = (pt >= low) * (pt < high)
                events = events[cut]

            # Finish extracting information from mg5 data.
            self.events  = events
            _log(f"Successfully create {channel} with {len(events)} events.")

            if subjet_radius != 0:
                # Reclustering fastjet events if subjet_radius > 0.
                self.generate_fastjet_events(subjet_radius)
            else:
                # We will use data with suffix start with "fast"
                events['fast_pt'] = events['fatjet_daughter_pt']
                events['fast_delta_eta'] = events['fatjet_daughter_eta'] - events['fatjet_eta']
                # Mod delta_phi into [-pi, pi)
                delta_phi = events['fatjet_daughter_phi'] - events['fatjet_phi']
                events['fast_delta_phi'] = np.mod(delta_phi+np.pi, 2*np.pi) - np.pi
        
        # The number of particles in original events data is zig-zagged. For 
        # analysis, we specify the maximum number of particles by `num_pt_ptcs`
        # argument. The default is set to `num_pt_ptcs="Full"`, which retrain
        # all particles. Specify an integer for maximum number of particles.
        if num_pt_ptcs != "Full":
            self.generate_max_pt_events(num_pt_ptcs)

    def generate_fastjet_events(
            self, 
            subjet_radius: int, 
            algorithm = fastjet.antikt_algorithm
            ) -> None:
        '''Reclustering particles into subjets.

        Since the number of particles is too large for quantum machine
        learning, we decrease by reclustering the particles into subjets.

        Args:
            subjet_radius : int
                Subjet radius for reclustering.
            algorithm (default fastjet.antikt_algorithm)
                Algorithm for reclustering particles (see fastjet for 
                further details).
        '''

        _log(f"Reclustering {self.channel} with radius {subjet_radius}")

        # Reclustering jets event by event.
        fastjet_list = []
        for event in self.events:
            four_momentums = ak.Array({
                "px": event["fatjet_daughter_px"], 
                "py": event["fatjet_daughter_py"], 
                "pz": event["fatjet_daughter_pz"],
                "E" : event["fatjet_daughter_e"],
                })
            jet_def = fastjet.JetDefinition(algorithm, float(subjet_radius))
            cluster = fastjet.ClusterSequence(four_momentums, jet_def)
            fastjet_list.append(cluster.inclusive_jets())
            
        # The fastjet outputs are represented in E, px, py, pz.
        fastjet_events = ak.Array(fastjet_list)
        E = fastjet_events["E"]
        px = fastjet_events["px"]
        py = fastjet_events["py"]
        pz = fastjet_events["pz"]
        jet_eta = self.events[f"fatjet_eta"]
        jet_phi = self.events[f"fatjet_phi"]
        fastjet_events["e"] = E
        fastjet_events["p"] = (px**2 + py**2 + pz**2) ** 0.5
        fastjet_events["pt"] = (px**2 + py**2) ** 0.5
        fastjet_events["theta"] = np.arccos(pz / fastjet_events["p"])
        fastjet_events["eta"] = - np.log(np.tan((fastjet_events["theta"]) / 2))
        fastjet_events["phi"] = np.arctan2(py, px)
        fastjet_events["delta_eta"] = fastjet_events["eta"] - jet_eta
        # Mod delta_phi into [-pi, pi)
        delta_phi = fastjet_events["phi"] - jet_phi
        fastjet_events["delta_phi"] = np.mod(delta_phi+np.pi, 2*np.pi) - np.pi
        
        # Finish reclustering and merge with original events.
        for field in fastjet_events.fields:
            self.events[f"fast_{field}"] = fastjet_events[field]
        
        _log(f"Finish reclustering {self.channel}")
    
    def generate_uniform_pt_events(self, bin: int, num_bin_data: int) -> ak.Array:
        '''Uniformly generate events in each pt bin
        
        Args:
            bin : int
                Number of pt bins.
            num_bin_data: int
                Number of data (events) in each bin.
        
        Returns:
            ak.Array: Concatenated events in each bin.
        '''

        # Determine the lower and upper limits of pt.
        if self.cut_pt is not None:
            low = min(self.cut_pt)
            high = max(self.cut_pt)
        else:
            low = min(self.events['fatjet_pt'])
            high = max(self.events['fatjet_pt'])
        bin_interval = (high - low) / bin
        
        # Uniformly generate events in each pt bin.
        events_buffer = []
        pt = self.events['fatjet_pt']
        for i in range(bin):
            bin_lower = low + bin_interval * i
            bin_upper = low + bin_interval * (i+1)
            bin_selected = (pt >= bin_lower) * (pt < bin_upper)
            bin_events = self.events[bin_selected]
            
            # Randomly select uniform events in each pt bin.
            if num_bin_data > len(bin_events):
                raise ValueError(f"num_bin_data {num_bin_data} > {len(bin_events)}")
            else:
                rnd_range = range(len(bin_events))
                rnd_index = random.sample(rnd_range, num_bin_data)
                events_buffer.append(bin_events[rnd_index])
        
        return ak.concatenate(events_buffer)
    
    def generate_max_pt_events(self, num_pt_ptcs:int) -> None:
        '''Reduce the number of particles in each jets
        
        Since the number of particles in each jets is too many for QML, 
        we retain certain `num_pt_ptcs` number of particles with highest
        transverse momentum pt.

        Args:
            num_pt_ptcs : int
                Number of particles to retain with highest pt.
        '''
        
        # Highest arguments sorted by pt.
        max_arg = ak.argsort(self.events["fast_pt"], ascending=False, axis=1)
        max_arg = max_arg[:, :num_pt_ptcs]

        # Retain information start with "fast_".
        for field in self.events.fields:
            if "fast_" in field:
                self.events[field] = self.events[field][max_arg]


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


class JetDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            sig_events: ak.Array, 
            bkg_events: ak.Array, 
            data_ratio: float, 
            batch_size: int, 
            graph: bool
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
            data_ratio : float
                The ratio of training data and total data (training +
                testing).
            batch_size : int
                Batch size for data loaders.
            graph : bool
                If True, the dataset is stored as graph (using pytorch-
                grometric). Otherwise, the dataset is stored in tensors.
        """

        super().__init__()
        self.graph = graph
        self.data_ratio = data_ratio
        self.batch_size = batch_size

        # Determine maximum number of particles within jets
        self.max_num_ptcs = max(
            max(ak.count(sig_events["fast_pt"], axis=1)),
            max(ak.count(bkg_events["fast_pt"], axis=1)),
            )
        _log(f"Max number of particles = {self.max_num_ptcs}")
        
        # Preprocess the events.
        sig_events = self._preprocess(sig_events)
        bkg_events = self._preprocess(bkg_events)

        # Prepare dataset for dataloaders.
        num_train = int(data_ratio * len(sig_events))
        self.train_dataset = (
            self._dataset(sig_events[:num_train], 1) 
            + self._dataset(bkg_events[:num_train], 0)
            )
        self.test_dataset = (
            self._dataset(sig_events[num_train:], 1) 
            + self._dataset(bkg_events[num_train:], 0)
            )

    def _preprocess(self, events: ak.Array) -> list[torch.tensor]:
        """Function for preprocessing events
        
        The fatjet radius is set to 0.8. Prefix that starts with "fast" 
        is a fastjet feature. If the subjet_radius is 0 then "fast"
        features are just original particle features.
        """
        # Jet (fatjet) radius used in MadGraph5.
        R = 0.8
        # Use pt, eta, phi as features f1, f2, f3
        f1 = np.arctan(events["fast_pt"] / events["fatjet_pt"])
        f2 = events["fast_delta_eta"] / R * (np.pi/2)
        f3 = events["fast_delta_phi"] / R * (np.pi/2)
        arrays = ak.zip([f1, f2, f3])
        arrays = arrays.to_list()
        events = []
        for i in range(len(arrays)):
            x = torch.tensor(arrays[i], dtype=torch.float32, requires_grad=False)
            events.append(x)
        return events

    def _dataset(self, events, y) -> Union[list, TorchDataset]:
        """Turn events into dataset objects depending on `self.graph`"""
        if self.graph:
            # Create graph structure dataset
            dataset = []
            for i in range(len(events)):
                x = events[i]
                # Use fully-connected edges.
                edge_index = list(product(range(len(x)), range(len(x))))
                edge_index = torch.tensor(edge_index, requires_grad=False)
                edge_index = edge_index.transpose(0, 1)
                # Turn into pytorch_geometric "Data" object
                dataset.append(Data(x=x, edge_index=edge_index, y=y))
        else:
            # Create padded tensor dataset
            def pad(x):
                return torch.nn.functional.pad(
                    input=x,
                    pad=(0,0,0,self.max_num_ptcs-len(x)),
                    mode="constant",
                    value=0,
                )
            dataset = TorchDataset(
                x=[pad(events[i]) for i in range(len(events))], 
                y=[y]*len(events)
                )
        return dataset

    def train_dataloader(self):
        """Training data loader"""
        if self.graph == True:
            return GeoDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            return TorchDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Validation data loader (same as testing data)"""
        if self.graph == True:
            return GeoDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            return TorchDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Testing data loader (same as validation data)"""
        if self.graph == True:
            return GeoDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            return TorchDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    

def save_hdf5(channel, data_info, ak_array):
    """Save events data to HDF5 file
    
    Save FatJetEvents.events to HDF5 file that have already been
    preprocessed.

    The codes below are followed from:
    https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#saving-awkward-arrays-to-hdf5
    """
    _log(f"Start creating {channel}|{data_info}.hdf5 file")
    dir_path = os.path.join(os.path.dirname(__file__), "jet_dataset")
    hdf5_name = f"{channel}|{data_info}.hdf5"
    hdf5_file = h5py.File(os.path.join(dir_path, hdf5_name), "w")
    hdf5_group = hdf5_file.create_group(channel)
    form, length, _ = ak.to_buffers(ak.to_packed(ak_array), container=hdf5_group)
    hdf5_group.attrs["form"] = form.to_json()
    hdf5_group.attrs["length"] = json.dumps(length)
    _log(f"Successfully creating {channel}|{data_info}.hdf5 file")


def load_hdf5(channel, data_info):
    """Load events data to HDF5 file
    
    Load HDF5 file that have already been preprocessed.

    The codes below are followed from:
    https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#reading-awkward-arrays-from-hdf5
    """
    hdf5_file = h5py.File(f"{os.path.dirname(__file__)}/jet_dataset/{channel}|{data_info}.hdf5", "r")
    hdf5_group = hdf5_file[channel]
    ak_array = ak.from_buffers(
        ak.forms.from_json(hdf5_group.attrs["form"]),
        json.loads(hdf5_group.attrs["length"]),
        {k: np.asarray(v) for k, v in hdf5_group.items()},
    )
    return ak_array