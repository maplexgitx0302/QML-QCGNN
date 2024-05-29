"""Module related to reading jet data.

To read data from MadGraph5 (MG5), use the class `FatJetEvents`. We 
assume the MG5 data is generated in directory `./dataset` in hdf5 format.

Loading dataset from full raw MG5 data may be time consuming, we provide
functions to save and load the energy flow information with `hdf5` files.
"""

import json
import os
import random

import awkward as ak
import h5py
import numpy as np
import uproot

from source.utils.path import root_path

dataset_dir = os.path.join(root_path, 'dataset')

# PID table.
pdgid_table = {
    'electron': 11,
    'muon': 13,
    'gamma': 22,
    'ch_hadron': 211,
    'neu_hadron': 130,
    'HF_hadron': 1,
    'HF_em': 2,
}


def _log(message: str) -> None:
    """Printing function for log."""
    print(f"# DataLog: {message}")


def read_full_data(channel):
    """Read original MadGraph5 root file through 'uproot'."""

    hdf5_path = os.path.join(dataset_dir, 'hvt', f"{channel}.root")
    events = uproot.open(hdf5_path + ':Delphes;1')

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

    # Retain information of the highest pt jets only.
    fields = [field for field in events.fields if 'ptcs' not in field]
    events = events[fields]

    return events


class FatJetEvents:
    def __init__(
            self,
            channel: str,
            num_bins: int,
            num_data_per_bin: int,
            pt_min: float = -1,
            pt_max: float = -1,
            pt_threshold: float = 0,
            max_num_ptcs: int = -1,
            num_ptcs_range: tuple[int, int] = None,
            **kwargs
        ):
        """Create dataset for jets of a particular channel.

        Args:
            channel : str
                The MG5 dataset directory name in `data/`.
            num_bins : int
                Number of bins in pt.
            num_data_per_bin: int
                Number of data (events) in each bin.
            pt_min : float (default -1)
                Minimum of pt to be selected.
            pt_max : float (default -1)
                Maximum of pt to be selected.
            pt_threshold : float (default 0)
                Retain particles with pt over threshold, with values
                (1 > pt_threshold >= 0) represented as percentage.
            max_num_ptcs : int (default -1)
                Number of particles to be selected within a jet. The 
                selection criteria is by sorting with pt.
            num_ptcs_range : tuple[int, int] (default None)
                If num_ptcs_range == (a, b), only select jets with 
                number of particles in range (a, b).
        """

        self.channel = channel
        self.num_bins = num_bins
        self.num_data_per_bin = num_data_per_bin
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.num_ptcs_range = num_ptcs_range
        self.fields = ['fatjet_pt', 'pt', 'delta_eta', 'delta_phi']

        # If `load_hdf5`, check existence and load it.
        try:
            events = self.load_hdf5_to_ak()
            _log(f"Load {channel}-pt_{pt_min}_{pt_max}.hdf5 file with {len(events['fatjet_pt'])} events.")
        
        except FileNotFoundError as _error:
            _log(_error)
            _log(f"Start reading from raw root file of channel {channel}.")
            
            events = read_full_data(channel)

            # Select events with pt_min <= jet_pt <= pt_max.
            if pt_min != -1:
                events = events[events['fatjet_pt'] >= pt_min]
            if pt_max != -1:
                events = events[events['fatjet_pt'] <= pt_max]

            # Finish extracting information from MG5 data.
            _log(f"Successfully create {channel} with {len(events)} events.")

            events['pt'] = events['fatjet_daughter_pt']
            events['delta_eta'] = events['fatjet_daughter_eta'] - events['fatjet_eta']
            events['delta_phi'] = events['fatjet_daughter_phi'] - events['fatjet_phi']
            events['delta_phi'] = np.mod(events['delta_phi'] + np.pi, 2 * np.pi) - np.pi

            # Only retain some of the fields.
            events = events[self.fields]

            # Sort by pt.
            pt_arg_sort = ak.argsort(events['pt'], ascending=False, axis=1)
            for field in ['pt', 'delta_eta', 'delta_phi']:
                events[field] = events[field][pt_arg_sort]

            self.save_ak_to_hdf5(events)

        # Set maximum number of daughter particles per jet.
        if max_num_ptcs != -1:
            for field in ['pt', 'delta_eta', 'delta_phi']:
                events[field] = events[field][:, :max_num_ptcs]

        # Retain particles with pt over threshold.
        if pt_threshold > 0:
            retain_idx = (events['pt'] >= (pt_threshold * events['fatjet_pt']))
            for field in ['pt', 'delta_eta', 'delta_phi']:
                events[field] = events[field][retain_idx]

        # Retain jets with number of particles at least 2.
        retain_idx = (ak.num(events['pt']) >= 2)
        events = events[retain_idx]

        # Finish data loading.
        self.events = events
        _log(f"After preprocessing, remaining num_events = {len(events['fatjet_pt'])}")

    def generate_uniform_pt_events(self) -> ak.Array:
        """Uniformly generate events in each pt bin."""

        # Determine the lower and upper limits of pt.
        events = self.events
        pt_min = self.pt_min if self.pt_min != -1 else min(events['fatjet_pt'])
        pt_max = self.pt_max if self.pt_max != -1 else max(events['fatjet_pt'])
        bin_interval = (pt_max - pt_min) / self.num_bins

        # Uniformly generate events in each pt bin.
        events_buffer = []
        for i in range(self.num_bins):
            bin_lower = pt_min + bin_interval * i
            bin_upper = pt_min + bin_interval * (i + 1)
            bin_selected = (events['fatjet_pt'] >= bin_lower) * (events['fatjet_pt'] < bin_upper)
            if self.num_ptcs_range is not None:
                num_ptcs = ak.num(events['pt'])
                bin_selected = bin_selected & (num_ptcs >= self.num_ptcs_range[0])
                bin_selected = bin_selected & (num_ptcs <= self.num_ptcs_range[1])
            bin_events = events[bin_selected]

            # Randomly select uniform events in each pt bin.
            if self.num_data_per_bin > len(bin_events):
                raise ValueError(f"{self.channel} # of data per bin {self.num_data_per_bin} > {len(bin_events)}")
            rnd_index = random.sample(range(len(bin_events)), self.num_data_per_bin)
            events_buffer.append(bin_events[rnd_index])

        return ak.concatenate(events_buffer)
    
    def print_bin_info(self):

        events = self.events
        pt_min = self.pt_min if self.pt_min != -1 else min(events['fatjet_pt'])
        pt_max = self.pt_max if self.pt_max != -1 else max(events['fatjet_pt'])
        bin_interval = (pt_max - pt_min) / self.num_bins
        
        print(f"---------- {self.channel} ----------")
        
        print(f" * Number of bins = {self.num_bins}")
        print(f" * Number of data per bin = {self.num_data_per_bin}")
        print(f" * Jet pt range in ({self.pt_min}, {self.pt_max})")

        for i in range(self.num_bins):
            bin_lower = self.pt_min + bin_interval * i
            bin_upper = pt_min + bin_interval * (i + 1)
            bin_selected = (events['fatjet_pt'] >= bin_lower) * (events['fatjet_pt'] < bin_upper)
            bin_events = events[bin_selected]
            print(f" * Number of events in pt ({bin_lower:.0f}, {bin_upper:.0f}) = {len(bin_events)}")

    def save_ak_to_hdf5(self, ak_array: ak.Array):
        """Save events data to HDF5 file

        Save FatJetEvents.events to HDF5 file that have already been
        preprocessed.

        The codes below are followed from:
        https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#saving-awkward-arrays-to-hdf5
        """

        # Use `h5py` package to save `hdf5` file.
        hdf5_name = f"{self.channel}-pt_{self.pt_min}_{self.pt_max}.hdf5"
        _log(f"Start creating {hdf5_name}.hdf5 file")
        hdf5_file = h5py.File(os.path.join(dataset_dir, hdf5_name), 'w')
        hdf5_group = hdf5_file.create_group(self.channel)
        form, length, _ = ak.to_buffers(ak.to_packed(ak_array), container=hdf5_group)
        hdf5_group.attrs['form'] = form.to_json()
        hdf5_group.attrs['length'] = json.dumps(length)
        _log(f"Successfully creating {hdf5_name}.hdf5 file")


    def load_hdf5_to_ak(self):
        """Load events data to HDF5 file

        Load HDF5 file that have already been preprocessed.

        The codes below are followed from:
        https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#reading-awkward-arrays-from-hdf5
        """

        # Use `h5py` package to load `hdf5` file.
        hdf5_name = f"{self.channel}-pt_{self.pt_min}_{self.pt_max}.hdf5"
        hdf5_file = h5py.File(os.path.join(dataset_dir, 'hvt', hdf5_name), 'r')
        hdf5_group = hdf5_file[self.channel]
        ak_array = ak.from_buffers(
            ak.forms.from_json(hdf5_group.attrs['form']),
            json.loads(hdf5_group.attrs['length']),
            {k: np.asarray(v) for k, v in hdf5_group.items()},
        )
        return ak_array