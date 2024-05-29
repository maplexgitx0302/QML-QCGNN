"""Module related to reading jet data.

Read open data:
- JetNet: https://zenodo.org/records/6975118
- Top Quark Tagging: https://zenodo.org/records/2603256
"""

from typing import Optional

import os
import random

import awkward as ak
import h5py
import numpy as np
import pandas as pd

from source.utils.path import root_path

dataset_dir = os.path.join(root_path, 'dataset')

class FatjetEvents:

    def __init__(
            self,
            events: ak.Array,
            mask: np.ndarray,
            min_num_ptcs: Optional[int] = None,
            max_num_ptcs: Optional[int] = None,
            pt_threshold: Optional[float] = None,
            pt_min: Optional[float] = None,
            pt_max: Optional[float] = None,
            **kwargs,
    ):
        """Fatjet events with particle information.
        
        Args:
            events : ak.Array
                Should contain fields: ['fatjet_pt', 'pt', 'pt_rel', 'delta_eta', 'delta_phi']
            mask : np.ndarray
                Mask to remove particles.
            min_num_ptcs : int, optional
                Minimum number of particles per jet.
            max_num_ptcs : int, optional
                Maximum number of particles per jet.
            pt_threshold : float, optional
                Minimum pt_rel threshold.
            pt_min : float, optional
                Minimum jet daughter particle pt.
            pt_max : float, optional
                Maximum jet daughter particle pt.
        """

        self.min_num_ptcs = min_num_ptcs
        self.max_num_ptcs = max_num_ptcs
        self.pt_threshold = pt_threshold
        self.pt_min = pt_min
        self.pt_max = pt_max
        
        self.events = self._preprocess_events(events, mask)

    def _read_dataset(self) -> tuple[ak.Array, np.ndarray]:
        """Return events and mask."""
        pass

    def _preprocess_events(self, events: ak.Array, mask: np.ndarray) -> ak.Array:
        """Cut events with mask."""

        # Throw out too soft particles.
        mask = mask | (events['pt_rel'] < 1e-8)

        # Daughter particle fields.
        daughter_fields = ['delta_eta', 'delta_phi', 'pt', 'pt_rel']
        
        # Drop `nan` values.
        for field in daughter_fields:
            # `ak.mask` will fill `mask == False` with `None`,
            # but our mask is `True` for the particles we want to remove.
            events[field] = ak.mask(events[field], ~mask)
            events[field] = ak.drop_none(events[field])

        # Select events with pt_min <= jet_pt <= pt_max.
        if self.pt_min is not None:
            events = events[events['fatjet_pt'] >= self.pt_min]
        if self.pt_max is not None:
            events = events[events['fatjet_pt'] <= self.pt_max]

        # Select particles with pt_rel >= pt_threshold.
        if self.pt_threshold is not None:
            retain_idx = (events['pt_rel'] >= self.pt_threshold)
            for field in daughter_fields:
                events[field] = events[field][retain_idx]

        # Select number of particles per jet in the constraint range.
        events['fatjet_num_ptcs'] = ak.num(events['pt'])
        if self.min_num_ptcs is not None:
            events = events[(events['fatjet_num_ptcs'] >= self.min_num_ptcs)]
        if self.max_num_ptcs is not None:
            events = events[(events['fatjet_num_ptcs'] <= self.max_num_ptcs)]

        return events

    def generate_events(self, num_events: int) -> ak.Array:
        """Generate events with random selection."""
        
        total_num_events = len(self.events['fatjet_pt'])
        
        return self.events[random.sample(range(total_num_events), num_events)]
    

class JetNetEvents(FatjetEvents):

    def __init__(self, channel: str, **kwargs):
        """JetNet dataset from https://zenodo.org/records/6975118
        
        Args:
            channel : str
                Source of the jet, could be ['q', 'g', 't', 'w', 'z'].
        """

        # Read dataset.
        if channel == 'q_g':
            events_q, mask_q = self._read_dataset('q')
            events_g, mask_g = self._read_dataset('g')
            events = ak.concatenate([events_q, events_g], axis=0)
            mask = np.concatenate([mask_q, mask_g], axis=0)
        elif channel == 'w_z':
            events_w, mask_w = self._read_dataset('w')
            events_z, mask_z = self._read_dataset('z')
            events = ak.concatenate([events_w, events_z], axis=0)
            mask = np.concatenate([mask_w, mask_z], axis=0)
        else:
            events, mask = self._read_dataset(channel)

        # Preprocess the events.
        super().__init__(events=events, mask=mask, **kwargs)

    def _read_dataset(self, channel: str) -> tuple[ak.Array, np.ndarray]:

        # Read `.hdf5` file.
        print(f"Loading JetNet dataset from channel: {channel}")
        hdf5_path = os.path.join(dataset_dir, 'jetnet', channel + '.hdf5')
        hdf5_file = h5py.File(hdf5_path, 'r')

        # `particle_features` shape (N, 30, 4) with 4 features (eta_rel, phi_rel, pt_rel, mask)
        ptc_events = np.array(hdf5_file['particle_features'])
        ptc_events, mask = ptc_events[..., :-1], (ptc_events[..., -1] == 0.)

        # `jet_features` shape (N, 4) with 4 features (pt, eta, mass, # of particles)
        jet_events = np.array(hdf5_file['jet_features'])

        # Convert to awkward array.
        events = ak.Array({

            # Fatjet information (R = 0.8).
            'fatjet_pt': jet_events[:, 0],
            
            # Daughter particle information.
            'delta_eta': ptc_events[..., 0],
            'delta_phi': ptc_events[..., 1],
            'pt_rel': ptc_events[..., 2],
            'pt': ptc_events[..., 2] * jet_events[:, 0, np.newaxis],

        })

        # Additional information.
        events['fatjet_eta'] = jet_events[:, 1]
        events['fatjet_mass'] = jet_events[:, 2]
        events['fatjet_num_ptcs'] = jet_events[:, 3]

        return events, mask


class TopQuarkEvents(FatjetEvents):

    def __init__(self, mode: str, is_signal_new: int, **kwargs):
        """Top quark tagging dataset from https://zenodo.org/records/2603256
        
        Args:
            mode : str
                Dataset to be load, could be ['train', 'val', 'test'].
            is_signal_new : int
                1 for top, 0 for QCD.
        """

        # Read dataset.
        events, mask = self._read_dataset(mode, is_signal_new)

        # Preprocess the events.
        super().__init__(events=events, mask=mask, **kwargs)

    def _read_dataset(self, mode: str, is_signal_new: int) -> tuple[ak.Array, np.ndarray]:

        # Read `.h5` file.
        print(f"Loading Top Quark dataset from mode: {mode}, is_signal_new: {is_signal_new}")
        h5_path = os.path.join(dataset_dir, 'top', mode + '.h5')
        df = pd.read_hdf(h5_path, key='table')

        # Select top or QCD.
        df = df[df['is_signal_new'].astype(int) == is_signal_new]

        # Find fatjet Px, Py, Pz.
        df['Fatjet_PX'] = df[[col for col in df.columns if col[:2] == 'PX']].sum(axis=1)
        df['Fatjet_PY'] = df[[col for col in df.columns if col[:2] == 'PY']].sum(axis=1)
        df['Fatjet_PZ'] = df[[col for col in df.columns if col[:2] == 'PZ']].sum(axis=1)

        # Convert (E, Px, Py, Pz) to (Pt, Eta, Phi)
        df['Fatjet_P'] = np.sqrt(df['Fatjet_PX'] ** 2 + df['Fatjet_PY'] ** 2 + df['Fatjet_PZ'] ** 2)
        df['Fatjet_Pt'] = np.sqrt(df['Fatjet_PX'] ** 2 + df['Fatjet_PY'] ** 2)
        df['Fatjet_Phi'] = np.arctan2(df['Fatjet_PY'], df['Fatjet_PX'])
        df['Fatjet_Phi'] = np.mod(df['Fatjet_Phi'] + np.pi, 2 * np.pi) - np.pi
        df['Fatjet_Eta'] = np.arctanh(df['Fatjet_PZ'] / df['Fatjet_P'])

        # Ignore warnings for calculating Pt, Phi, Eta of padded particles.
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

        # Get Pt, Phi, Eta of daughter particles.
        for i in range(200):
            df[f"P_{i}"] = np.sqrt(df[f"PX_{i}"] ** 2 + df[f"PY_{i}"] ** 2 + df[f"PZ_{i}"] ** 2)
            df[f"Pt_{i}"] = np.sqrt(df[f"PX_{i}"] ** 2 + df[f"PY_{i}"] ** 2)
            df[f"Phi_{i}"] = np.arctan2(df[f"PY_{i}"], df[f"PX_{i}"])
            df[f"Phi_{i}"] = np.mod(df[f"Phi_{i}"] + np.pi, 2 * np.pi) - np.pi
            df[f"Eta_{i}"] = np.arctanh(df[f"PZ_{i}"] / df[f"P_{i}"])

        # Turn into numpy array for convenience.
        pt = df[[f"Pt_{i}" for i in range(200)]].to_numpy()
        eta = df[[f"Eta_{i}" for i in range(200)]].to_numpy()
        phi = df[[f"Phi_{i}" for i in range(200)]].to_numpy()

        fatjet_pt = df['Fatjet_Pt'].to_numpy()
        fatjet_eta = df['Fatjet_Eta'].to_numpy()
        fatjet_phi = df['Fatjet_Phi'].to_numpy()

        pt_rel = pt / fatjet_pt[:, np.newaxis]
        delta_eta = eta - fatjet_eta[:, np.newaxis]
        delta_phi = phi - fatjet_phi[:, np.newaxis]

        # Mask for padded particles -> (pt_rel < 1e-8) | (eta == nan).
        mask = (pt_rel == 0.) | np.isnan(eta)

        events = ak.Array({

            # Fatjet information (R = 0.8).
            'fatjet_pt': fatjet_pt,
            'fatjet_eta': fatjet_eta,
            'fatjet_phi': fatjet_phi,

            # Daughter particle information.
            'delta_eta': delta_eta,
            'delta_phi': delta_phi,
            'pt_rel': pt_rel,
            'pt': pt,
            
        })

        return events, mask