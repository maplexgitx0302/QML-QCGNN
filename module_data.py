# basic tools
import os, random
import numpy as np
from itertools import product

# HEP tools
import h5py, json
import awkward as ak
import uproot, fastjet

# ML tools
import torch
import lightning.pytorch as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

# PID table
pdgid_table = {"electron": 11, "muon": 13, "gamma": 22, "ch_hadron": 211, "neu_hadron": 130, "HF_hadron": 1, "HF_em": 2,}

# run fastjet 1 time then it won't show up cite reminder
_jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
_array   = ak.Array([{"px": 0.1, "py": 0.2, "pz": 0.3, "E": 0.4},])
fastjet.ClusterSequence(_array, _jet_def)

class FatJetEvents:
    def __init__(self, 
                 channel:str,                        # channel name (see dir `jet_dataset`)
                 cut_pt:tuple[float,float] = None,   # cut jet pt in a specific region
                 subjet_radius:float       = 0,      # radius for reclustering jets in subjets if needed
                 num_pt_ptcs:int           = "Full", # number of selected particles in jets (sorted by pt)
                 check_hdf5:bool           = True,   # whether to load hdf5 files if needed
                 ):
        '''Construct mg5 fatjet events with energy flow information'''
        self.channel       = channel
        self.cut_pt        = cut_pt
        self.subjet_radius = subjet_radius
        self.num_pt_ptcs   = num_pt_ptcs
        
        if check_hdf5 == True:
            # if hdf5 file exists, just load it
            data_info   = f"c{cut_pt[0]}_{cut_pt[1]}_r{subjet_radius}"
            print(f"# DataLog: Now loading hdf5 file {channel}|{data_info}.hdf5")
            self.events = load_hdf5(channel, data_info)
            print(f"# DataLog: Successfully loading hdf5 file {channel}|{data_info}.hdf5\n")
        else:
            # read original MadGraph5 root file through 'uproot'
            dir_path  = f"{os.path.dirname(__file__)}/jet_dataset"
            root_path = f"{dir_path}/{channel}/Events/run_01/tag_1_delphes_events.root"
            events    = uproot.open(root_path + ":Delphes;1")
            self.keys = events.keys()

            # select features
            aliases = {
                'fatjet_pt':'FatJet/FatJet.PT', 'fatjet_eta':'FatJet/FatJet.Eta', 'fatjet_phi':'FatJet/FatJet.Phi', 
                'fatjet_ptcs':'FatJet/FatJet.Particles', 'ptcs_pid':'Particle/Particle.PID', 'ptcs_e':'Particle/Particle.E',
                'ptcs_pt':'Particle/Particle.PT', 'ptcs_eta':'Particle/Particle.Eta', 'ptcs_phi':'Particle/Particle.Phi',
                'ptcs_px':'Particle/Particle.Px', 'ptcs_py':'Particle/Particle.Py', 'ptcs_pz':'Particle/Particle.Pz',
            }
            expressions = aliases.keys()
            events      = events.arrays(expressions=expressions, cut=None, aliases=aliases)
            events      = events[(ak.num(events['fatjet_pt']) > 0)]

            # find the fatjet with highest pt
            max_index = ak.firsts(ak.argsort(events[f'fatjet_pt'], ascending=False), axis=1)
            max_index = ak.unflatten(max_index, counts=ak.ones_like(max_index))
            events['fatjet_pt']  = ak.flatten(events['fatjet_pt'][max_index])
            events['fatjet_eta'] = ak.flatten(events['fatjet_eta'][max_index])
            events['fatjet_phi'] = ak.flatten(events['fatjet_phi'][max_index])

            # get daughters of the fatjet
            refs = events['fatjet_ptcs'][max_index].refs[:, 0] - 1
            events['fatjet_daughter_e']   = events['ptcs_e'][refs]
            events['fatjet_daughter_pt']  = events['ptcs_pt'][refs]
            events['fatjet_daughter_eta'] = events['ptcs_eta'][refs]
            events['fatjet_daughter_phi'] = events['ptcs_phi'][refs]
            events['fatjet_daughter_pid'] = events['ptcs_pid'][refs]
            events['fatjet_daughter_px']  = events['ptcs_px'][refs]
            events['fatjet_daughter_py']  = events['ptcs_py'][refs]
            events['fatjet_daughter_pz']  = events['ptcs_pz'][refs]

            # default fast pt, eta, phi (for subjet_radius=0 use)
            events['fast_pt']        = events['fatjet_daughter_pt']
            events['fast_delta_eta'] = events['fatjet_daughter_eta'] - events['fatjet_eta']
            events['fast_delta_phi'] = events['fatjet_daughter_phi'] - events['fatjet_phi']
            events['fast_delta_phi'] = np.mod(events['fast_delta_phi'] + np.pi, 2*np.pi) - np.pi

            # remove unnecessary records which contain 'ptcs' in field name
            remain_fields = [field for field in events.fields if 'ptcs' not in field]
            events        = events[remain_fields]

            # cut events that not in pt region by cut_pt
            if cut_pt != None:
                idx_in_cut_pt = (events['fatjet_pt']>=min(cut_pt)) * (events['fatjet_pt']<max(cut_pt))
                events        = events[idx_in_cut_pt]

            # finish loading fatjet events
            self.events  = events
            print(f"\n# DataLog: Successfully create {channel} with {len(events)} events.\n")

            # reclustering fastjet events
            if subjet_radius != 0:
                self.generate_fastjet_events(subjet_radius)
        
        # whether remain all particles in jets
        # default "Full"      -> all particles in jets remain
        # assign with a int N -> N particles with highest pt remain for each jet
        if num_pt_ptcs != "Full":
            self.generate_max_pt_events(num_pt_ptcs)

    def generate_fastjet_events(self, subjet_radius, algorithm=fastjet.antikt_algorithm):
        '''start reclustering particles into subjets'''
        print(f"\n# DataLog: Start reclustering {self.channel} with radius {subjet_radius}")
        fastjet_list = []
        for event in self.events:
            four_momentums = ak.Array({
                "px":event["fatjet_daughter_px"], 
                "py":event["fatjet_daughter_py"], 
                "pz":event["fatjet_daughter_pz"],
                "E" :event["fatjet_daughter_e"],
                })
            jet_def = fastjet.JetDefinition(algorithm, float(subjet_radius))
            cluster = fastjet.ClusterSequence(four_momentums, jet_def)
            fastjet_list.append(cluster.inclusive_jets())
            
        # the output will only be 4-momentum of newly clustered jets
        fastjet_events          = ak.Array(fastjet_list)
        fastjet_events["e"]     = fastjet_events["E"]
        fastjet_events["p"]     = (fastjet_events["px"]**2 + fastjet_events["py"]**2 + fastjet_events["pz"]**2) ** 0.5
        fastjet_events["pt"]    = (fastjet_events["px"]**2 + fastjet_events["py"]**2) ** 0.5
        fastjet_events["theta"] = np.arccos(fastjet_events["pz"]/fastjet_events["p"])
        fastjet_events["eta"]   = -np.log(np.tan((fastjet_events["theta"])/2))
        fastjet_events["phi"]   = np.arctan2(fastjet_events["py"], fastjet_events["px"])
        fastjet_events["delta_eta"] = fastjet_events["eta"] - self.events[f"fatjet_eta"]
        fastjet_events["delta_phi"] = fastjet_events["phi"] - self.events[f"fatjet_phi"]
        fastjet_events["delta_phi"] = np.mod(fastjet_events["delta_phi"] + np.pi, 2*np.pi) - np.pi
        
        # finish reclustering and merge with original events
        for field in fastjet_events.fields:
            self.events[f"fast_{field}"] = fastjet_events[field]
        print(f"# DataLog: Finish reclustering {self.channel} with radius {subjet_radius}\n")
    
    def generate_uniform_pt_events(self, bin, num_bin_data, log=False):
        '''randomly generate uniform events'''
        # determine the lower and upper limits of pt
        if self.cut_pt is not None:
            cut_pt = self.cut_pt
        else:
            cut_pt = (min(self.events['fatjet_pt']), max(self.events['fatjet_pt']))
        bin_interval = (max(cut_pt) - min(cut_pt)) / bin
        bin_list     = []
        
        for i in range(bin):
            # select the target bin range
            bin_lower    = min(cut_pt) + bin_interval * i
            bin_upper    = min(cut_pt) + bin_interval * (i+1)
            bin_selected = (self.events['fatjet_pt'] >= bin_lower) * (self.events['fatjet_pt'] < bin_upper)
            bin_events   = self.events[bin_selected]
            
            # randomly select uniform events
            assert num_bin_data <= len(bin_events), f"# DataLog: num_bin_data is not enough -> {num_bin_data} > {len(bin_events)}"
            idx = list(range(len(bin_events)))
            random.shuffle(idx)
            idx = idx[:num_bin_data]
            bin_list.append(bin_events[idx])
            if log:
                print(f"# DataLog: Generate uniform Pt events ({i+1}/{bin}) | number of bin events = {num_bin_data}/{len(bin_events)}")
        
        return ak.concatenate(bin_list)
    
    def generate_max_pt_events(self, num_pt_ptcs):
        '''retain num_pt_ptcs particles with highest pt'''
        max_arg = ak.argsort(self.events["fast_pt"], ascending=False, axis=1)
        max_arg = max_arg[:, :num_pt_ptcs]
        for field in self.events.fields:
            if "fast_" in field:
                self.events[field] = self.events[field][max_arg]


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class JetDataModule(pl.LightningDataModule):
    def __init__(self, sig_events, bkg_events, data_ratio, batch_size, graph):
        super().__init__()
        # whether transform to torch_geometric graph data
        self.graph = graph
        self.data_ratio = data_ratio
        self.batch_size = batch_size

        # jet events
        self.max_num_ptcs = max(
            max(ak.count(sig_events["fast_pt"], axis=1)),
            max(ak.count(bkg_events["fast_pt"], axis=1)))
        sig_events = self._preprocess(sig_events)
        bkg_events = self._preprocess(bkg_events)
        print(f"\n# DataLog: Max number of particles = {self.max_num_ptcs}\n")

        # prepare dataset for dataloader
        train_idx = int(data_ratio * len(sig_events))
        self.train_dataset = self._dataset(sig_events[:train_idx], 1) + self._dataset(bkg_events[:train_idx], 0)
        self.test_dataset  = self._dataset(sig_events[train_idx:], 1) + self._dataset(bkg_events[train_idx:], 0)

    def _preprocess(self, events):
        # "_" prefix means that it is a fastjet feature
        fatjet_radius = 0.8
        f1 = np.arctan(events["fast_pt"] / events["fatjet_pt"])
        f2 = events["fast_delta_eta"] / fatjet_radius * (np.pi/2)
        f3 = events["fast_delta_phi"] / fatjet_radius * (np.pi/2)
        arrays = ak.zip([f1, f2, f3])
        arrays = arrays.to_list()
        events = [torch.tensor(arrays[i], dtype=torch.float32, requires_grad=False) for i in range(len(arrays))]
        return events

    def _dataset(self, events, y):
        if self.graph == True:
            # create pytorch_geometric "Data" object
            dataset = []
            for i in range(len(events)):
                x = events[i]
                edge_index = list(product(range(len(x)), range(len(x))))
                edge_index = torch.tensor(edge_index, requires_grad=False).transpose(0, 1)
                dataset.append(Data(x=x, edge_index=edge_index, y=y))
        else:
            pad     = lambda x: torch.nn.functional.pad(x, (0,0,0,self.max_num_ptcs-len(x)), mode="constant", value=0)
            dataset = TorchDataset(x=[pad(events[i]) for i in range(len(events))], y=[y]*len(events))
        return dataset

    def train_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            return TorchDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            return TorchDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        if self.graph == True:
            return GeoDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            return TorchDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    
def save_hdf5(channel, data_info, ak_array):
    # see https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#saving-awkward-arrays-to-hdf5
    print(f"# DataLog: Start creating {channel}|{data_info}.hdf5 file")
    hdf5_file  = h5py.File(f"{os.path.dirname(__file__)}/jet_dataset/{channel}|{data_info}.hdf5", "w")
    hdf5_group = hdf5_file.create_group(channel)
    form, length, container    = ak.to_buffers(ak.to_packed(ak_array), container=hdf5_group)
    hdf5_group.attrs["form"]   = form.to_json()
    hdf5_group.attrs["length"] = json.dumps(length)
    print(f"# DataLog: Successfully creating {channel}|{data_info}.hdf5 file")

def load_hdf5(channel, data_info):
    # see https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#reading-awkward-arrays-from-hdf5
    hdf5_file  = h5py.File(f"{os.path.dirname(__file__)}/jet_dataset/{channel}|{data_info}.hdf5", "r")
    hdf5_group = hdf5_file[channel]
    ak_array   = ak.from_buffers(
        ak.forms.from_json(hdf5_group.attrs["form"]),
        json.loads(hdf5_group.attrs["length"]),
        {k: np.asarray(v) for k, v in hdf5_group.items()},
    )
    return ak_array

# if __name__ == "__main__":
    # # example for generating hdf5 file
    # subjet_radius = 0.25
    # channels = ["VzToQCD"]
    # channels = ["VzToTt"]
    # events = FatJetEvents(channel=channel, cut_pt=(800, 1000), subjet_radius=subjet_radius, check_hdf5=False)
    # save_hdf5(channel=channel, data_info=f"c{800}_{1000}_r{subjet_radius}", ak_array=events.events)