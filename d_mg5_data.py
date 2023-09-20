import os, random
import uproot, fastjet
import numpy as np
import awkward as ak
import h5py, json

# PID table
pdgid_table = {"electron": 11, "muon": 13, "gamma": 22, "ch_hadron": 211, "neu_hadron": 130, "HF_hadron": 1, "HF_em": 2,}

# run fastjet 1 time then it won't show up cite reminder
_jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 1.0)
_array   = ak.Array([{"px": 0.1, "py": 0.2, "pz": 0.3, "E": 0.4},])
fastjet.ClusterSequence(_array, _jet_def)

class FatJetEvents:
    def __init__(self, channel:str, cut_pt:tuple[float,float]=None, subjet_radius:float=None, num_pt_ptcs:int=None, check_hdf5:bool=True):
        '''Construct mg5 fatjet events with energy flow information'''
        self.channel       = channel
        self.cut_pt        = cut_pt
        self.subjet_radius = subjet_radius
        self.num_pt_ptcs   = num_pt_ptcs
        
        if check_hdf5 == True:
            data_info   = f"c{cut_pt[0]}_{cut_pt[1]}_r{subjet_radius}"
            print(f"DataLog: Now loading hdf5 file {channel}|{data_info}.hdf5")
            self.events = load_hdf5(channel, data_info)
            print(f"DataLog: Successfully loading hdf5 file {channel}|{data_info}.hdf5")
        else:
            # read MadGraph5 root file through 'uproot'
            dir_path  = f"{os.path.dirname(__file__)}/data"
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

            # remove unnecessary records (ptcs)
            remain_fields = [field for field in events.fields if 'ptcs' not in field]
            events        = events[remain_fields]
            if cut_pt != None:
                idx_in_cut_pt = (events['fatjet_pt']>=min(cut_pt)) * (events['fatjet_pt']<max(cut_pt))
                events        = events[idx_in_cut_pt]

            # finish loading fatjet events
            self.events  = events
            print(f"\nDataLog: Successfully create {channel} with {len(events)} events.\n")

            # reclustering fastjet events
            if subjet_radius is not None:
                self.generate_fastjet_events(subjet_radius)
        
        if num_pt_ptcs is not None:
            self.generate_max_pt_events()

    def generate_fastjet_events(self, subjet_radius, algorithm=fastjet.antikt_algorithm):
        # start reclustering particles into subjets
        print(f"\nDataLog: Start reclustering {self.channel} with radius {subjet_radius}")
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
        print(f"DataLog: Finish reclustering {self.channel} with radius {subjet_radius}\n")
    
    def generate_uniform_pt_events(self, bin, num_bin_data, num_ptcs_limit=None):
        # determine the lower and upper limits of pt
        cut_pt = self.cut_pt if self.cut_pt is not None else (min(self.events['fatjet_pt']), max(self.events['fatjet_pt']))
        bin_interval = (max(cut_pt) - min(cut_pt)) / bin
        bin_list     = []
        
        for i in range(bin):
            # select the target bin range
            bin_lower    = min(cut_pt) + bin_interval * i
            bin_upper    = min(cut_pt) + bin_interval * (i+1)
            bin_selected = (self.events['fatjet_pt'] >= bin_lower) * (self.events['fatjet_pt'] < bin_upper)
            # whether select events with num_ptcs <= num_ptcs_limit
            if num_ptcs_limit is not None:
                bin_selected = bin_selected * (ak.num(self.events["fast_pt"]) <= num_ptcs_limit)
            bin_events   = self.events[bin_selected]
            
            # randomly select uniform events
            assert num_bin_data <= len(bin_events), f"DataLog: num_bin_data is not enough -> {num_bin_data} > {len(bin_events)}"
            idx = list(range(len(bin_events)))
            random.shuffle(idx)
            idx = idx[:num_bin_data]
            bin_list.append(bin_events[idx])
            print(f"DataLog: Generate uniform Pt events ({i+1}/{bin}) | number of bin events = {num_bin_data}/{len(bin_events)}")
        
        return ak.concatenate(bin_list)
    
    def generate_max_pt_events(self):
        max_arg = ak.argsort(self.events["fast_pt"], ascending=False, axis=1)
        max_arg = max_arg[:, :self.num_pt_ptcs]
        for field in self.events.fields:
            if "fast_" in field:
                self.events[field] = self.events[field][max_arg]
        
    
def save_hdf5(channel, data_info, ak_array):
    # see https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#saving-awkward-arrays-to-hdf5
    print(f"DataLog: Start creating {channel}|{data_info}.hdf5 file")
    hdf5_file  = h5py.File(f"{os.path.dirname(__file__)}/data/{channel}|{data_info}.hdf5", "w")
    hdf5_group = hdf5_file.create_group(channel)
    form, length, container    = ak.to_buffers(ak.to_packed(ak_array), container=hdf5_group)
    hdf5_group.attrs["form"]   = form.to_json()
    hdf5_group.attrs["length"] = json.dumps(length)
    print(f"DataLog: Successfully creating {channel}|{data_info}.hdf5 file")

def load_hdf5(channel, data_info):
    # see https://awkward-array.org/doc/main/user-guide/how-to-convert-buffers.html#reading-awkward-arrays-from-hdf5
    hdf5_file  = h5py.File(f"{os.path.dirname(__file__)}/data/{channel}|{data_info}.hdf5", "r")
    hdf5_group = hdf5_file[channel]
    ak_array   = ak.from_buffers(
        ak.forms.from_json(hdf5_group.attrs["form"]),
        json.loads(hdf5_group.attrs["length"]),
        {k: np.asarray(v) for k, v in hdf5_group.items()},
    )
    return ak_array