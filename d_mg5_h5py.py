import d_mg5_data

channels = ["VzToZhToVevebb", "VzToQCD"]

for channel in channels:
    events = d_mg5_data.FatJetEvents(channel=channel, cut_pt=(800, 1000), subjet_radius=0.1)
    d_mg5_data.save_h5py(channel_info=f"{channel}_c{800}_{1000}_r_{0.1}", ak_array=events.events)