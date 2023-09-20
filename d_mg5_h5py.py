import d_mg5_data

generate_new_h5py_file = True
# subjet_radius = 0.25
# channels = ["VzToQCD"]
channels = ["VzToZhToVevebb", "VzToQCD"]

if generate_new_h5py_file:
    for subjet_radius in [None]:
        for channel in channels:
            events = d_mg5_data.FatJetEvents(channel=channel, cut_pt=(800, 1000), subjet_radius=subjet_radius, check_hdf5=False)
            d_mg5_data.save_hdf5(channel=channel, data_info=f"c{800}_{1000}_r{subjet_radius}", ak_array=events.events)