import d_mg5_data

generate_new_h5py_file = True
# subjet_radius = 0.25
# channels = ["VzToQCD"]
channels = ["VzToTt"]

if generate_new_h5py_file:
    for subjet_radius in [0, 0.1, 0.2, 0.3, 0.4]:
        for channel in channels:
            events = d_mg5_data.FatJetEvents(channel=channel, cut_pt=(800, 1000), subjet_radius=subjet_radius, check_hdf5=False)
            d_mg5_data.save_hdf5(channel=channel, data_info=f"c{800}_{1000}_r{subjet_radius}", ak_array=events.events)