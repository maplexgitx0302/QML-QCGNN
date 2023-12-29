# QCGNN
Jet Discrimination with Quantum Complete Graph Neural Network

---

### Prerequisites
The code is fully written in **Python** environment.

##### Python packages
The main training workflow is based on [PyTorch](https://pytorch.org) and [Pennylane](https://pennylane.ai), and Python 3.9 or newer is required. To succesfully reproduce the output, install the packages through
```bash
# /bin/bash
pip install -r requirements.txt
```

> Note: The version of Pennylane should be newer than `0.31.0` due to `qml.qnn.TorchLayer` [issue](https://discuss.pennylane.ai/t/inputs-dimension-mix-with-batch-dimension-in-qml-qnn-torchlayer/3824/8).

##### Dataset
The jet dataset can be downloaded from [jet_dataset](https://drive.google.com/drive/folders/1cY__Pj9Rf2n7a8ErMRzOcIppd40MWIEC?usp=share_link) to local directory `./jet_dataset`

##### Pre-trained model
The pre-trained model parameters can be loaded from `ckpt` files and can be downloaded from [ckpt](https://drive.google.com/drive/folders/1cY__Pj9Rf2n7a8ErMRzOcIppd40MWIEC?usp=share_link) to local directory `./ckpt`.

##### Jupyter
The main python scripts are written in jupyter format (`.ipynb`), for user who wants to use traditional python scripts (`.py`), you can download `ipynb-py-convert` package:
```bash
# /bin/bash
pip install ipynb-py-convert
ipynb-py-convert g_main.ipynb a.py
python a.py
```

### Quick Start
##### Training and Prediction
All main procedure can be executed through `./g_main.ipynb`, and most of the configuration can be setup in `./config.json`. You can simply uncomment the code you want to test in the last two sections (*Training* and *Prediction*).

- Training
    - **Uncomment** and modify the model hyperparameters you want to train, also set up the random seeds if needed.
    - The dataset hyperparameters can be set in `data_config` dictionaries. 
    - The default training mode is **quick start mode**, i.e., trained with `max_epochs=1` and `num_bin_data=1`, no matter whatever you set in `./config.json`. Set `config["quick_start"]=false` in `./config.json` for full training.

- Prediction
  - Make sure you have the `ckpt` files placed correctly, e.g., `./ckpt/MODEL_DESCRIPTION/checkpoints/EPOCH-STEP.ckpt`, or simply download the pretrained models from [ckpt](https://drive.google.com/drive/folders/1cY__Pj9Rf2n7a8ErMRzOcIppd40MWIEC?usp=share_link).
  - **Uncomment** and modify the code you want to test.

### File descriptions
- Files with prefix `./module_`: Modules for detail of reading/loading data, constructing models, training procedures.
- Files with prefix `./demo_`: Demo for reading out or extracting the information of data, presenting in historgrams or other plots.
- Files with prefix `./result_`: Related to getting training results.
- Files in `./jet_dataset`:
  - `VzToQCD`: 1-prong light quark QCD jets.
  - `VzToZhToVevebb`: 2-prong jets from $H\rightarrow b\bar{b}$.
  - `VzToTt`: 3-prong jets from $t\rightarrow bW^+$.
  - `c800_1000`: Jets with transverse momentum $p_T$ in range $(800,1000)$.
  - `r0.x`: Jets reclustered through *anti-$k_T$* with subjet radius $0.x$ (default using `r0`)

### Generating original jet data by MG5
The data in `./jet_dataset` is generated through [MadGraph5_aMC@NLO](https://launchpad.net/mg5amcnlo) with [Heavy Vector Triplets (HVT)](https://hepmdb.soton.ac.uk/index.php?mod=user&act=showmodel&id=0214.0151) model:
```bash
import HVT

# VzToQCD
generate p p > vz > j j

# VzToZhToVevebb
generate p p > vz > z h, z > ve ve~, h > b b~

# VzToTt
generate p p > vz > t t~, (t > b w+, w+ > j j), (t~ > b~ w-, w- > j j)
```