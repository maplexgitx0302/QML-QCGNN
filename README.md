# QCGNN

Jet Discrimination with Quantum Complete Graph Neural Network
- [Google Drive](https://drive.google.com/drive/folders/1cY__Pj9Rf2n7a8ErMRzOcIppd40MWIEC?usp=share_link) (Dataset, pretained models, etc.)

---

# Prerequisites
The code is fully written in **Python** environment.

### Python packages
The main training workflow is based on [PyTorch](https://pytorch.org) and [Pennylane](https://pennylane.ai), and Python 3.9 or newer is required. To succesfully reproduce the output, install the packages through
```bash
# /bin/bash
# It is highyly recommended to install `PyTorch` seperately (see warning below).
pip install -r requirements.txt
```

> **Warning**: It is highly recommended to install `PyTorch` independently, see [official website](https://pytorch.org) for corresponding installation.

> Note: The version of `PennyLane` should be newer than `0.31.0` due to `qml.qnn.TorchLayer` [issue](https://discuss.pennylane.ai/t/inputs-dimension-mix-with-batch-dimension-in-qml-qnn-torchlayer/3824/8).

> Note: The [fastjet](https://fastjet.readthedocs.io/en/latest/) package might not work in some system (e.g. Apple M1 above), ignore it if needed. The default code will not use it unless you need to recluster the particles into subjets.

### Dataset & Pre-trained models

- Quick download (through `gdown`):
  1. Make sure `gdown` is installed (```pip install gdown```)
  2. Automatically download through command line ```python quick_download.py```, this will create `./jet_dataset` and `./pretrain_ckpt` directories.

- Links to dataset and pre-trained models (optional):
  - The full jet dataset can be downloaded from [jet_dataset](https://drive.google.com/drive/folders/1i0wG-YqQr4hbMl4SNnhKbOK0UB_aHWGw?usp=share_link) to local directory `./jet_dataset`

  - The pre-trained model parameters can be loaded from `ckpt` (checkpoints) files and can be downloaded from [pretrain_ckpt](https://drive.google.com/drive/folders/1yAEV5jiHGTpHcaPzBhBPZNguxWcGs_kI?usp=share_link) to local directory `./pretrain_ckpt`.

### Jupyter Environment (optional)
The main python scripts are written in jupyter format (`.ipynb`), for user who wants to use traditional python scripts (`.py`), you can download `ipynb-py-convert` package:
```bash
# /bin/bash
pip install ipynb-py-convert
ipynb-py-convert g_main.ipynb g_main.py
python g_main.py
```

# Quick Guide
### Training
All training procedure can be executed through `./g_main.ipynb` (or `g_main.py`), and most of the configuration can be setup in `./config.json`. You can simply uncomment the code you want to test in the last section (*Training Cell*).

- **Uncomment** and modify the model hyperparameters you want to train, also set up the random seeds if needed.
- The dataset hyperparameters can be set in `./config.json`, for detail usage, see function `generate_data_config` in `./g_main.ipynb` (or `./g_main.py`).
- The default training mode is **quick start mode**, i.e., trained with `max_epochs=1` and `num_bin_data=1`, independent of settings in `./config.json`. Set `config["quick_start"]=false` in `./config.json` for full training.

### Run on IBMQ device (backend)
To run with IBMQ real devices, see `./g_ibmq.ipynb` for detail.
- Make sure you have the `pretrain_ckpt` files placed correctly, e.g., `./pretrain_ckpt/MODEL_DESCRIPTION/checkpoints/EPOCH-STEP.ckpt`, or simply download the pretrained models from [pretrain_ckpt](https://drive.google.com/drive/folders/1yAEV5jiHGTpHcaPzBhBPZNguxWcGs_kI?usp=share_link).
- You can send the email when finish running on IBMQ device, see `./module_gmail.py` and [gmail notification tutorial](https://www.youtube.com/watch?v=g_j6ILT-X0k) (default turned on if `gmail.json` file exists).

> In case you trained your own model, the pretrained `ckpt` will be stored in `./training_logs`, move the corresponding directories to `./pretrain_ckpt` so that when running on IBMQ, the pretrained model can be loaded.

---

# MISC

### File descriptions
- Files with prefix `./module_`: Modules for detail of reading/loading data, constructing models, training procedures.
- Files with prefix `./demo_`: Demo for reading out or extracting the information of data, comparing encodings, etc.
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
# In MadGraph5_aMC command line interface
import HVT

# VzToQCD
generate p p > vz > j j

# VzToZhToVevebb
generate p p > vz > z h, z > ve ve~, h > b b~

# VzToTt
generate p p > vz > t t~, (t > b w+, w+ > j j), (t~ > b~ w-, w- > j j)
```