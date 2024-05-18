# Jet Discrimination with Quantum Complete Graph Neural Network

The source code for [arXiv:2403.04990](https://arxiv.org/abs/2403.04990). The main python script for training is written in `main.ipynb`, with other modules in `src/`.

---

### Installation

##### Setup environment via `setup.py`
The main training workflow is based on [PyTorch](https://pytorch.org) and [PennyLane](https://pennylane.ai), and Python 3.9 or newer is required. You can easily install the required packages with:
```bash
pip install . # Will install automatically from `setup.py`

# Optional
rm -rf build QCGNN.egg-info
```

> **Warning**: It is highly recommended to install `PyTorch` independently, see [official website](https://pytorch.org) for corresponding installation.

> Note: The version of `PennyLane` should be newer than `0.31.0` due to `qml.qnn.TorchLayer` [issue](https://discuss.pennylane.ai/t/inputs-dimension-mix-with-batch-dimension-in-qml-qnn-torchlayer/3824/8).

##### Jupyter Environment (optional)
Some of python scripts are written in [Jupyter](https://jupyter.org) format (`.ipynb`). For user who wants to run with traditional python scripts (`.py`), you can download `ipynb-py-convert` package via:
```bash
# /bin/bash
pip install ipynb-py-convert
ipynb-py-convert some_notebook_file.ipynb to_py_file.py
python to_py_file.py
```

---

### MISC

##### Data description in `./dataset`
- `VzToQCD`: 1-prong light quark QCD jets.
- `VzToZhToVevebb`: 2-prong jets from $H\rightarrow b\bar{b}$.
- `VzToTt`: 3-prong jets from $t\rightarrow bW^+$.
- `pt_800_1000`: Jets with transverse momentum $p_T$ in range $(800,1000)$.

##### Generating original jet data by MG5
The data in `./dataset` is generated through [MadGraph5_aMC@NLO](https://launchpad.net/mg5amcnlo) with [Heavy Vector Triplets (HVT)](https://hepmdb.soton.ac.uk/index.php?mod=user&act=showmodel&id=0214.0151) model:
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
