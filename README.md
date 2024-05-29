# Jet Discrimination with Quantum Complete Graph Neural Network

The source code for [arXiv:2403.04990](https://arxiv.org/abs/2403.04990). The main python script for training is written in `main.ipynb`, with other modules in `source/`.

---

### Installation

##### Setup environment via `setup.py`
The main training workflow is based on [PyTorch](https://pytorch.org) and [PennyLane](https://pennylane.ai), and python 3.9 or newer is required. You can easily install the required packages with:
```bash
# Install with `setup.py`
pip install .

# Optional
rm -rf build QCGNN.egg-info
```

> Note: The version of `PennyLane` should be newer than `0.31.0` due to `qml.qnn.TorchLayer` [issue](https://discuss.pennylane.ai/t/inputs-dimension-mix-with-batch-dimension-in-qml-qnn-torchlayer/3824/8).

##### Jupyter Environment (optional)
Some of python scripts (notebooks) are written in [Jupyter](https://jupyter.org) format (`.ipynb`). For user who wants to run with traditional python scripts (`.py`), you can download `ipynb-py-convert` package via:
```bash
# /bin/bash
pip install ipynb-py-convert
ipynb-py-convert some_notebook_file.ipynb to_py_file.py
python to_py_file.py
```

---

### Dataset
We demonstrate the feasibility of each model with two different Monte Carlo datasets below. Both contain energy flow information of fatjets (R = 0.8) initiated from different particles, and can be downloaded from:

- [JetNet Dataset](https://zenodo.org/records/6975118): Used for multi-class classification task, with ~1 TeV fat-jets coming from gluon, light-quark, top-quark, W-boson and Z-boson. See [2106.11535](https://arxiv.org/abs/2106.11535) for further detail. **The doanloaded `hdf5` files should be placed in `./dataset/jetnet`.**

- [Top Quark Tagging](https://zenodo.org/records/2603256): Used for binary classification task, with fatjet momentum in [550, 650] GeV. **The doanloaded `hdf5` files should be placed in `./dataset/top`.**
