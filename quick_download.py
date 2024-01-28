"""Download required datasets and pretrained models through `gdown`.

This python script require `pip install gdown` first, then can download
some dataset and pretrained model from google drive just using python.
"""

import os
import zipfile

import gdown


def gdrive_download(dir_name, url):
    """Download files (zip files) from Google Drive."""

    # Download zip file.
    pwd = os.path.dirname(__file__)
    zip_path = os.path.join(pwd, f"{dir_name}.zip")
    gdown.download(url=url, output=zip_path, quiet=False, fuzzy=True)

    # Extract it and remove it.
    print(f"Extracting {zip_path}.")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(pwd)
    os.remove(zip_path)
    print(f"{zip_path} removed.")


# `pretrain_ckpt` (pretrain models).
url_pretrain_ckpt = "https://drive.google.com/file/d/1DcC2WetceAOkTg5HGoUdzUrBkFtMlYKk/view?usp=sharing"
gdrive_download(dir_name="pretrain_ckpt", url=url_pretrain_ckpt)

# `jet_dataset_0` (jet dataset without reclustering).
url_jet_dataset = "https://drive.google.com/file/d/1FP_SOqcbStRfvXim-wXEdh1-VVxRjKuF/view?usp=sharing"
gdrive_download(dir_name="jet_dataset", url=url_jet_dataset)
