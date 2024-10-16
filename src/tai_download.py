import gzip
from io import BytesIO
import json

import requests

from tai_constants import CFG_PATH
from tai_config import load_config


def download_regdata(cfg_path=CFG_PATH):
    return _download_data("regdata", cfg_path=cfg_path)


def download_chatdata(cfg_path=CFG_PATH):
    return _download_data("chatdata", cfg_path=cfg_path)


def _download_data(data_key, cfg_path=CFG_PATH):

    cfg = load_config(cfg_path)
    if cfg[data_key]["dl_type"] is not None:
        suffix = cfg[data_key]["dl_type"]
    else:
        suffix = ""
    dl_file = cfg[data_key]["dl_path"] + "/" + (cfg[data_key]["file"] + suffix)
    out_file = cfg[data_key]["path"] / cfg[data_key]["file"]
    out_file.parent.mkdir(exist_ok=True, parents=True)

    try:
        response = requests.get(dl_file)
    except:
        raise requests.HTTPError()

    if response.status_code == 200:
        content = BytesIO(response.content)

        with gzip.open(content, mode="rb") as zip:
            with open(out_file, "wb") as f:
                f.write(zip.read())
    else:
        raise requests.HTTPError(f"status code: {response.status_code}")
