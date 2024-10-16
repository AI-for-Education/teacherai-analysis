from pathlib import Path

import yaml

from tai_constants import CFG_PATH, DATADIR

def load_config(cfg_path = CFG_PATH):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for data_key in ["regdata", "chatdata"]:
        if cfg[data_key]["path"] == "$DATA":
            cfg[data_key]["path"] = DATADIR
        else:
            cfg[data_key]["path"] = Path(cfg[data_key]["path"])
    
    return cfg