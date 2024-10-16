# %%
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from allocation.allocate_fun import allocate
from tai_download import download_regdata
from tai_constants import DATADIR
from tai_config import load_config

DATESTR = datetime.now(timezone.utc).strftime(r"%Y%m%d")

# %%
# load the config
cfg = load_config()
regfile = cfg["regdata"]["path"] / cfg["regdata"]["file"]

if not regfile.exists():
    try: 
        download_regdata()
    except:
        raise IOError(f"{regfile} doesn't exist")

# load the registration data
if regfile.suffix == ".csv":
    regdata = pd.read_csv(regfile)
elif regfile.suffix == ".xlsx":
    regdata = pd.read_excel(regfile)

dist_to_usetown = regdata["dist_to_usetown"]

# %%
## set allocation parameters
allocate_params = cfg["allocate_params"]

max_size_difference = allocate_params["max_size_difference"]
balance_histogram = allocate_params["balance_histogram"]
balance_mean = allocate_params["balance_mean"]
balance_pairwise = allocate_params["balance_pairwise"]
max_unique_linked = allocate_params["max_unique_linked"]

dthresh = np.inf
must_be_in_group = {}
for d_to_usetown, (val, (_, row)) in zip(
    dist_to_usetown, enumerate(regdata.iterrows())
):
    mbg = []
    if d_to_usetown > dthresh:
        mbg.extend([0, 1, 3])
    if mbg:
        must_be_in_group[val] = sorted(set(mbg))
group_objective = {2: (dist_to_usetown - dist_to_usetown.max()) - 1}
must_allocate_all = True

## run allocation algorithm
x = allocate(
    regdata,
    linkvar="schid_clean",
    ngroups=4,
    max_size_difference=max_size_difference,
    balance_histogram=balance_histogram,
    must_be_in_group=must_be_in_group,
    group_objective=group_objective,
    balance_mean=balance_mean,
    must_allocate_all=must_allocate_all,
    balance_pairwise=balance_pairwise,
    max_unique_linked=max_unique_linked,
)
groupdfs = [regdata.loc[g > 0].copy() for g in x]

## print logs
logfile = DATADIR / f"group_assignments_{DATESTR}_descriptives.txt"
with open(logfile, "w") as logf:
    for i, gdf in enumerate(groupdfs):
        # ReDF.from_df(gdf).scatter(shape=shapedata, ax=ax, scale=20)
        print(f"Group: {i+1 :d}\n", file=logf)
        linked_contam = any(
            gdf["schid_clean"].isin(gdf_["schid_clean"]).sum()
            for i_, gdf_ in enumerate(groupdfs)
            if i_ != i
        )
        print(f"Linked contamination: {linked_contam}", file=logf)
        print(f"Group size: {len(gdf)}", file=logf)
        print(f"N Unique Linked: {len(gdf['schid_clean'].unique())}", file=logf)
        for var in balance_histogram:
            print(gdf.groupby(var)[var].agg(len), file=logf)
            print(file=logf)
        for var in balance_mean:
            print(f"Mean({var}): {gdf[var].mean()}", file=logf)
            print(file=logf)
        print("\n\n", file=logf)

## format output
for i, df in enumerate(groupdfs):
    df["Group"] = i + 1

combodf = pd.concat(groupdfs, axis=0, ignore_index=True)

# %%
## save result
outname = DATADIR / f"group_assignments_{DATESTR}.xlsx"
combodf.to_excel(outname, index=False)
