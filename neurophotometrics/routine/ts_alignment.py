import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import label

from .utilities import load_ts

BEHAV_LS = ["G", "D", "O", "M", "F", "N", "R"]


def align_ts(data, ts_files, behav_ls=BEHAV_LS) -> None:
    data = data.rename(
        columns={
            "SystemTimestamp": "ts_fp",
            "FrameCounter": "fm_fp",
            "ComputerTimestamp": "ts",
        }
    )
    # load input ts
    ts_dict = dict()
    for dname, dat in ts_files.items():
        dat, ts_type = load_ts(dat.copy())
        print("Interpreting {} as {}".format(dname, ts_type))
        if ts_type == "ts_behav" or ts_type == "ts_fp":
            if ts_type in ts_dict.keys():
                raise ValueError(
                    "Multiple {} supplied but only one expected.".format(ts_type)
                )
            ts_dict[ts_type] = dat
        else:
            ts_dict[dname] = dat
    # add ts to fm_fp
    if "ts_fp" in ts_dict:
        ts_fp = ts_dict.pop("ts_fp")
        fm_diff = int(ts_fp["fm_fp"].max()) - int(data["fm_fp"].max())
        if fm_diff != 0:
            diff_txt = (
                "{} frames more".format(abs(fm_diff))
                if fm_diff > 0
                else "{} frames less".format(abs(fm_diff))
            )
            warnings.warn("FP timestamp file has {} than data file".format(diff_txt))
        data = data.merge(ts_fp, on="fm_fp", how="left", validate="one_to_one")
    # align fm_behav basd on ts
    if "ts_behav" in ts_dict:
        ts_behav = ts_dict.pop("ts_behav")
        if "ts" in data.columns:
            ts_behav = pd.merge_asof(
                ts_behav, data[["fm_fp", "ts"]], on="ts", direction="nearest"
            ).rename(columns={"ts": "ts_behav"})
            ts_behav_dup = ts_behav[ts_behav["fm_fp"].duplicated(keep=False)]
            if len(ts_behav_dup) > 0:
                warnings.warn(
                    "Multiple Behavior frames mapped to the same FP frame\n"
                    + str(ts_behav_dup)
                )
            data = (
                data.merge(ts_behav, on="fm_fp", how="outer")
                .sort_values("fm_fp")
                .reset_index(drop=True)
            )
        else:
            warnings.warn("No FP TS supplied, cannot align Behavior frames")
    # align custom events
    evts = []
    for dname, dat in ts_dict.items():
        if "fm_fp" in dat.columns:
            evts.append(dat[["fm_fp", "event", "event_type"]])
        elif "fm_behav" in dat.columns:
            if "fm_behav" not in data.columns:
                warnings.warn(
                    "No Behavior frames supplied, cannot align {}".format(dname)
                )
                continue
            dat = dat.merge(data[["fm_fp", "fm_behav"]], on="fm_behav", how="left")
            evts.append(dat[["fm_fp", "fm_behav", "event", "event_type"]])
        elif "ts_fp" in dat.columns:
            if "ts_fp" not in data.columns:
                warnings.warn(
                    "No FP timestamps supplied, cannot align {}".format(dname)
                )
                continue
            dat = pd.merge_asof(
                dat, data[["fm_fp", "ts_fp"]], on="ts_fp", direction="nearest"
            )
            evts.append(dat[["fm_fp", "ts_fp", "event", "event_type"]])
        elif "ts" in dat.columns:
            if "ts" not in data.columns:
                warnings.warn(
                    "No computer timestamps supplied, cannot align {}".format(dname)
                )
                continue
            dat = pd.merge_asof(
                dat, data[["fm_fp", "ts"]], on="ts", direction="nearest"
            )
            evts.append(dat[["fm_fp", "ts", "event", "event_type"]])
        print("aligned {}".format(dname))
    if not len(evts) > 0:
        return data, ts_dict
    evts = pd.concat(evts, ignore_index=True).sort_values("fm_fp")
    if behav_ls is not None:
        evts = evts[evts["event"].isin(behav_ls)].copy()
    evts_dup = evts[evts["fm_fp"].duplicated(keep=False)]
    if len(evts_dup) > 0:
        warnings.warn("Multiple events mapped to the same FP frame\n" + str(evts_dup))
    data = (
        data.merge(evts[["fm_fp", "event", "event_type"]], on="fm_fp", how="outer")
        .sort_values("fm_fp")
        .reset_index(drop=True)
    )
    return data, ts_dict


def label_bout(data, name) -> pd.DataFrame:
    lb, nlb = label(data[name])
    data[name + "_label"] = lb
    return data
