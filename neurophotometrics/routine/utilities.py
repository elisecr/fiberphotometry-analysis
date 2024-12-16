import itertools as itt

import numpy as np
import pandas as pd
import pandas.api.types as pdt


def cut_df(df, nrow, sortby="SystemTimestamp"):
    return df.sort_values(sortby).iloc[:nrow]


def exp2(x, a, b, c, d, e):
    return a * np.exp(b * x) + c * np.exp(d * x) + e


def load_data(data_file, discard_time, led_dict, roi_dict):
    if isinstance(data_file, pd.DataFrame):
        data = data_file
    else:
        data = pd.read_csv(data_file)
    data = data[
        data["SystemTimestamp"] > data["SystemTimestamp"].min() + discard_time
    ].copy()
    data["signal"] = data["LedState"].map(led_dict)
    nfm = data.groupby("signal").size().min()
    data = (
        data.groupby("signal", group_keys=False)
        .apply(cut_df, nrow=nfm)
        .reset_index(drop=True)
        .rename(columns=roi_dict)
    )
    return data


def load_ts(ts):
    ts = df_to_numeric(ts)
    if len(ts.columns) == 2:
        if pdt.is_object_dtype(ts[0]) and pdt.is_float_dtype(ts[1]):
            ts.columns = ["event", "ts"]
            ts["event_type"] = "keydown"
            ts_type = "ts_keydown"
        elif pdt.is_integer_dtype(ts[0]) and pdt.is_float_dtype(ts[1]):
            ts.columns = ["fm_behav", "ts"]
            ts_type = "ts_behav"
        elif pdt.is_integer_dtype(ts[0]) and (
            pdt.is_object_dtype(ts[1]) or pdt.is_bool_dtype(ts[1])
        ):
            ts.columns = ["fm_behav", "event"]
            ts["event_type"] = "user"
            ts_type = "ts_events"
        else:
            raise ValueError("Don't know how to handle TS")
    elif len(ts.columns) == 5:
        if ts.iloc[0, 0] == "DigitalIOName":
            ts = ts.iloc[1:].copy()
        ts = df_to_numeric(ts)
        ts.columns = ["io_name", "io_flag", "io_state", "ts_fp", "ts"]
        ts["event"] = (
            ts["io_name"].astype(str)
            + "-"
            + ts["io_flag"].astype(str)
            + "-"
            + ts["io_state"].astype(str)
        )
        ts["event_type"] = "arduino"
        ts_type = "ts_arduino"
    else:
        raise ValueError("Don't know how to handle TS")
    return ts, ts_type


def enumerated_product(*args):
    yield from zip(itt.product(*(range(len(x)) for x in args)), itt.product(*args))


def df_to_numeric(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def compute_fps(
    df, fm_col="fm_fp", tcol="ts_fp", ledcol="LedState", mul_fac=1, precision=2
):
    nled = (df[ledcol].count() > 5).sum()
    mdf = df[tcol].diff().mean()
    mfm = df[fm_col].diff().mean()
    return round(float(mfm / mdf * mul_fac * nled), precision)
