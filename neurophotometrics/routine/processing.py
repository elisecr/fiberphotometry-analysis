import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import zscore
from sklearn.linear_model import HuberRegressor

from .utilities import exp2


def photobleach_correction(data, baseline_sig, rois=None):
    # auto set rois
    if rois is None:
        rois = list(
            set([r[0] for r in baseline_sig.keys()])
            | set([r[0] for r in baseline_sig.values()])
        )
    # making sure signal exists
    base_dict = dict()
    for (roi, sig), (base_roi, base_sig) in baseline_sig.items():
        if len(data.loc[data["signal"] == sig, roi]) == 0:
            continue
        if len(data.loc[data["signal"] == base_sig, base_roi]) == 0:
            warnings.warn(
                "Cannot find signal '{}' in roi '{}', skipping correction for signal '{}' roi '{}'".format(
                    base_sig, base_roi, sig, roi
                )
            )
            continue
        base_dict[(roi, sig)] = (base_roi, base_sig)
    # fit baseline
    base_dfs = dict()
    for base_roi, base_sig in set(base_dict.values()):
        if base_sig in base_dfs:
            base_df = base_dfs[base_sig]
        else:
            base_df = data.loc[data["signal"] == base_sig].copy()
            base_df["signal"] = base_sig + "-fit"
            base_df[rois] = np.nan
        dat_fit = data.loc[data["signal"] == base_sig, base_roi]
        x = np.linspace(0, 1, len(dat_fit))
        base_fit = fit_exp2(dat_fit, x)
        base_df[base_roi] = base_fit
        base_dfs[base_sig] = base_df
    # correct signals
    sig_dfs = dict()
    for (roi, sig), (base_roi, base_sig) in base_dict.items():
        if sig in sig_dfs:
            sig_df = sig_dfs[sig]
            norm_df = sig_df.loc[sig_df["signal"] == sig + "-norm"].copy()
            zs_df = sig_df.loc[sig_df["signal"] == sig + "-norm-zs"].copy()
        else:
            norm_df = data.loc[data["signal"] == sig].copy()
            norm_df["signal"] = sig + "-norm"
            norm_df[rois] = np.nan
            zs_df = data.loc[data["signal"] == sig].copy()
            zs_df["signal"] = sig + "-norm-zs"
            zs_df[rois] = np.nan
        dat_sig = data.loc[data["signal"] == sig, roi]
        baseline = np.array(base_dfs[base_sig][base_roi])
        model = HuberRegressor()
        model.fit(baseline.reshape((-1, 1)), dat_sig)
        norm_df[roi] = dat_sig - model.predict(baseline.reshape((-1, 1)))
        zs_df[roi] = zscore(norm_df[roi])
        sig_dfs[sig] = pd.concat([norm_df, zs_df])
    data_norm = pd.concat(
        [data] + list(base_dfs.values()) + list(sig_dfs.values()), ignore_index=True
    )
    return data_norm


def fit_exp2(a, x):
    dmax, dmin = a[:50].median(), a[-50:].median()
    drg = dmax - dmin
    p0 = (drg, -10, drg, 0.1, dmin - drg)
    try:
        popt, pcov = curve_fit(exp2, x, a, p0=p0, method="trf", ftol=1e-6, maxfev=1e4)
    except:
        warnings.warn("Biexponential fit failed")
        popt = p0
    return exp2(x, *popt)


def compute_dff(data, rois, sigs=["415nm", "470nm"]):
    if sigs is not None:
        data = data[data["signal"].isin(sigs)].copy()
    res_ls = []
    for sig, dat_sig in data.groupby("signal"):
        dat_fit = dat_sig.copy()
        dat_dff = dat_sig.copy()
        dat_fit["signal"] = sig + "-fit"
        dat_dff["signal"] = sig + "-dff"
        x = np.linspace(0, 1, len(dat_sig))
        for roi in rois:
            dat = dat_sig[roi]
            popt, pcov = curve_fit(
                exp2,
                x,
                dat,
                p0=(1.0, 0, 1.0, 0, dat.mean()),
                bounds=(
                    np.array([-np.inf, -np.inf, -np.inf, -np.inf, dat.min()]),
                    np.array([np.inf, np.inf, np.inf, np.inf, dat.max()]),
                ),
            )
            cur_fit = exp2(x, *popt)
            dat_fit[roi] = cur_fit
            dat_dff[roi] = 100 * (dat - cur_fit) / cur_fit
        res_ls.extend([dat_fit, dat_dff])
    return pd.concat([data] + res_ls, ignore_index=True)


def find_pks(data, rois, prominence, freq_wd=None, sigs=None):
    for sig, dat_sig in data.groupby("signal"):
        if sigs is not None and sig in sigs:
            for roi in rois:
                dat = dat_sig[roi]
                pks, props = find_peaks(dat, prominence=prominence)
                pvec = np.zeros_like(dat, dtype=bool)
                pvec[pks] = 1
                data.loc[dat_sig.index, roi + "-pks"] = pvec
                if freq_wd is not None:
                    dat_sig.loc[dat_sig.index, roi + "-freq"] = (
                        dat_sig.loc[dat_sig.index, roi + "-pks"].rolling(freq_wd).sum()
                    )
    return data
