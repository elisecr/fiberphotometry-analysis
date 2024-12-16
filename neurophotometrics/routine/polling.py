import numpy as np
import pandas as pd


def poll_events(
    data,
    evt_range,
    rois,
    evt_sep=None,
    evt_duration=None,
    evt_ts="ts_fp",
    norm=True,
):
    assert "event" in data.columns, "Please align event timestamps first!"
    assert "fm_fp" in data.columns, "Missing photometry frame index"
    evt_idx = data[data["event"].notnull()].index
    data.loc[evt_idx, "event"] = data.loc[evt_idx, "event"].astype(str)
    if evt_ts is not None:
        assert (
            evt_ts in data.columns
        ), "column '{}' not found in data, cannot find bout".format(evt_ts)
        data["evt_id"] = np.nan
        data["evt_id"] = data["evt_id"].astype("object")
        for evt, evt_df in data.loc[evt_idx].groupby("event"):
            tdiff = evt_df[evt_ts].diff().fillna(evt_sep + 1)
            sep = tdiff >= evt_sep
            sep_idx = sep[sep].index
            for start_idx, end_idx in zip(
                sep_idx, np.append(sep_idx[1:] - 1, evt_df.index[-1])
            ):
                seg = evt_df.loc[start_idx:end_idx]
                if seg[evt_ts].max() - seg[evt_ts].min() >= evt_duration:
                    data.loc[seg.index[0] : seg.index[-1], "evt_id"] = (
                        evt + "-" + seg["fm_fp"].min().astype(str)
                    )
    else:
        data.loc[evt_idx, "evt_id"] = (
            data.loc[evt_idx, "event"] + "-" + data.loc[evt_idx, "fm_fp"].astype(str)
        )
    max_fm = data["fm_fp"].max()
    evt_df = []
    for evt_id, seg in data[data["evt_id"].notnull()].groupby("evt_id"):
        fms = np.array(seg["fm_fp"])
        fm0, fm1 = fms[0], fms[-1]
        fm_range = tuple(
            (np.array([fm0 + evt_range[0], fm1 + evt_range[1]])).clip(0, max_fm)
        )
        dat_sub = data[data["fm_fp"].between(*fm_range)].copy()
        dat_sub["fm_evt"] = dat_sub["fm_fp"] - fm0
        dat_sub["event_type"] = seg["event_type"].dropna().unique().item()
        dat_sub["event"] = seg["event"].dropna().unique().item()
        dat_sub["evt_id"] = evt_id
        dat_sub.loc[dat_sub["fm_fp"] < fm0, "evt_phase"] = "before"
        dat_sub.loc[dat_sub["fm_fp"].between(fm0, fm1), "evt_phase"] = "during"
        dat_sub.loc[dat_sub["fm_fp"] >= fm1, "evt_phase"] = "after"
        if norm:
            for roi in rois:
                mean = dat_sub.loc[dat_sub["fm_evt"] < 0, roi].mean()
                std = dat_sub.loc[dat_sub["fm_evt"] < 0, roi].std()
                if std > 0:
                    dat_sub[roi] = (dat_sub[roi] - mean) / std
                else:
                    dat_sub[roi] = 0
        evt_df.append(dat_sub)
    evt_df = pd.concat(evt_df, ignore_index=True)
    return evt_df


def agg_polled_events(data, rois, ts_col="ts_fp"):
    agg_df = []
    for (evt_type, evt, evt_id, evt_phase), evtdat in data.groupby(
        ["event_type", "event", "evt_id", "evt_phase"]
    ):
        for roi in rois:
            agg_dat = pd.Series(
                {
                    "event_type": evt_type,
                    "event": evt,
                    "evt_id": evt_id,
                    "evt_phase": evt_phase,
                    "roi": roi,
                    "mean": evtdat[roi].mean(),
                    "AUC": evtdat[roi].sum(),
                }
            )
            pk_col = roi + "-pks"
            if pk_col in evtdat.columns:
                dur = np.ptp(evtdat[ts_col])
                agg_dat = pd.concat(
                    [agg_dat, pd.Series({"pk_freq": evtdat[pk_col].sum() / dur})]
                )
            agg_df.append(agg_dat)
    return pd.concat(agg_df, axis="columns", ignore_index=True).T
