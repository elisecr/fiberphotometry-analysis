import itertools as itt

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import convert_colors_to_same_type, qualitative, unlabel_rgb
from plotly.subplots import make_subplots

from .utilities import enumerated_product

pn.extension("plotly")


def construct_cmap(keys, cmap=qualitative.Plotly):
    return {k: c for k, c in zip(keys, itt.cycle(cmap))}


def add_alpha(col, alpha):
    col = unlabel_rgb(convert_colors_to_same_type(col)[0][0])
    return "rgba({},{},{},{})".format(*col, alpha)


def plot_signals(data, rois, fps=30, default_window=None, group_dict=None):
    dat_long = data[["SystemTimestamp", "signal"] + rois].melt(
        id_vars=["SystemTimestamp", "signal"], var_name="roi", value_name="raw"
    )
    t0 = dat_long["SystemTimestamp"].min()
    dat_long["Time (s)"] = dat_long["SystemTimestamp"] - t0
    if group_dict is not None:
        dat_long["signal_group"] = dat_long["signal"].map(group_dict)
    else:
        dat_long["signal_group"] = dat_long["signal"]
    fig = px.line(
        dat_long,
        x="Time (s)",
        y="raw",
        facet_row="roi",
        facet_col="signal_group",
        color="signal",
        range_x=default_window,
        facet_col_spacing=0.04,
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    return fig


def plot_polled_signal(evt_df, rois, fps=30, cmap=None):
    id_vars = ["fm_evt", "evt_id", "event"]
    evt_df = (
        evt_df[id_vars + rois]
        .drop_duplicates()
        .melt(id_vars=id_vars, var_name="roi", value_name="fluorescence")
    )
    evt_df["Time (s)"] = evt_df["fm_evt"] / fps
    fig = px.line(
        evt_df,
        x="Time (s)",
        y="fluorescence",
        color="evt_id",
        facet_row="event",
        facet_col="roi",
        color_discrete_map=cmap,
    )
    fig.update_layout(title="Polled Signals", height=300 * evt_df["event"].nunique())
    return fig


def facet_plotly(
    data: pd.DataFrame,
    facet_row: str,
    facet_col: str,
    title_dim: str = None,
    specs: dict = None,
    col_wrap: int = None,
    **kwargs,
):
    row_crd = data[facet_row].unique()
    col_crd = data[facet_col].unique()
    layout_ls = []
    iiter = 0
    for (ir, ic), (r, c) in enumerated_product(row_crd, col_crd):
        dat_sub = data[(data[facet_row] == r) & (data[facet_col] == c)]
        if not len(dat_sub) > 0:
            continue
        if title_dim is not None:
            title = dat_sub[title_dim].unique().item()
        else:
            if facet_row == "DUMMY_FACET_ROW":
                title = "{}={}".format(facet_col, c)
            elif facet_col == "DUMMY_FACET_COL":
                title = "{}={}".format(facet_row, r)
            else:
                title = "{}={}; {}={}".format(facet_row, r, facet_col, c)
        if col_wrap is not None:
            ir = iiter // col_wrap
            ic = iiter % col_wrap
            iiter += 1
        layout_ls.append(
            {"row": ir, "col": ic, "row_label": r, "col_label": c, "title": title}
        )
    layout = pd.DataFrame(layout_ls).set_index(["row_label", "col_label"])
    if col_wrap is not None:
        nrow, ncol = int(layout["row"].max() + 1), int(layout["col"].max() + 1)
    else:
        nrow, ncol = len(row_crd), len(col_crd)
    if specs is not None:
        specs = np.full((nrow, ncol), specs).tolist()
    fig = make_subplots(
        rows=nrow,
        cols=ncol,
        subplot_titles=layout["title"].values,
        specs=specs,
        **kwargs,
    )
    return fig, layout


def construct_layout(row_crd=None, col_crd=None, row_name="", col_name="", **kwargs):
    if row_crd is None:
        row_crd = [None]
    if col_crd is None:
        col_crd = [None]
    layout_ls = []
    for (ir, ic), (r, c) in enumerated_product(row_crd, col_crd):
        tt = ""
        if r:
            if row_name:
                tt = tt + row_name + ": " + r
            else:
                tt = tt + r
            tt = tt + ", "
        if c:
            if col_name:
                tt = tt + col_name + ": " + c
            else:
                tt = tt + c
        layout_ls.append(
            {"row": ir, "col": ic, "row_label": r, "col_label": c, "title": tt}
        )
    layout = pd.DataFrame(layout_ls)
    nrow, ncol = len(row_crd), len(col_crd)
    fig = make_subplots(
        rows=nrow, cols=ncol, subplot_titles=layout["title"].values, **kwargs
    )
    return fig, layout


def plot_peaks(data, rois, ts_col="SystemTimestamp", default_window=None):
    sigs = data["signal"].unique()
    t0 = data[ts_col].min()
    data["t"] = data[ts_col] - t0
    fig, layout = construct_layout(
        rois, sigs, "roi", "signal", shared_xaxes=True, x_title="Time (s)"
    )
    for (roi, sig), ly in layout.groupby(["row_label", "col_label"]):
        dat = data[data["signal"] == sig]
        pks = dat[dat[roi + "-pks"]]
        ly = ly.squeeze()
        if len(dat) > 0:
            fig.add_trace(
                go.Scatter(
                    x=dat["t"],
                    y=dat[roi],
                    mode="lines",
                    name="signal",
                    legendgroup="signal",
                    line={"color": "#636EFA"},
                ),
                row=ly["row"] + 1,
                col=ly["col"] + 1,
            )
            if roi + "-freq" in dat.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dat["t"],
                        y=dat[roi + "-freq"],
                        mode="lines",
                        name="freq",
                        legendgroup="freq",
                        line={"color": "grey"},
                    ),
                    row=ly["row"] + 1,
                    col=ly["col"] + 1,
                )
            fig.add_trace(
                go.Scatter(
                    x=pks["t"],
                    y=pks[roi],
                    mode="markers",
                    marker={"size": 8, "color": "#EF553B", "symbol": "cross"},
                    name="peaks",
                ),
                row=ly["row"] + 1,
                col=ly["col"] + 1,
            )
    return fig


def plot_events(data, evtdf, rois, cmap=None, ts_col="SystemTimestamp"):
    data = data.copy()
    evtdf = evtdf.copy()
    t0 = data[ts_col].min()
    data["t"] = data[ts_col] - t0
    evtdf["t"] = evtdf[ts_col] - t0
    fig, layout = construct_layout(
        rois, row_name="roi", shared_xaxes=True, x_title="Time (s)"
    )
    for roi, ly in layout.groupby("row_label"):
        ly = ly.squeeze()
        if len(data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=data["t"],
                    y=data[roi],
                    mode="lines",
                    name="signal",
                    showlegend=False,
                    line={"color": "#404040", "width": 1.5},
                ),
                row=ly["row"] + 1,
                col=ly["col"] + 1,
            )
    for evt_id, dat in evtdf.groupby("evt_id"):
        if cmap is not None:
            fc = cmap[evt_id]
        else:
            fc = "gray"
        fig.add_vrect(
            x0=dat["t"].min(),
            x1=dat["t"].max(),
            annotation_text=evt_id,
            fillcolor=fc,
            opacity=0.7,
            line_width=0,
        )
    fig.update_layout(title="Events")
    return fig


def plot_agg_polled(data):
    data = data.melt(
        id_vars=["event_type", "event", "evt_id", "evt_phase", "roi"], var_name="metric"
    )
    tb_ls = []
    figs_dict = dict()
    cmap = construct_cmap(data["evt_id"].unique())
    cmap_alpha = {k: add_alpha(c, 0.3) for k, c in cmap.items()}
    for met, met_df in data.groupby("metric"):
        fig, layout = construct_layout(
            met_df["event"].unique(),
            met_df["roi"].unique(),
            row_name="event",
            col_name="roi",
        )
        show_leg = {k: True for k in cmap.keys()}
        met_df["evt_phase"] = pd.Categorical(
            met_df["evt_phase"], ["before", "during", "after"]
        )
        met_df = met_df.sort_values(
            ["event_type", "event", "roi", "evt_id", "evt_phase"]
        ).set_index(["event", "roi"])
        for (evt, roi), ly in layout.groupby(["row_label", "col_label"]):
            dat_sub = met_df.loc[evt, roi]
            ly = ly.squeeze()
            dat_bar = (
                dat_sub.groupby("evt_phase", observed=True)["value"]
                .agg(["mean", "sem"])
                .reset_index()
            )
            fig.add_trace(
                go.Bar(
                    x=dat_bar["evt_phase"],
                    y=dat_bar["mean"],
                    error_y={
                        "type": "data",
                        "array": dat_bar["sem"],
                        "color": "#404040",
                        "thickness": 3,
                        "width": 10,
                    },
                    marker={
                        "color": "rgba(0,0,0,0.2)",
                        "line": {"color": "#404040", "width": 2},
                    },
                    showlegend=False,
                ),
                row=ly["row"] + 1,
                col=ly["col"] + 1,
            )
            for evtid, d in dat_sub.groupby("evt_id"):
                fig.add_trace(
                    go.Scatter(
                        x=d["evt_phase"],
                        y=d["value"],
                        name=evtid,
                        mode="lines+markers",
                        legendgroup=evtid,
                        showlegend=show_leg[evtid],
                        marker={
                            "color": cmap_alpha[evtid],
                            "size": 10,
                            "line": {"color": cmap[evtid], "width": 1.8},
                        },
                        line={"color": cmap[evtid], "dash": "dot", "width": 1.2},
                    ),
                    row=ly["row"] + 1,
                    col=ly["col"] + 1,
                )
                show_leg[evtid] = False
        fig.update_layout(height=300 * layout["row_label"].nunique())
        figs_dict[met] = fig
        tb_ls.append((met, pn.pane.Plotly(fig, sizing_mode="stretch_width")))
    return pn.Tabs(*tb_ls, dynamic=True), figs_dict
