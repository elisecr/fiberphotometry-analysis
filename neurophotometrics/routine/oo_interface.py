import io
import itertools as itt
import os
import re
from warnings import warn

import numpy as np
import pandas as pd
import panel as pn
from ipyfilechooser import FileChooser
from IPython.display import display
from ipywidgets import Layout, widgets
from plotly.colors import qualitative

from .plotting import (
    construct_cmap,
    plot_agg_polled,
    plot_events,
    plot_peaks,
    plot_polled_signal,
    plot_signals,
)
from .polling import agg_polled_events, poll_events
from .processing import find_pks, photobleach_correction
from .ts_alignment import align_ts, label_bout
from .utilities import compute_fps, load_data


class NPMBase:
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        self.wgt_opts = {
            "style": {"description_width": "initial"},
            "layout": Layout(width="80%"),
        }
        self.data = None
        self.fig_path = fig_path
        self.out_path = out_path
        os.makedirs(self.fig_path, exist_ok=True)
        os.makedirs(self.out_path, exist_ok=True)

    def set_data(self, dpath: str = None, source: str = "local") -> None:
        if dpath is None:
            if source == "local":
                lab = widgets.Label("Select Data: ", layout=Layout(width="75px"))
                fc = FileChooser(".", **self.wgt_opts)
                fc.register_callback(self.on_set_data_local)
                display(widgets.HBox([lab, fc]))
            elif source == "remote":
                w_data = widgets.FileUpload(
                    accept=".csv",
                    multiple=False,
                    description="Upload Data File",
                    tooltip="Select data file to analyze",
                    **self.wgt_opts,
                )
                w_data.observe(self.on_set_data_remote, names="value")
                display(w_data)
        else:
            self.data = pd.read_csv(dpath)

    def on_set_data_remote(self, change) -> None:
        dat = change["new"][0]["content"].tobytes()
        self.data = pd.read_csv(io.BytesIO(dat), encoding="utf8")

    def on_set_data_local(self, fc) -> None:
        self.data = pd.read_csv(fc.selected)

    def set_paths(self, fig_path=None, out_path=None) -> None:
        if fig_path is None:
            lab = widgets.Label("Figure Path: ", layout=Layout(width="75px"))
            fc = FileChooser(self.fig_path, show_only_dirs=True, **self.wgt_opts)
            fc.register_callback(self.on_figpath)
            display(widgets.HBox([lab, fc]))
        else:
            self.fig_path = fig_path
            os.makedirs(fig_path, exist_ok=True)
        if out_path is None:
            lab = widgets.Label("Output Path: ", layout=Layout(width="75px"))
            fc = FileChooser(self.out_path, show_only_dirs=True, **self.wgt_opts)
            fc.register_callback(self.on_outpath)
            display(widgets.HBox([lab, fc]))
        else:
            self.out_path = out_path
            os.makedirs(out_path, exist_ok=True)

    def on_figpath(self, fc) -> None:
        self.fig_path = fc.selected_path
        os.makedirs(self.fig_path, exist_ok=True)

    def on_outpath(self, fc) -> None:
        self.out_path = fc.selected_path
        os.makedirs(self.out_path, exist_ok=True)


class NPMProcess(NPMBase):
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        super().__init__(fig_path, out_path)
        self.param_discard_time = None
        self.param_pk_prominence = None
        self.param_led_dict = {7: "initial", 1: "415nm", 2: "470nm", 4: "560nm"}
        self.param_roi_dict = None
        self.param_base_sig = None
        self.data_norm = None
        print("Process initialized")

    def set_discard_time(self, discard_time: float = None) -> None:
        assert self.data is not None, "Please set data first!"
        if discard_time is None:
            w_txt = widgets.Label(
                "Number of Seconds to Discard from Beginning of Recording"
            )
            w_nfm = widgets.FloatSlider(
                min=0,
                value=0,
                max=10,
                step=0.01,
                tooltip="Cropping data points at the beginning of the recording can improve curve fitting.",
                **self.wgt_opts,
            )
            self.param_discard_time = 0
            w_nfm.observe(self.on_discard, names="value")
            display(widgets.VBox([w_txt, w_nfm]))
        else:
            self.param_discard_time = discard_time

    def on_discard(self, change) -> None:
        self.param_discard_time = float(change["new"])

    def set_pk_prominence(self, prom: int = None) -> None:
        if prom is None:
            w_txt = widgets.Label("Peak Prominence")
            w_pk = widgets.FloatSlider(
                min=0,
                value=0.1,
                max=3,
                step=1e-3,
                readout_format=".3f",
                **self.wgt_opts,
            )
            self.param_pk_prominence = 0.1
            w_pk.observe(self.on_pk_prominence, names="value")
            display(widgets.VBox([w_txt, w_pk]))
        else:
            self.param_pk_prominence = prom

    def on_pk_prominence(self, change) -> None:
        self.param_pk_prominence = change["new"]

    def set_roi(self, roi_dict: dict = None) -> None:
        assert self.data is not None, "Please set data first!"
        if roi_dict is None:
            w_txt = widgets.Label("ROIs to analyze (CTRL/CMD click to Select Multiple)")
            w_roi = widgets.SelectMultiple(
                options=self.data.columns,
                tooltip="Region1G Region2R etc",
                **self.wgt_opts,
            )
            w_roi.observe(self.on_roi, names="value")
            display(widgets.VBox([w_txt, w_roi]))
        else:
            self.param_roi_dict = roi_dict

    def on_roi(self, change) -> None:
        rois = change["new"]
        self.param_roi_dict = {r: r for r in rois}

    def set_roi_names(self, roi_dict: dict = None) -> None:
        assert self.param_roi_dict is not None, "Please set roi first!"
        if roi_dict is None:
            rois = list(self.param_roi_dict.keys())
            w_rois = [
                widgets.Text(
                    value=r,
                    placeholder=r,
                    description="Region or Animal Corresponding to {}".format(r),
                    **self.wgt_opts,
                )
                for r in rois
            ]
            for w in w_rois:
                w.observe(self.on_roi_name, names="value")
                display(w)
        else:
            self.param_roi_dict = {k: v.replace("-", "_") for k, v in roi_dict.items()}

    def on_roi_name(self, change) -> None:
        k, v = change["owner"].placeholder, change["new"]
        self.param_roi_dict[k] = v.replace("-", "_")

    def set_baseline(self, base_sig: dict = None):
        assert self.data is not None, "Please set data first!"
        if base_sig is None:
            rois = list(self.param_roi_dict.values())
            sigs = list(set(self.param_led_dict.values()) - set(["initial"]))
            roi_sig = list(itt.product(rois, sigs))
            for key_r, key_s in roi_sig:
                opts = [("-".join(rs), {(key_r, key_s): rs}) for rs in roi_sig]
                opts = opts + [("No correction", {(key_r, key_s): None})]
                w_base = widgets.Dropdown(
                    description="{}-{}: ".format(key_r, key_s),
                    options=opts,
                    value={(key_r, key_s): None},
                    **self.wgt_opts,
                )
                w_base.observe(self.on_baseline, names="value")
                self.param_base_sig = dict()
                display(w_base)
        else:
            self.param_base_sig = base_sig

    def on_baseline(self, change) -> None:
        self.param_base_sig.update(change["new"])
        self.param_base_sig = {
            k: v for k, v in self.param_base_sig.items() if v is not None
        }

    def load_data(self, show_fig=True) -> None:
        assert self.data is not None, "Please set data first!"
        assert self.param_roi_dict is not None, "Please set ROIs first!"
        assert self.param_discard_time is not None, "Please set time to discard first!"
        self.data = load_data(
            self.data, self.param_discard_time, self.param_led_dict, self.param_roi_dict
        )
        fig = plot_signals(
            self.data, list(self.param_roi_dict.values()), default_window=(0, 10)
        )
        fig.write_html(os.path.join(self.fig_path, "raw_signals.html"))
        if show_fig:
            nroi = len(self.param_roi_dict)
            fig.update_layout(height=350 * nroi)
            display(fig)

    def photobleach_correction(self, show_fig=True) -> None:
        assert self.data is not None, "Please set data first!"
        assert self.param_roi_dict is not None, "Please set ROIs first!"
        assert self.param_base_sig is not None, "Please set baseline signal first!"
        self.data_norm = photobleach_correction(
            self.data, self.param_base_sig, rois=list(self.param_roi_dict.values())
        )
        fig = plot_signals(
            self.data_norm,
            list(self.param_roi_dict.values()),
            group_dict=lambda s: s.split("-")[0],
        )
        fig.write_html(os.path.join(self.fig_path, "photobleaching_correction.html"))
        if show_fig:
            nroi = len(self.param_roi_dict)
            fig.update_layout(height=350 * nroi)
            display(fig)

    def find_peaks(self, show_fig=True) -> None:
        self.data_norm = find_pks(
            self.data_norm,
            rois=list(self.param_roi_dict.values()),
            prominence=self.param_pk_prominence,
            sigs=["470nm-norm-zs"],
        )
        fig = plot_peaks(
            self.data_norm[self.data_norm["signal"] == "470nm-norm-zs"].copy(),
            rois=list(self.param_roi_dict.values()),
        )
        nroi = len(self.param_roi_dict)
        fig.update_layout(height=350 * nroi)
        fig.write_html(os.path.join(self.fig_path, "peaks.html"))
        if show_fig:
            display(fig)

    def export_data(
        self, sigs=["415nm", "470nm-norm", "470nm-norm-zs"], ds_path=None
    ) -> None:
        assert self.data_norm is not None, "Please process data first!"
        d = self.data_norm
        if ds_path is None:
            ds_path = os.path.join(self.out_path, "signals")
        os.makedirs(ds_path, exist_ok=True)
        for sig in sigs:
            fpath = os.path.join(ds_path, "{}.csv".format(sig))
            d[d["signal"] == sig].drop(columns=["signal"]).to_csv(fpath, index=False)
            print("data saved to {}".format(fpath))


class NPMAlign(NPMBase):
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        super().__init__(fig_path, out_path)
        self.ts_dict = dict()
        self.data_align = None
        print("Alignment initialized")

    def set_ts(self, ts_ls: list = None, source: str = "local") -> None:
        if ts_ls is None:
            if source == "local":
                fs = pn.widgets.FileSelector(
                    directory=".",
                    root_directory="/",
                    only_files=True,
                    name="Select Timestamp Files",
                )
                fs.param.watch(self.on_ts_local, ["value"], onlychanged=True)
                display(fs)
            elif source == "remote":
                w_ts = widgets.FileUpload(
                    accept=".csv",
                    multiple=True,
                    description="Upload Timestamp Files",
                    tooltip="Select timestamps to align",
                    **self.wgt_opts,
                )
                w_ts.observe(self.on_ts_remote, names="value")
                display(w_ts)
        else:
            for ts_path in ts_ls:
                ts_name, ts = self.load_ts(ts_path)
                self.ts_dict[ts_name] = ts

    def on_ts_remote(self, change) -> None:
        for dfile in change["new"]:
            dname = dfile["name"]
            dat = dfile["content"].tobytes()
            self.ts_dict[dname] = pd.read_csv(
                io.BytesIO(dat), encoding="utf8", header=None
            )

    def on_ts_local(self, event) -> None:
        for dpath in event.new:
            ts_name, ts = self.load_ts(dpath)
            self.ts_dict[ts_name] = ts

    def load_ts(self, ts_path: str) -> pd.DataFrame:
        ts_name = os.path.split(ts_path)[1]
        if ts_name.endswith(".csv"):
            return ts_name, pd.read_csv(ts_path, header=None)
        elif ts_path.endswith(".xlsx"):
            return ts_name, pd.read_excel(ts_path, header=None)
        else:
            raise NotImplementedError("Unable to read {}".format(ts_path))

    def align_data(self) -> None:
        # self.data = label_bout(self.data, "Stimulation") # depracated
        self.data_align, self.ts = align_ts(self.data, self.ts_dict)

    def export_data(self, ds_path=None, out_name=None) -> None:
        assert self.data_align is not None, "Please align ts first!"
        if ds_path is None:
            ds_path = os.path.join(self.out_path, "aligned")
        if out_name is None:
            out_name = "master.csv"
        os.makedirs(ds_path, exist_ok=True)
        fpath = os.path.join(ds_path, out_name)
        self.data_align.to_csv(fpath, index=False)
        print("data saved to {}".format(fpath))


class NPMPolling(NPMBase):
    def __init__(self, fig_path="./figs/process", out_path="./output/process") -> None:
        super().__init__(fig_path, out_path)
        self.param_evt_range = None
        self.param_evt_sep = 1
        self.param_evt_duration = 1
        print("Pooling initialized")

    def set_evt_range(self, evt_range: tuple = None, fps: float = None) -> None:
        if fps is None:
            assert self.data is not None, "Please set data first!"
            self.fps = compute_fps(self.data)
        else:
            self.fps = fps
        print("Assuming Framerate of {:.2f}".format(self.fps))
        if evt_range is None:
            txt_evt_range = widgets.Label(
                "Time (seconds) to Include Before and After Event"
            )
            w_evt_range = widgets.FloatRangeSlider(
                value=(-10, 10),
                min=-100,
                max=100,
                step=0.01,
                tooltip="Use the markers to specify the time (seconds) before and after each event",
                **self.wgt_opts,
            )
            self.param_evt_range = tuple(
                np.around(np.array((-10, 10)) * self.fps).astype(int)
            )
            w_evt_range.observe(self.on_evt_range, names="value")
            display(widgets.VBox([txt_evt_range, w_evt_range]))
        else:
            self.param_evt_range = evt_range

    def on_evt_range(self, change) -> None:
        self.param_evt_range = tuple(
            np.around(np.array(change["new"]) * self.fps).astype(int)
        )

    def set_evt_sep(self, evt_sep: float = None) -> None:
        if evt_sep is None:
            w_txt = widgets.Label("Minimum seperation between events (seconds)")
            w_evt_sep = widgets.FloatSlider(
                min=0, value=0, max=10, step=0.01, **self.wgt_opts
            )
            self.param_evt_sep = 0
            w_evt_sep.observe(self.on_evt_sep, names="value")
            display(widgets.VBox([w_txt, w_evt_sep]))
        else:
            self.param_evt_sep = evt_sep

    def on_evt_sep(self, change) -> None:
        self.param_evt_sep = float(change["new"])

    def set_evt_duration(self, evt_duration: float = None) -> None:
        if evt_duration is None:
            w_txt = widgets.Label("Minimum duration of events (seconds)")
            w_evt_dur = widgets.FloatSlider(
                min=0, value=0, max=10, step=0.01, **self.wgt_opts
            )
            self.param_evt_duration = 0
            w_evt_dur.observe(self.on_evt_dur, names="value")
            display(widgets.VBox([w_txt, w_evt_dur]))
        else:
            self.param_evt_duration = evt_duration

    def on_evt_dur(self, change) -> None:
        self.param_evt_duration = float(change["new"])

    def set_roi(self, roi_dict: dict = None) -> None:
        assert self.data is not None, "Please set data first!"
        if roi_dict is None:
            w_txt = widgets.Label("ROIs to analyze (CTRL/CMD click to Select Multiple)")
            w_roi = widgets.SelectMultiple(
                options=self.data.columns,
                tooltip="Region1G Region2R etc",
                **self.wgt_opts,
            )
            w_roi.observe(self.on_roi, names="value")
            display(widgets.VBox([w_txt, w_roi]))
        else:
            self.param_roi_dict = roi_dict

    def on_roi(self, change) -> None:
        rois = change["new"]
        self.param_roi_dict = {r: r for r in rois}

    def poll_events(self, show_fig=True) -> None:
        self.evtdf = poll_events(
            self.data,
            self.param_evt_range,
            list(self.param_roi_dict.values()),
            self.param_evt_sep,
            self.param_evt_duration,
        )
        cmap = construct_cmap(self.evtdf["evt_id"].unique(), qualitative.Plotly)
        fig = plot_events(
            self.data,
            self.evtdf,
            list(self.param_roi_dict.values()),
            ts_col="ts_fp",
            cmap=cmap,
        )
        fig.write_html(os.path.join(self.fig_path, "events.html"))
        if show_fig:
            display(fig)
        fig = plot_polled_signal(
            self.evtdf, list(self.param_roi_dict.values()), fps=self.fps, cmap=cmap
        )
        fig.write_html(os.path.join(self.fig_path, "polled_signals.html"))
        if show_fig:
            display(fig)

    def agg_polled_events(self, show_fig=True) -> None:
        self.evt_agg = agg_polled_events(self.evtdf, list(self.param_roi_dict.values()))
        fig, figs = plot_agg_polled(self.evt_agg)
        for met, cur_fig in figs.items():
            cur_fig.write_html(
                os.path.join(self.fig_path, "polled_signals-{}.html".format(met))
            )
        if show_fig:
            display(fig)

    def export_data(self, ds_path=None, evt_out_name=None, agg_out_name=None) -> None:
        assert self.evtdf is not None, "Please poll events first!"
        assert self.evt_agg is not None, "Please aggregate polled events first!"
        if ds_path is None:
            ds_path = os.path.join(self.out_path, "polled")
        if evt_out_name is None:
            evt_out_name = "events.csv"
        if agg_out_name is None:
            agg_out_name = "aggregated.csv"
        os.makedirs(ds_path, exist_ok=True)
        dpath = os.path.join(ds_path, evt_out_name)
        self.evtdf.to_csv(dpath, index=False)
        print("data saved to {}".format(dpath))
        dpath = os.path.join(ds_path, agg_out_name)
        self.evt_agg.to_csv(dpath, index=False)
        print("data saved to {}".format(dpath))


class NPMBatch(NPMBase):
    def __init__(self, fig_path="./figs/batch", out_path="./output/batch") -> None:
        super().__init__(fig_path, out_path)
        self.data_path = None
        self.flist = None
        self.dat_pattern = None
        self.ts_pattern = None
        self.dat_files = None
        self.ts_files = None
        self.process = NPMProcess()
        self.polling = NPMPolling()

    def set_data_path(self, data_path: str = None) -> None:
        if data_path is None:
            lab = widgets.Label("Select Data: ", layout=Layout(width="75px"))
            fc = FileChooser(".", **self.wgt_opts)
            fc.register_callback(self.on_set_data_path)
            display(widgets.HBox([lab, fc]))
        else:
            self.data_path = data_path
            self.set_flist()

    def on_set_data_path(self, fc) -> None:
        self.data_path = fc.selected
        self.set_flist()

    def set_flist(self) -> None:
        flist = [
            os.path.join(self.data_path, f)
            for f in os.listdir(self.data_path)
            if f.endswith(".csv")
        ]
        print("Found {} csv files".format(len(flist)))
        self.flist = flist

    def set_data_pattern(self, dat_pattern: str = None) -> None:
        assert self.flist is not None, "Please specify data path first!"
        if dat_pattern is None:
            self.dat_pattern = r"^(?P<key>.*)-fiber-photometry0\.csv$"
            w_dat = widgets.Text(
                value=r"^(?P<key>.*)-fiber-photometry0\.csv$",
                description="Data File Pattern",
                **self.wgt_opts,
            )
            w_but = widgets.Button(description="Confirm")
            w_dat.observe(self.on_data_pattern)
            w_but.on_click(self.on_data_pattern_confirm)
            display(widgets.HBox([w_dat, w_but]))
        else:
            self.dat_pattern = dat_pattern
            self.set_data_files()

    def on_data_pattern(self, change) -> None:
        self.dat_pattern = change["new"]

    def on_data_pattern_confirm(self, b) -> None:
        self.set_data_files()

    def set_data_files(self) -> None:
        dat_files = dict()
        for fpath in self.flist:
            fname = os.path.basename(fpath)
            m = re.search(self.dat_pattern, fname)
            if m is not None:
                fkey = m.groupdict()["key"]
                dat_files[fkey] = fpath
        print("Found {} data files".format(len(dat_files)))
        for fkey, fpath in dat_files.items():
            print("{}: {}".format(fkey, fpath))
        self.dat_files = dat_files
        self.process.set_data(list(dat_files.values())[0])

    def set_ts_pattern(self, ts_pattern: str = None) -> None:
        assert self.flist is not None, "Please specify data path first!"
        assert self.dat_files is not None, "Please specify data pattern first!"
        if ts_pattern is None:
            self.ts_pattern = r"^(?P<key>.*)-KeydownTimeStamp0\.csv$"
            w_dat = widgets.Text(
                value=r"^(?P<key>.*)-KeydownTimeStamp0\.csv$",
                description="Timestamp File Pattern",
                **self.wgt_opts,
            )
            w_but = widgets.Button(description="Confirm")
            w_dat.observe(self.on_ts_pattern)
            w_but.on_click(self.on_ts_pattern_confirm)
            display(widgets.HBox([w_dat, w_but]))
        else:
            self.ts_pattern = ts_pattern
            self.set_ts_files()

    def on_ts_pattern(self, change) -> None:
        self.ts_pattern = change["new"]

    def on_ts_pattern_confirm(self, b) -> None:
        self.set_ts_files()

    def set_ts_files(self) -> None:
        ts_files = dict()
        for fpath in self.flist:
            fname = os.path.basename(fpath)
            m = re.search(self.ts_pattern, fname)
            if m is not None:
                fkey = m.groupdict()["key"]
                assert (
                    fkey in self.dat_files
                ), "Cannot find matching data for timestamp with key: {}".format(fkey)
                ts_files[fkey] = fpath
        print("Found {} timestamp files".format(len(ts_files)))
        for fkey, fpath in ts_files.items():
            print("{}: {}".format(fkey, fpath))
        self.ts_files = ts_files

    def set_discard_time(self, discard_time: float = None) -> None:
        self.process.set_discard_time(discard_time)

    def set_pk_prominence(self, prom: int = None) -> None:
        self.process.set_pk_prominence(prom)

    def set_roi(self, roi_dict: dict = None) -> None:
        self.process.set_roi(roi_dict)

    def set_roi_names(self, roi_dict: dict = None) -> None:
        self.process.set_roi_names(roi_dict)

    def set_baseline(self, base_sig: dict = None) -> None:
        self.process.set_baseline(base_sig)

    def set_evt_range(self, evt_range: tuple = None) -> None:
        exp_data = pd.read_csv(list(self.dat_files.values())[0])
        self.polling.set_evt_range(
            evt_range,
            fps=compute_fps(exp_data, fm_col="FrameCounter", tcol="SystemTimestamp"),
        )

    def set_evt_sep(self, evt_sep: float = None) -> None:
        self.polling.set_evt_sep(evt_sep)

    def set_evt_duration(self, evt_duration: float = None) -> None:
        self.polling.set_evt_duration(evt_duration)

    def batch_process(self) -> None:
        for fkey, dat_path in self.dat_files.items():
            print("processing {}".format(fkey))
            fig_path = os.path.join(self.fig_path, fkey)
            out_path = os.path.join(self.out_path, fkey)
            self.process.set_paths(fig_path, out_path)
            self.process.set_data(dat_path)
            self.process.load_data(show_fig=False)
            self.process.photobleach_correction(show_fig=False)
            self.process.find_peaks(show_fig=False)
            self.process.export_data(ds_path=out_path)
            try:
                ts_path = self.ts_files[fkey]
            except KeyError:
                warn("skipping alignment due to missing timestamps for {}".format(fkey))
                continue
            algn = NPMAlign()
            algn.set_paths(fig_path, out_path)
            algn.set_data(os.path.join(out_path, "470nm-norm.csv"))
            algn.set_ts([ts_path])
            algn.align_data()
            algn.export_data(ds_path=out_path, out_name="{}-aligned.csv".format(fkey))
            self.polling.set_paths(fig_path, out_path)
            self.polling.set_data(os.path.join(out_path, "{}-aligned.csv".format(fkey)))
            self.polling.set_roi(self.process.param_roi_dict)
            self.polling.poll_events(show_fig=False)
            self.polling.agg_polled_events(show_fig=False)
            self.polling.export_data(
                ds_path=out_path,
                evt_out_name="{}-events.csv".format(fkey),
                agg_out_name="{}-aggregated.csv".format(fkey),
            )
