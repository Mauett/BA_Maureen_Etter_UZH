# import os
# import warnings
# from typing import Optional

# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd
# from pyparsing import line

# plt.rcParams.update({
#     "font.size": 14,           # base font
#     "axes.titlesize": 16,      # plot title
#     "axes.labelsize": 15,      # axis labels
#     "xtick.labelsize": 13,
#     "ytick.labelsize": 13,
#     "legend.fontsize": 12,
#     "figure.titlesize": 16,
# })


# class PlottingAfterpulses:
#     """
#     Class for plots related to the afterpulse processing.
#     """

#     def __init__(
#         self,
#         df: pd.DataFrame,
#         adc_f: float,
#         ap_rate_dict: dict = None,
#         save_plots: bool = False,
#         show_plots: bool = True,
#         save_dir: Optional[str] = None,
#         save_name_suffix: Optional[str] = None,
#         adc_area_to_e: Optional[float] = None,
#         gain: Optional[float] = None,
#         pmt_serial: Optional[str] = None,
#         hv_v: Optional[str] = None,
#         lamp_v: Optional[str] = None,
#         frame_type: Optional[str] = None,
#         n_samples_per_waveform: Optional[int] = None,
#         n_waveforms_analyzed: Optional[int] = None,
#     ):
#         self.df = df
#         self.ap_rate_dict = ap_rate_dict
#         self.adc_f = adc_f
#         self.adc_area_to_e = adc_area_to_e
#         self.gain = gain
#         self.save_plots = save_plots
#         self.show_plots = show_plots
#         self.save_dir = save_dir
#         self.save_name_suffix = save_name_suffix

#         self.pmt_serial = pmt_serial
#         self.hv_v = hv_v
#         self.lamp_v = lamp_v
#         self.frame_type = frame_type
#         self.n_samples_per_waveform = n_samples_per_waveform
#         self.n_waveforms_analyzed = n_waveforms_analyzed

#         if save_plots and (save_dir is None):
#             raise NameError("save_dir must be defined if save_plots is True.")
#         if save_plots and (save_name_suffix is None):
#             raise NameError("save_name_suffix must be defined if save_plots is True.")

#     def _plot_info_text(self) -> str:
#         lines = []

#         if self.pmt_serial is not None:
#             lines.append(rf"$\mathrm{{PMT}} = {self.pmt_serial}$")
#         if self.gain is not None:
#             lines.append(rf"$G = {self.gain:.3e}$")
#         if self.hv_v is not None:
#             hv_num = self.hv_v.replace(" V", "")
#             lines.append(rf"$V_{{\mathrm{{HV}}}} = {hv_num}\,\mathrm{{V}}$")
#         if self.lamp_v is not None:
#             lamp_num = self.lamp_v.replace(" Vpp", "")
#             lines.append(rf"$V_{{\mathrm{{Lamp}}}} = {lamp_num}\,\mathrm{{V_{{pp}}}}$")
#         if self.n_samples_per_waveform is not None:
#             lines.append(rf"$N_{{\mathrm{{frame}}}} = {self.n_samples_per_waveform}$")
#         if self.n_waveforms_analyzed is not None:
#             lines.append(rf"$N_{{\mathrm{{wf}}}} = {self.n_waveforms_analyzed}$")

#         if self.ap_rate_dict is not None:
#             if self.ap_rate_dict.get("area_thr_ap") is not None:
#                 lines.append(rf"$A_{{\mathrm{{thr}}}} = {self.ap_rate_dict['area_thr_ap']}\,\mathrm{{PE}}$")
#             if self.ap_rate_dict.get("t_thr_ap") is None:
#                 lines.append(r"$t_{\mathrm{thr}} = \mathrm{None}$")
#             else:
#                 lines.append(rf"$t_{{\mathrm{{thr}}}} = {self.ap_rate_dict['t_thr_ap']}\,\mathrm{{ns}}$")

#         return "\n".join(lines)

#     def _add_info_box(self):
#         info_text = self._plot_info_text()

#         fig = plt.gcf()

#         # Mehr Platz rechts für Box + Legende
#         fig.subplots_adjust(right=0.72)

#         fig.text(
#             0.75, 0.88,
#             info_text,
#             ha="left",
#             va="top",
#             fontsize=10,
#             bbox=dict(
#                 boxstyle="round",
#                 facecolor="white",
#                 edgecolor="black",
#                 alpha=0.92
#             )
#         )

#     def plot_wf(self, i: int = 0):
#         """Plot i-th afterpulse candidate waveform."""
#         plt.figure(figsize=(12, 6))

#         if (i >= self.df.shape[0]) or (i < 0):
#             raise IndexError(
#                 "Integer-location based index i must be between 0 and {}".format(
#                     self.df.shape[0] - 1
#                 )
#             )

#         separability = "Separable" if self.df.iloc[i]["separable"] else "Non-Separable"
#         x_dummy = np.arange(len(self.df.iloc[i]["input_data_converted"])) / self.adc_f * 1e9

#         plt.step(x_dummy, self.df.iloc[i]["input_data_converted"], where="mid", linewidth=1.2, label="Waveform")
#         plt.axvline(
#             x=self.df.iloc[i]["p0_position"] / self.adc_f * 1e9,
#             c="gray",
#             linestyle="dashed",
#             zorder=-1,
#             label="Peak positions\n" + r"$\Delta t = {}\,\mathrm{{ns}}$".format(self.df.iloc[i]["t_diff_ns"]),
#         )
#         plt.axvline(
#             x=self.df.iloc[i]["p1_position"] / self.adc_f * 1e9,
#             c="gray",
#             linestyle="dashed",
#             zorder=-1,
#         )
#         plt.axvspan(
#             self.df.iloc[i]["p0_lower_bound"] / self.adc_f * 1e9 - 0.5,
#             self.df.iloc[i]["p0_upper_bound"] / self.adc_f * 1e9,
#             color="C1",
#             lw=0,
#             alpha=0.4,
#             zorder=-2,
#             label="Main pulse",
#         )
#         plt.axvspan(
#             self.df.iloc[i]["p1_lower_bound"] / self.adc_f * 1e9,
#             self.df.iloc[i]["p1_upper_bound"] / self.adc_f * 1e9 + 0.5,
#             color="C3",
#             lw=0,
#             alpha=0.4,
#             zorder=-2,
#             label="Afterpulse\n({})".format(separability),
#         )

#         plt.xlim(0, self.n_samples_per_waveform)
#         plt.xlabel(r"Time $t\,[\mathrm{ns}]$")
#         plt.ylabel(r"Amplitude $[\mathrm{ADC}]$")

#         legend = plt.legend(loc="upper left", bbox_to_anchor=(1.01, 0.55))
#         legend.get_frame().set_linewidth(matplotlib.rcParams["axes.linewidth"])

#         self._add_info_box()

#         filename = "ap_candidate_wf_example_{}_{}".format(i, self.save_name_suffix)
#         if self.save_plots:
#             plt.savefig(os.path.join(self.save_dir, filename + ".png"), bbox_inches="tight")
#         if self.show_plots:
#             plt.show()
#         else:
#             plt.close()

#     def plot_first_n_wfs(self, n: int = 3):
#         """Plot first n afterpulse candidate waveforms."""
#         for i in range(min(n, self.df.shape[0])):
#             self.plot_wf(i)

#     def plot_hist_tdiff(self):
#         """Plot time differences afterpulse - main pulse."""
#         plt.figure(figsize=(12, 6))

#         x_dummy = np.arange(0, int(self.df["t_diff_ns"].max()), step=int(1e9 / self.adc_f))
#         n_all, bins_edges, _ = plt.hist(
#             self.df["t_diff_ns"],
#             bins=x_dummy - 0.5,
#             histtype="step",
#             color="C0",
#             label="All afterpulse candidates",
#         )
#         bins_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
#         plt.fill_between(
#             bins_centers,
#             n_all - np.sqrt(n_all),
#             n_all + np.sqrt(n_all),
#             color="C0",
#             alpha=0.5,
#             zorder=-1,
#         )

#         n_sep, _, _ = plt.hist(
#             self.df[self.df["separable"]]["t_diff_ns"],
#             bins=x_dummy - 0.5,
#             histtype="step",
#             color="C1",
#             label="Separable afterpulse candidates",
#         )
#         plt.fill_between(
#             bins_centers,
#             n_sep - np.sqrt(n_sep),
#             n_sep + np.sqrt(n_sep),
#             color="C1",
#             alpha=0.5,
#             zorder=-1,
#         )

#         plt.xlabel(r"Time difference $\Delta t\,[\mathrm{ns}]$")
#         plt.ylabel("Entries")
#         plt.yscale("log")
#         plt.xlim(right=x_dummy[-1])

#         legend = plt.legend(loc="upper left", bbox_to_anchor=(1.01, 0.55))
#         legend.get_frame().set_linewidth(matplotlib.rcParams["axes.linewidth"])

#         self._add_info_box()

#         filename = "ap_tdiff_{}".format(self.save_name_suffix)
#         if self.save_plots:
#             plt.savefig(os.path.join(self.save_dir, filename + ".png"), bbox_inches="tight")
#         if self.show_plots:
#             plt.show()
#         else:
#             plt.close()

#     def plot_ap_area_hist(self, xmax=10, binsize=0.1, separable_overlay=True, show_thr=True):
#         """Plot histogram of afterpulse area in PE."""
#         plt.figure(figsize=(12, 6))

#         well_defined = True
#         if self.adc_area_to_e is None:
#             warnings.warn(
#                 "Attribute adc_area_to_e set to None, will default to value 3047.6 for ADC V1730D."
#             )
#             adc_area_to_e = 3047.6
#             well_defined = False
#         else:
#             adc_area_to_e = self.adc_area_to_e

#         if self.gain is None:
#             warnings.warn("Attribute gain set to None, will default to value 3e6.")
#             gain = 3e6
#             well_defined = False
#         else:
#             gain = self.gain

#         self.df["p1_area_conv"] = self.df["p1_area"] * adc_area_to_e / gain

#         bins = np.arange(0, xmax + binsize, binsize)

#         df_plot_all = self.df[
#             (self.df["p1_area_conv"] >= 0) & (self.df["p1_area_conv"] <= xmax)
#         ]

#         plt.hist(
#             df_plot_all["p1_area_conv"],
#             bins=bins,
#             histtype="step",
#             label="All afterpulse candidates",
#         )

#         if separable_overlay:
#             df_plot_sep = self.df[self.df["separable"]]
#             df_plot_sep = df_plot_sep[
#                 (df_plot_sep["p1_area_conv"] >= 0) & (df_plot_sep["p1_area_conv"] <= xmax)
#             ]

#             plt.hist(
#                 df_plot_sep["p1_area_conv"],
#                 bins=bins,
#                 histtype="step",
#                 label="Separable afterpulse candidates",
#             )

#         if show_thr and (self.ap_rate_dict is not None):
#             if self.ap_rate_dict.get("area_thr_ap") is not None:
#                 plt.axvline(
#                     self.ap_rate_dict["area_thr_ap"],
#                     color="gray",
#                     linestyle="dashed",
#                     label=r"$A_{\mathrm{thr}}$",
#                 )

#         if well_defined:
#             plt.xlabel(r"Afterpulse area $A_{\mathrm{AP}}\,[\mathrm{PE}]$")
#         else:
#             plt.xlabel(r"Afterpulse area $[\mathrm{A.U.}]$")

#         plt.ylabel("Entries")
#         plt.xlim(0, xmax)
#         plt.yscale("log")

#         legend = plt.legend(loc="upper left", bbox_to_anchor=(1.01, 0.55))
#         legend.get_frame().set_linewidth(matplotlib.rcParams["axes.linewidth"])

#         self._add_info_box()

#         filename = f"ap_area_hist_{self.save_name_suffix}"

#         if self.save_plots:
#             plt.savefig(os.path.join(self.save_dir, filename + ".png"), bbox_inches="tight")
#         if self.show_plots:
#             plt.show()
#         else:
#             plt.close()

#         self.df.drop(columns=["p1_area_conv"], inplace=True)

#     def plot_ap_area_vs_tdiff(self, separable_only=True, ymax=25, xmax=2.0, show_thr=True):
#         """Plot afterpulse area vs time difference histogram."""
#         plt.figure(figsize=(12, 6))

#         well_defined = True
#         if self.adc_area_to_e is None:
#             warnings.warn(
#                 "Attribute adc_area_to_e set to None, will default to value 3047.6 for ADC V1730D."
#             )
#             adc_area_to_e = 3047.6
#             well_defined = False
#         else:
#             adc_area_to_e = self.adc_area_to_e

#         if self.gain is None:
#             warnings.warn("Attribute gain set to None, will default to value 3e6.")
#             gain = 3e6
#             well_defined = False
#         else:
#             gain = self.gain

#         self.df["p1_area_conv"] = self.df["p1_area"] * adc_area_to_e / gain
#         self.df["t_diff_us"] = self.df["t_diff_ns"] / 1e3

#         binsx = np.arange(0, xmax + 0.02, 0.02)
#         binsy = np.arange(0, ymax + 0.2, 0.2)

#         if separable_only:
#             df_plot = self.df[self.df["separable"]]
#         else:
#             df_plot = self.df

#         df_plot = df_plot[(df_plot["t_diff_us"] >= 0) & (df_plot["t_diff_us"] <= xmax)]
#         df_plot = df_plot[(df_plot["p1_area_conv"] >= 0) & (df_plot["p1_area_conv"] <= ymax)]

#         if len(df_plot) > 0:
#             plt.hist2d(
#                 df_plot["t_diff_us"],
#                 df_plot["p1_area_conv"],
#                 bins=(binsx, binsy),
#                 norm=matplotlib.colors.LogNorm(),
#             )
#         else:
#             warnings.warn("No entries available inside requested plotting range.")

#         if show_thr:
#             if self.ap_rate_dict is None:
#                 warnings.warn("Cannot display thresholds as parameter ap_rate_dict is None.")
#             else:
#                 if self.ap_rate_dict.get("area_thr_ap") is not None:
#                     plt.axhline(y=self.ap_rate_dict["area_thr_ap"], color="gray")
#                 if self.ap_rate_dict.get("t_thr_ap") is not None:
#                     plt.axvline(x=self.ap_rate_dict["t_thr_ap"] / 1e3, color="gray")

#         plt.xlabel(r"Time difference $\Delta t\,[\mu\mathrm{s}]$")
#         plt.xlim(0, xmax)
#         plt.ylim(0, ymax)

#         if well_defined:
#             plt.ylabel(r"Afterpulse area $A_{\mathrm{AP}}\,[\mathrm{PE}]$")
#         else:
#             plt.ylabel(r"Afterpulse area $[\mathrm{A.U.}]$")

#         plt.colorbar(label="Entries")

#         self._add_info_box()

#         if separable_only:
#             filename = f"ap_area_vs_tdiff_separable_{self.save_name_suffix}"
#         else:
#             filename = f"ap_area_vs_tdiff_{self.save_name_suffix}"

#         if self.save_plots:
#             plt.savefig(os.path.join(self.save_dir, filename + ".png"), bbox_inches="tight")
#         if self.show_plots:
#             plt.show()
#         else:
#             plt.close()

#         self.df.drop(columns=["p1_area_conv", "t_diff_us"], inplace=True)

#     def plot_essentials(self):
#         """Plot essential plots for afterpulse studies."""
#         self.plot_first_n_wfs(n=5)
#         self.plot_hist_tdiff()
#         self.plot_ap_area_hist(xmax=10, binsize=0.1)
#         self.plot_ap_area_vs_tdiff(separable_only=False, ymax=25, xmax=2.0)
#         self.plot_ap_area_vs_tdiff(separable_only=True, ymax=25, xmax=2.0)

#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Scientific plotting style for afterpulse plots.

Drop this class into AP_run_analysis_full_data.py, or import it from this file.
Important: if you keep this class in AP_run_analysis_full_data.py, remove:

    from pmt_analysis.plotting.afterpulses import PlottingAfterpulses

otherwise that import overwrites this local class.
"""

import os
import warnings
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})


def make_scientific_figure(figsize=(10.0, 6.0)):
    """
    Create a figure with one main plotting axis and one clean right-side
    information axis, matching the gain-plot style.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[3.25, 1.05],
        left=0.10,
        right=0.98,
        bottom=0.13,
        top=0.90,
        wspace=0.12,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    ax_info.axvline(0.0, color="0.70", linewidth=0.8)
    return fig, ax, ax_info


def choose_empty_legend_corner(ax):
    """
    Choose a legend corner using the plotted line data.

    This keeps legends attached to the plot while avoiding the densest part of
    the waveform or histogram curves. If no line data are available, the upper
    right corner is usually the least intrusive for these afterpulse plots.
    """
    candidates = [
        ("upper right", (0.60, 1.00, 0.66, 1.00)),
        ("upper left", (0.00, 0.40, 0.66, 1.00)),
        ("lower right", (0.60, 1.00, 0.00, 0.34)),
        ("lower left", (0.00, 0.40, 0.00, 0.34)),
    ]

    points = []
    to_axes = ax.transAxes.inverted()

    for line in ax.lines:
        x = np.asarray(line.get_xdata(orig=False), dtype=float)
        y = np.asarray(line.get_ydata(orig=False), dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            continue
        xy_display = ax.transData.transform(np.column_stack([x[mask], y[mask]]))
        xy_axes = to_axes.transform(xy_display)
        points.append(xy_axes)

    if not points:
        return "upper right"

    points = np.vstack(points)
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite]

    if points.size == 0:
        return "upper right"

    scores = []
    for loc, (x0, x1, y0, y1) in candidates:
        inside = (
            (points[:, 0] >= x0)
            & (points[:, 0] <= x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] <= y1)
        )
        scores.append((int(np.count_nonzero(inside)), loc))

    return min(scores, key=lambda item: item[0])[1]


def add_info_and_legend(ax, ax_info, info_text, legend=True, legend_y=0.02, legend_loc="auto"):
    """
    Put scientific text in the right-side column and the legend inside
    the plotting axes.
    """
    ax_info.text(
        0.05,
        0.98,
        info_text,
        transform=ax_info.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.25,
    )

    if not legend:
        return None

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    if legend_loc == "auto":
        legend_loc = choose_empty_legend_corner(ax)

    leg = ax.legend(
        handles,
        labels,
        loc=legend_loc,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.45",
        fontsize=9.0,
        borderpad=0.40,
        labelspacing=0.36,
        handlelength=2.4,
        handletextpad=0.65,
    )
    leg.get_frame().set_linewidth(0.8)
    return leg


def save_show_close(fig, save_plots, show_plots, save_dir, filename):
    if save_plots:
        fig.savefig(os.path.join(save_dir, filename + ".png"), dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


class PlottingAfterpulses:
    """
    Class for afterpulse plots with a clean scientific right-side information
    column and an in-plot legend.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        adc_f: float,
        ap_rate_dict: dict = None,
        save_plots: bool = False,
        show_plots: bool = True,
        save_dir: Optional[str] = None,
        save_name_suffix: Optional[str] = None,
        adc_area_to_e: Optional[float] = None,
        gain: Optional[float] = None,
        pmt_serial: Optional[str] = None,
        hv_v: Optional[str] = None,
        lamp_v: Optional[str] = None,
        frame_type: Optional[str] = None,
        n_samples_per_waveform: Optional[int] = None,
        n_waveforms_analyzed: Optional[int] = None,
    ):
        self.df = df
        self.ap_rate_dict = ap_rate_dict
        self.adc_f = adc_f
        self.adc_area_to_e = adc_area_to_e
        self.gain = gain
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.save_dir = save_dir
        self.save_name_suffix = save_name_suffix

        self.pmt_serial = pmt_serial
        self.hv_v = hv_v
        self.lamp_v = lamp_v
        self.frame_type = frame_type
        self.n_samples_per_waveform = n_samples_per_waveform
        self.n_waveforms_analyzed = n_waveforms_analyzed

        if save_plots and save_dir is None:
            raise NameError("save_dir must be defined if save_plots is True.")
        if save_plots and save_name_suffix is None:
            raise NameError("save_name_suffix must be defined if save_plots is True.")

    def _plot_info_text(self) -> str:
        lines = [r"$\mathbf{Afterpulse\ analysis}$"]

        if self.pmt_serial is not None:
            lines.append(rf"$\mathrm{{PMT}} = \mathrm{{{self.pmt_serial}}}$")
        if self.gain is not None:
            lines.append(rf"$G = {self.gain:.3e}\,e^-$")
        if self.hv_v is not None:
            hv_num = str(self.hv_v).replace(" V", "")
            lines.append(rf"$V_{{\mathrm{{HV}}}} = {hv_num}\,\mathrm{{V}}$")
        if self.lamp_v is not None:
            lamp_num = str(self.lamp_v).replace(" Vpp", "")
            lines.append(rf"$V_{{\mathrm{{lamp}}}} = {lamp_num}\,\mathrm{{V_{{pp}}}}$")
        if self.n_samples_per_waveform is not None:
            lines.append(rf"$N_{{\mathrm{{frame}}}} = {self.n_samples_per_waveform}$")
        if self.n_waveforms_analyzed is not None:
            lines.append(rf"$N_{{\mathrm{{wf,plot}}}} = {self.n_waveforms_analyzed}$")

        if self.ap_rate_dict is not None:
            lines.append("")
            #lines.append(r"$\mathbf{Selection}$")
            # if self.ap_rate_dict.get("area_thr_ap") is not None:
            #     lines.append(
            #         rf"$A_{{\mathrm{{thr}}}} = {self.ap_rate_dict['area_thr_ap']}\,\mathrm{{PE}}$"
            #     )
                
            # if self.ap_rate_dict.get("t_thr_ap") is None:
            #     lines.append(r"$t_{\mathrm{thr}} = \mathrm{None}$")
            # else:
            #     lines.append(rf"$t_{{\mathrm{{thr}}}} = {self.ap_rate_dict['t_thr_ap']}\,\mathrm{{ns}}$")

            if self.ap_rate_dict.get("ap_rate_per_pe_separable") is not None:
                lines.append("")
                lines.append(r"$\mathbf{Afterpulse\ rate}$")
                lines.append(
                    rf"$R_{{\mathrm{{AP}}}}/\mathrm{{PE}} = "
                    rf"{self.ap_rate_dict['ap_rate_per_pe_separable']:.3e}$"
                )

        return "\n".join(lines)

    def _finish(self, fig, ax, ax_info, filename, legend=True, legend_y=0.02, legend_loc="auto"):
        add_info_and_legend(
            ax=ax,
            ax_info=ax_info,
            info_text=self._plot_info_text(),
            legend=legend,
            legend_y=legend_y,
            legend_loc=legend_loc,
        )
        save_show_close(fig, self.save_plots, self.show_plots, self.save_dir, filename)

    def plot_wf(self, i: int = 0):
        """Plot i-th afterpulse candidate waveform."""
        if (i >= self.df.shape[0]) or (i < 0):
            raise IndexError(
                "Integer-location based index i must be between 0 and {}".format(
                    self.df.shape[0] - 1
                )
            )

        fig, ax, ax_info = make_scientific_figure()

        row = self.df.iloc[i]
        separability = "separable" if row["separable"] else "non-separable"
        x_dummy = np.arange(len(row["input_data_converted"])) / self.adc_f * 1e9

        ax.step(x_dummy, row["input_data_converted"], where="mid", linewidth=1.2, label="Waveform")
        ax.axvline(
            x=row["p0_position"] / self.adc_f * 1e9,
            c="gray",
            linestyle="dashed",
            zorder=-1,
            label=rf"Peak positions, $\Delta t = {row['t_diff_ns']}\,\mathrm{{ns}}$",
        )
        ax.axvline(
            x=row["p1_position"] / self.adc_f * 1e9,
            c="gray",
            linestyle="dashed",
            zorder=-1,
        )
        ax.axvspan(
            row["p0_lower_bound"] / self.adc_f * 1e9 - 0.5,
            row["p0_upper_bound"] / self.adc_f * 1e9,
            color="C1",
            lw=0,
            alpha=0.35,
            zorder=-2,
            label="Main pulse",
        )
        ax.axvspan(
            row["p1_lower_bound"] / self.adc_f * 1e9,
            row["p1_upper_bound"] / self.adc_f * 1e9 + 0.5,
            color="C3",
            lw=0,
            alpha=0.35,
            zorder=-2,
            label=f"Afterpulse ({separability})",
        )

        t_max_ns = len(row["input_data_converted"]) / self.adc_f * 1e9
        ax.set_xlim(0, t_max_ns)
        ax.set_xlabel(r"Time $t\,[\mathrm{ns}]$")
        ax.set_ylabel(r"Amplitude $[\mathrm{ADC}]$")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Afterpulse candidate waveform #{i} from PMT {self.pmt_serial}")

        filename = "ap_candidate_wf_example_{}_{}".format(i, self.save_name_suffix)
        self._finish(fig, ax, ax_info, filename, legend=True, legend_loc="auto")

    def plot_first_n_wfs(self, n: int = 3):
        """Plot first n afterpulse candidate waveforms."""
        for i in range(min(n, self.df.shape[0])):
            self.plot_wf(i)

    def plot_hist_tdiff(self):
        """Plot time differences afterpulse - main pulse."""
        fig, ax, ax_info = make_scientific_figure()

        step_ns = int(1e9 / self.adc_f)
        x_dummy = np.arange(0, int(self.df["t_diff_ns"].max()), step=step_ns)

        n_all, bins_edges, _ = ax.hist(
            self.df["t_diff_ns"],
            bins=x_dummy - 0.5,
            histtype="step",
            color="C0",
            label="All afterpulse candidates",
        )
        bins_centers = (bins_edges[1:] + bins_edges[:-1]) / 2
        ax.fill_between(
            bins_centers,
            n_all - np.sqrt(n_all),
            n_all + np.sqrt(n_all),
            color="C0",
            alpha=0.35,
            zorder=-1,
        )

        n_sep, _, _ = ax.hist(
            self.df[self.df["separable"]]["t_diff_ns"],
            bins=x_dummy - 0.5,
            histtype="step",
            color="C1",
            label="Separable afterpulse candidates",
            linestyle="dashed",
        )
        ax.fill_between(
            bins_centers,
            n_sep - np.sqrt(n_sep),
            n_sep + np.sqrt(n_sep),
            color="C1",
            alpha=0.35,
            zorder=-1,
        )

        ax.set_xlabel(r"Time difference $\Delta t\,[\mathrm{ns}]$")
        ax.set_ylabel("Entries")
        ax.set_yscale("log")
        ax.set_xlim(right=x_dummy[-1])
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"Time difference histogram for PMT {self.pmt_serial}")

        filename = "ap_tdiff_{}".format(self.save_name_suffix)
        self._finish(fig, ax, ax_info, filename, legend=True, legend_loc="auto")

    def plot_ap_area_hist(self, xmax=10, binsize=0.1, separable_overlay=True, show_thr=True):
        """Plot histogram of afterpulse area in PE."""
        fig, ax, ax_info = make_scientific_figure()

        well_defined = True
        if self.adc_area_to_e is None:
            warnings.warn(
                "Attribute adc_area_to_e set to None, will default to value 3047.6 for ADC V1730D."
            )
            adc_area_to_e = 3047.6
            well_defined = False
        else:
            adc_area_to_e = self.adc_area_to_e

        if self.gain is None:
            warnings.warn("Attribute gain set to None, will default to value 3e6.")
            gain = 3e6
            well_defined = False
        else:
            gain = self.gain

        df_plot = self.df.copy()
        df_plot["p1_area_conv"] = df_plot["p1_area"] * adc_area_to_e / gain
        bins = np.arange(0, xmax + binsize, binsize)

        df_plot_all = df_plot[
            (df_plot["p1_area_conv"] >= 0) & (df_plot["p1_area_conv"] <= xmax)
        ]

        ax.hist(
            df_plot_all["p1_area_conv"],
            bins=bins,
            histtype="step",
            label="All afterpulse candidates",
        )

        if separable_overlay:
            df_plot_sep = df_plot[df_plot["separable"]]
            df_plot_sep = df_plot_sep[
                (df_plot_sep["p1_area_conv"] >= 0) & (df_plot_sep["p1_area_conv"] <= xmax)
            ]
            ax.hist(
                df_plot_sep["p1_area_conv"],
                bins=bins,
                histtype="step",
                label="Separable afterpulse candidates",
                linestyle="dashed",
            )

        if show_thr and self.ap_rate_dict is not None and self.ap_rate_dict.get("area_thr_ap") is not None:
            ax.axvline(
                self.ap_rate_dict["area_thr_ap"],
                color="gray",
                linestyle="dashed",
                label=r"$A_{\mathrm{thr}}$",
            )

        if well_defined:
            ax.set_xlabel(r"Afterpulse area $A_{\mathrm{AP}}\,[\mathrm{PE}]$")
        else:
            ax.set_xlabel(r"Afterpulse area $[\mathrm{A.U.}]$")

        ax.set_ylabel("Entries")
        ax.set_xlim(0, xmax)
        ax.set_yscale("log")
        ax.set_title(f"Afterpulse area histogram for PMT {self.pmt_serial}")
        ax.grid(True, which="both", alpha=0.25)

        filename = f"ap_area_hist_{self.save_name_suffix}"
        self._finish(fig, ax, ax_info, filename, legend=True, legend_loc="auto")

    def plot_ap_area_vs_tdiff(self, separable_only=True, ymax=25, xmax=2.0, show_thr=True):
        """Plot afterpulse area vs time difference histogram."""
        fig, ax, ax_info = make_scientific_figure()

        well_defined = True
        if self.adc_area_to_e is None:
            warnings.warn(
                "Attribute adc_area_to_e set to None, will default to value 3047.6 for ADC V1730D."
            )
            adc_area_to_e = 3047.6
            well_defined = False
        else:
            adc_area_to_e = self.adc_area_to_e

        if self.gain is None:
            warnings.warn("Attribute gain set to None, will default to value 3e6.")
            gain = 3e6
            well_defined = False
        else:
            gain = self.gain

        df_plot = self.df.copy()
        df_plot["p1_area_conv"] = df_plot["p1_area"] * adc_area_to_e / gain
        df_plot["t_diff_us"] = df_plot["t_diff_ns"] / 1e3

        binsx = np.arange(0, xmax + 0.02, 0.02)
        binsy = np.arange(0, ymax + 0.2, 0.2)

        if separable_only:
            df_plot = df_plot[df_plot["separable"]]

        df_plot = df_plot[(df_plot["t_diff_us"] >= 0) & (df_plot["t_diff_us"] <= xmax)]
        df_plot = df_plot[(df_plot["p1_area_conv"] >= 0) & (df_plot["p1_area_conv"] <= ymax)]

        if len(df_plot) > 0:
            hist = ax.hist2d(
                df_plot["t_diff_us"],
                df_plot["p1_area_conv"],
                bins=(binsx, binsy),
                norm=matplotlib.colors.LogNorm(),
            )
            fig.colorbar(hist[3], ax=ax, pad=0.02, label="Entries")
        else:
            warnings.warn("No entries available inside requested plotting range.")

        if show_thr and self.ap_rate_dict is not None:
            if self.ap_rate_dict.get("area_thr_ap") is not None:
                ax.axhline(y=self.ap_rate_dict["area_thr_ap"], color="gray", linestyle="dashed")
            if self.ap_rate_dict.get("t_thr_ap") is not None:
                ax.axvline(x=self.ap_rate_dict["t_thr_ap"] / 1e3, color="gray", linestyle="dashed")

        ax.set_xlabel(r"Time difference $\Delta t\,[\mu\mathrm{s}]$")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_title(f"Afterpulse area vs time difference for PMT {self.pmt_serial}")

        if well_defined:
            ax.set_ylabel(r"Afterpulse area $A_{\mathrm{AP}}\,[\mathrm{PE}]$")
        else:
            ax.set_ylabel(r"Afterpulse area $[\mathrm{A.U.}]$")

        ax.grid(False)

        if separable_only:
            filename = f"ap_area_vs_tdiff_separable_{self.save_name_suffix}"
        else:
            filename = f"ap_area_vs_tdiff_{self.save_name_suffix}"

        self._finish(fig, ax, ax_info, filename, legend=False)

    def plot_essentials(self):
        """Plot essential plots for afterpulse studies."""
        self.plot_first_n_wfs(n=5)
        self.plot_hist_tdiff()
        self.plot_ap_area_hist(xmax=10, binsize=0.1)
        self.plot_ap_area_vs_tdiff(separable_only=False, ymax=25, xmax=2.0)
        self.plot_ap_area_vs_tdiff(separable_only=True, ymax=25, xmax=2.0)


def add_summary_column_to_existing_axes(fig, ax, summary_text, legend=True):
    """
    Helper for your final summary plots.

    Use this instead of fig.subplots_adjust(right=0.64) + add_summary_box_right.
    It keeps the same gain-plot style: plot on left, unboxed information and
    a legend inside the plotting axes.
    """
    fig.set_size_inches(10.0, 6.0, forward=True)
    fig.subplots_adjust(left=0.10, right=0.70, bottom=0.18, top=0.90)
    ax_info = fig.add_axes([0.73, 0.18, 0.24, 0.72])
    ax_info.axis("off")
    ax_info.axvline(0.0, color="0.70", linewidth=0.8)
    add_info_and_legend(ax, ax_info, summary_text, legend=legend, legend_y=0.02)
    return ax_info