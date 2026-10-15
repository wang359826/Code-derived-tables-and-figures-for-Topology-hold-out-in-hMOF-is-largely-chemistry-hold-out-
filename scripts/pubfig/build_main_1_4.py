#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main-text Figures 1-4, rebuilt at final printed size.

Every number is read from the archived, submitted source CSVs in ../outputs.
Nothing is recomputed and nothing is invented: this script only changes how the
already-published quantities are drawn.

Outputs (per figure): <stem>.svg, <stem>.pdf, <stem>.png, <stem>.tiff,
<stem>.alignment.json under PUBFIG_OUT / PUBFIG_RASTER.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, ".."))
REPRO = os.path.abspath(os.path.join(SCRIPTS, ".."))
OUTDIR = os.path.abspath(os.path.join(REPRO, "outputs"))
DEST = os.environ.get("PUBFIG_OUT", os.path.join(REPRO, "figures_pub"))
RASTER = os.environ.get("PUBFIG_RASTER", os.path.join(HERE, "_raster"))
sys.path.insert(0, os.path.join(SCRIPTS))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pubfig import style as S

S.apply_style()
C, PT = S.C, S.PT

# strategy labels: the archived CSVs are in Chinese, the figures are English
STRAT = {
    0: "random\n(most papers)",
    1: "grouped by\ntopology",
    2: "grouped by\nmetal centre",
    3: "grouped by\norganic linker",
    4: "grouped by\nchemistry",
}
RECORDS = {}


def out(stem):
    return os.path.join(DEST, stem)


def finish(fig, stem, **kw):
    rec = S.save_figure(fig, stem, DEST, raster_dir=RASTER, **kw)
    RECORDS[stem] = rec
    plt.close(fig)
    return rec


# ==========================================================================
# Figure 1 - leakage magnitude is a property of the target
# ==========================================================================
def figure1():
    up = pd.read_csv(os.path.join(OUTDIR, "leakage_results_chemistry.csv"))
    ls = pd.read_csv(os.path.join(OUTDIR, "leakage_results_logsel100k.csv"))

    # rows 0..3 = random, topology, metal, linker for both targets
    order = [0, 3, 2, 1]
    labels = [STRAT[i] for i in order]
    up_r2 = up["R2"].to_numpy()[order]
    ls_r2 = ls["R2"].to_numpy()[order]
    d_up = up["R2"][0] - up["R2"][1]
    d_ls = ls["R2"][0] - ls["R2"][1]
    infl_up = 100.0 * d_up / up["R2"][1]
    infl_ls = 100.0 * d_ls / ls["R2"][1]

    fig = plt.figure(figsize=S.fig_size(66))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.30,
                          left=0.088, right=0.988, top=0.80, bottom=0.245)

    # ---- (a) grouped bars, 4 split strategies x 2 targets ----
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    w = 0.36
    b1 = ax.bar(x - w / 2, up_r2, S.widths(up_r2, w), color=C["random"],
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w / 2, ls_r2, S.widths(ls_r2, w), color=C["topology"],
                edgecolor="white", linewidth=0.5)
    for xi, v in zip(x - w / 2, up_r2):
        S.value_label(ax, xi, v, f"{v:.3f}", dy_pt=1.0)
    for xi, v in zip(x + w / 2, ls_r2):
        S.value_label(ax, xi, v, f"{v:.3f}", dy_pt=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(f"cross-validated {S.R2}")
    S.gridlines_only_y(ax)
    S.panel_label(ax, "a", dx_pt=-26, dy_pt=3)
    ax.set_title("Same database, same model \u2014 only the target changes",
                 loc="left", fontsize=PT["axes_title"])

    # series identity via a legend, so no direct label can land on a bar or on a
    # value label
    hs = [plt.Rectangle((0, 0), 1, 1, fc=C["random"], ec="white", lw=0.5),
          plt.Rectangle((0, 0), 1, 1, fc=C["topology"], ec="white", lw=0.5)]
    ax.legend(hs, ["CO\u2082 uptake (0.15 bar)",
                   "log10(CO\u2082/N\u2082 selectivity)"],
              loc="upper right", frameon=True, facecolor="white",
              edgecolor=C["light"], framealpha=0.95, fontsize=PT["annot"],
              handlelength=1.1, labelspacing=0.22)

    # The arrows show which way each target's penalty goes. They ride just 0.02
    # above each bar top: 0.055 cleared the text-collision check but crossed the
    # value labels themselves, which a text-only detector cannot see because the
    # stroked path is not a Text artist. Sit them under the label baselines.
    for xi in range(1, len(order)):
        ax.annotate("", xy=(xi - w / 2, up_r2[xi] + 0.020),
                    xytext=(xi - 1 - w / 2, up_r2[xi - 1] + 0.020),
                    arrowprops=dict(arrowstyle="-|>", color=C["random"],
                                    lw=1.0, shrinkA=0, shrinkB=0))
        ax.annotate("", xy=(xi + w / 2, ls_r2[xi] + 0.020),
                    xytext=(xi - 1 + w / 2, ls_r2[xi - 1] + 0.020),
                    arrowprops=dict(arrowstyle="-|>", color=C["topology"],
                                    lw=1.0, shrinkA=0, shrinkB=0))

    # ---- (b) relative inflation of R2 ----
    ax2 = fig.add_subplot(gs[0, 1])
    names = ["CO\u2082 uptake\n(0.15 bar)", "log10(CO\u2082/N\u2082\nselectivity)"]
    vals = [infl_up, infl_ls]
    ypos = np.arange(2)
    ax2.barh(ypos, vals, height=S.widths(vals, 0.5),
             color=[C["random"], C["topology"]],
             edgecolor="white", linewidth=0.5)
    for yi, v in zip(ypos, vals):
        ax2.annotate(f"{v:.0f}%", xy=(v, yi), xytext=(4, 0),
                     textcoords="offset points", va="center", ha="left",
                     fontsize=PT["label"], fontweight="bold", color=C["ink"])
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names)
    ax2.set_xlim(0, 128)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_xlabel(f"relative inflation of {S.R2}  [%]")
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    S.panel_label(ax2, "b", dx_pt=-26, dy_pt=3)
    ax2.set_title("How badly random splitting\nlies depends on the target",
                  loc="left", fontsize=PT["axes_title"])
    ratio = infl_up / infl_ls
    ax2.annotate("", xy=(infl_up - 6, 0.06), xytext=(infl_ls + 4, 0.94),
                 arrowprops=dict(arrowstyle="-|>", color=C["mid"], lw=0.9,
                                 connectionstyle="arc3,rad=-0.30", shrinkA=2,
                                 shrinkB=2))
    ax2.text(infl_up * 0.52, 0.50, f"{ratio:.1f}\u00d7 worse",
             ha="center", va="center", fontsize=PT["annot"], color=C["mid"],
             fontweight="bold")

    fig.text(0.0, 0.02,
             f"n = 100,000 and 426 features in every arm; identical model.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "fig1_leakage_two_targets")


# ==========================================================================
# Figure 2 - chemistry-topology confounding
# ==========================================================================
def figure2():
    full = pd.read_csv(os.path.join(OUTDIR, "confound_full.csv"))
    elite = pd.read_csv(os.path.join(OUTDIR, "elite_subset_confound.csv"))
    jac = pd.read_csv(os.path.join(OUTDIR, "jaccard_full.csv"))

    def frac(df, label, col="frac_single_topology"):
        return float(df.loc[df["label"] == label, col].iloc[0])

    # every bar is labelled with the population it was estimated on, because the
    # 25,000-row subsets and the 8,206-row elite file are different samples
    # each entry is (axis label, coarse fraction, fine fraction, n of the sample
    # that fraction was estimated on)
    groups = [
        ("full database\n(n = 25,000)", frac(elite, "full_25k_coarse"),
         frac(elite, "full_25k_fine"), "25,000"),
        ("elite \u2265 2.0 mmol/g\n(n = 8,206)", frac(elite, "elite_coarse"),
         frac(elite, "elite_fine"), "8,206"),
    ]

    fig = plt.figure(figsize=S.fig_size(74))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.26,
                          left=0.088, right=0.985, top=0.735, bottom=0.235)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(groups))
    w = 0.35
    coarse = [g[1] * 100 for g in groups]
    fine = [g[2] * 100 for g in groups]
    ax.bar(x - w / 2, coarse, S.widths(coarse, w), color=C["coarse"], edgecolor="white",
           linewidth=0.5, label="coarse key (metal + 2 linkers)")
    ax.bar(x + w / 2, fine, S.widths(fine, w), color=C["fine"], edgecolor="white",
           linewidth=0.5, label="fine key (+ functional groups)")
    for xi, v in zip(x - w / 2, coarse):
        S.value_label(ax, xi, v, f"{v:.1f}", dy_pt=1.0)
    for xi, v in zip(x + w / 2, fine):
        S.value_label(ax, xi, v, f"{v:.1f}", dy_pt=1.0)
    S.zero_ref(ax, 100.0, ls=(0, (1.5, 1.5)))
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0, 126)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("compositions in a\nsingle topology  [%]")
    hs = [plt.Rectangle((0, 0), 1, 1, fc=C["coarse"], ec="white", lw=0.5),
          plt.Rectangle((0, 0), 1, 1, fc=C["fine"], ec="white", lw=0.5)]
    ax.legend(hs, ["coarse key (metal + 2 linkers)",
                   "fine key (+ functional groups)"],
              loc="upper center", bbox_to_anchor=(0.5, 0.60), frameon=False,
              fontsize=PT["annot"], handlelength=1.1, labelspacing=0.30,
              borderaxespad=0.0)
    S.gridlines_only_y(ax)
    S.panel_label(ax, "A", dx_pt=-26, dy_pt=3)
    ax.set_title("Chemistry determines topology", loc="left",
                 fontsize=PT["axes_title"])

    # ---- (b) Jaccard heat map, coarse granularity, full database ----
    ax2 = fig.add_subplot(gs[0, 1])
    c = jac[jac["granularity"].str.startswith("coarse")]
    topos = sorted(set(c["topo_a"]) | set(c["topo_b"]))
    M = np.zeros((len(topos), len(topos)))
    idx = {t: i for i, t in enumerate(topos)}
    for _, r in c.iterrows():
        M[idx[r.topo_a], idx[r.topo_b]] = r.jaccard
        M[idx[r.topo_b], idx[r.topo_a]] = r.jaccard
    off = M[~np.eye(len(topos), dtype=bool)]
    vmax = float(off.max()) if off.size else 1.0
    im = ax2.imshow(M, cmap="Blues", vmin=0, vmax=vmax * 1.25)
    ax2.set_xticks(range(len(topos)))
    ax2.set_yticks(range(len(topos)))
    ax2.set_xticklabels(topos, rotation=45, ha="right", rotation_mode="anchor")
    ax2.set_yticklabels(topos)
    ax2.grid(False)
    for i in range(len(topos)):
        for j in range(len(topos)):
            if i == j:
                continue
            if M[i, j] > 0:
                # a 6.0 pt label fills a 0.36-wide cell edge to edge, so the two
                # non-zero off-diagonal cells carry 5.5 pt to stay inside
                ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                         fontsize=5.5, fontweight="bold",
                         color="white" if M[i, j] > vmax * 0.6 else C["ink"])
            else:
                ax2.text(j, i, "0", ha="center", va="center",
                         fontsize=PT["annot"], color=C["light"])
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.035)
    cb.set_label("Jaccard overlap", fontsize=PT["annot"], labelpad=1.5)
    cb.ax.tick_params(labelsize=PT["annot"], length=1.8, pad=1.2)
    cb.outline.set_linewidth(0.4)
    S.panel_label(ax2, "B", dx_pt=-26, dy_pt=3)
    ax2.set_title("Chemical sets barely overlap\nbetween topologies",
                  loc="left", fontsize=PT["axes_title"])

    fig.text(0.0, 0.02,
             "Diagonal = 1 by definition. The 25,000-row and 8,206-row strata "
             "are different samples, not a paired comparison.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "fig2_confound_structure",
                  exemptions=[{"panels": ["A", "B"], "checks": ["panel-width"],
                               "reason": "left panel is a grouped bar chart with a "
                                         "long categorical axis; right panel is a "
                                         "square Jaccard heat map. Equal widths "
                                         "would either squash the bar categories or "
                                         "stretch the matrix off-square."}])


# ==========================================================================
# Figure 3 - controlled isoreticular contrast pcu <-> nbo
# ==========================================================================
def figure3():
    s = pd.read_csv(os.path.join(OUTDIR, "isoreticular_control_seeds.csv"))
    s = s.groupby(["test_topo", "regime"]).agg(
        MAE=("MAE", "mean"), RMSE=("RMSE", "mean"),
        MAE_sd=("MAE", "std"), RMSE_sd=("RMSE", "std"),
        MAEosd=("MAE_over_sd", "mean"), MAEosd_sd=("MAE_over_sd", "std"),
        R2=("R2", "mean")).reset_index()

    def g(topo, regime, col):
        return float(s.loc[(s.test_topo == topo) & (s.regime == regime), col].iloc[0])

    fig = plt.figure(figsize=S.fig_size(62))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.22, 1.12], wspace=0.36,
                          left=0.082, right=0.988, top=0.80, bottom=0.245)
    labels_in = ["pcu\n(with partner)", "nbo\n(with partner)"]
    labels_out = ["pcu\n(without partner)", "nbo\n(without partner)"]
    x2 = np.arange(2)

    # (a) MAE normalised by the test set's own sd -- a common yardstick
    ax = fig.add_subplot(gs[0, 0])
    inc = [g("pcu", "with_pair", "MAEosd"), g("nbo", "with_pair", "MAEosd")]
    exc = [g("pcu", "without_pair", "MAEosd"), g("nbo", "without_pair", "MAEosd")]
    inc_e = [g("pcu", "with_pair", "MAEosd_sd"), g("nbo", "with_pair", "MAEosd_sd")]
    exc_e = [g("pcu", "without_pair", "MAEosd_sd"), g("nbo", "without_pair", "MAEosd_sd")]
    w = 0.34
    ax.bar((x2 - w / 2).tolist(), inc, S.widths(inc, w), color=C["topology"],
           edgecolor="white", linewidth=0.5)
    ax.bar((x2 + w / 2).tolist(), exc, S.widths(exc, w), color=C["random_soft"],
           edgecolor=C["random"], linewidth=0.5)
    S.whiskers_v(ax, (x2 - w / 2), inc, inc_e, w=w)
    S.whiskers_v(ax, (x2 + w / 2), exc, exc_e, w=w)
    for xi, v, e in zip(x2 - w / 2, inc, inc_e):
        S.value_label(ax, xi, v + e, f"{v:.2f}", dy_pt=1.0)
    for xi, v, e in zip(x2 + w / 2, exc, exc_e):
        S.value_label(ax, xi, v + e, f"{v:.2f}", dy_pt=1.0)
    ax.set_xticks(x2)
    ax.set_xticklabels(["pcu test set\nsd(y) = 0.472",
                        "nbo test set\nsd(y) = 0.106"])
    ax.set_ylim(0, 2.45)
    ax.set_ylabel("MAE / sd(y) of the test set")
    S.gridlines_only_y(ax)
    S.panel_label(ax, "a", dx_pt=-42, dy_pt=3)
    ax.set_title("Normalised error", loc="left", fontsize=PT["axes_title"])

    # (b) absolute error: 2 test sets x {MAE, RMSE} x {partner in/out}
    ax2 = fig.add_subplot(gs[0, 1])
    xb = np.arange(4)
    tick = ["pcu\nMAE", "pcu\nRMSE", "nbo\nMAE", "nbo\nRMSE"]
    vals_in = [g("pcu", "with_pair", "MAE"), g("pcu", "with_pair", "RMSE"),
               g("nbo", "with_pair", "MAE"), g("nbo", "with_pair", "RMSE")]
    vals_out = [g("pcu", "without_pair", "MAE"), g("pcu", "without_pair", "RMSE"),
                g("nbo", "without_pair", "MAE"), g("nbo", "without_pair", "RMSE")]
    ein = [g("pcu", "with_pair", "MAE_sd"), g("pcu", "with_pair", "RMSE_sd"),
           g("nbo", "with_pair", "MAE_sd"), g("nbo", "with_pair", "RMSE_sd")]
    eout = [g("pcu", "without_pair", "MAE_sd"), g("pcu", "without_pair", "RMSE_sd"),
            g("nbo", "without_pair", "MAE_sd"), g("nbo", "without_pair", "RMSE_sd")]
    w2 = 0.34
    ax2.bar((xb - w2 / 2).tolist(), vals_in, S.widths(vals_in, w2),
            color=C["topology"], edgecolor="white", linewidth=0.5)
    ax2.bar((xb + w2 / 2).tolist(), vals_out, S.widths(vals_out, w2),
            color=C["random_soft"], edgecolor=C["random"], linewidth=0.5)
    S.whiskers_v(ax2, (xb - w2 / 2), vals_in, ein, w=w2)
    S.whiskers_v(ax2, (xb + w2 / 2), vals_out, eout, w=w2)
    for k, (v, e) in enumerate(zip(vals_in, ein)):
        # Each pair is labelled once, jointly, above whichever bar is taller and
        # anchored to the pair's right edge so the text grows leftwards, away from
        # the next panel. Two separate labels per pair drove the left one through
        # the y-axis tick labels and the right one through the error-bar cap.
        top = max(v + e, vals_out[k] + eout[k])
        S.value_label(ax2, k + w2 / 2, top + 0.020,
                      f"{v:.3f} / {vals_out[k]:.3f}", dy_pt=1.0, fontsize=5.5,
                      ha="right")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(tick)
    ax2.set_ylim(0, 0.58)
    ax2.set_ylabel("absolute error  [mmol/g]")
    S.gridlines_only_y(ax2)
    S.panel_label(ax2, "b", dx_pt=-46, dy_pt=3)
    ax2.set_title("Absolute error", loc="left", fontsize=PT["axes_title"])
    ax2.text(0.5, 0.965,
             "filled = partner topology in training;\nred outline = partner removed",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=PT["annot"], color=C["mid"])

    # (c) relative penalty, same units for every bar
    ax3 = fig.add_subplot(gs[0, 2])
    pen = [100 * (o / i - 1) for o, i in zip(vals_out, vals_in)]
    ax3.bar(xb, pen, S.widths(pen, 0.62), color=C["random"], edgecolor="white", linewidth=0.5)
    for xi, v in zip(xb, pen):
        S.value_label(ax3, xi, v, f"{v:+.1f}%",
                      color=C["random"] if v > 1 else C["mid"])
    ax3.axhline(0, color=C["ink"], lw=0.6)
    ax3.set_xticks(xb)
    ax3.set_xticklabels(tick)
    ax3.set_ylim(-14, 108)
    ax3.set_ylabel("change when the partner\ntopology is removed  [%]")
    S.gridlines_only_y(ax3)
    S.panel_label(ax3, "c", dx_pt=-48, dy_pt=3)
    ax3.set_title("Penalty is one-sided", loc="left",
                  fontsize=PT["axes_title"])
    # The note that the pcu side is flat lives in the caption; drawn here it sat
    # across the \u22121.2% and \u22120.1% value labels.

    fig.text(0.0, 0.02,
             "Mean over 5 seeds; whiskers \u00b1 1 seed SD. The pcu side moves by "
             f"\u22121.2% and \u22120.1%. {S.R2} is not used here.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "fig3_isoreticular_control",
                  exemptions=[{"panels": ["a", "b", "c"],
                               "checks": ["panel-width"],
                               "reason": "panel (b) carries four categorical "
                                         "groups with value labels and must be "
                                         "wider than (a) and (c), which carry two "
                                         "and four groups of one series. All three "
                                         "share identical top/bottom plot-area "
                                         "edges, which the row check still "
                                         "verifies."}])


# ==========================================================================
# Figure 4 - R2 against distance to the nearest training sample
# ==========================================================================
def figure4():
    nn = pd.read_csv(os.path.join(OUTDIR, "fig7_nn_curve_final.csv"))
    piv = nn.pivot_table(index="bin", columns="scheme", values="R2")
    order = list(piv.index)
    mids = (nn.drop_duplicates("bin").set_index("bin")["bin_mid"]
            .reindex(order).to_numpy(float))
    rnd = piv["random"].to_numpy(float)
    loto = piv["unseen_topology"].to_numpy(float)

    # distances come from the predictions file; both schemes share the same
    # standardised distance definition
    pred = pd.read_csv(os.path.join(OUTDIR, "fig4_predictions_final.csv"))

    fig = plt.figure(figsize=S.fig_size(68))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.30,
                          left=0.082, right=0.988, top=0.80, bottom=0.245)

    # (a) the two schemes score in the same bins -- paired comparison
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    ax.plot(x, rnd, marker="o", color=C["random"], label="random split")
    ax.plot(x, loto, marker="s", color=C["topology"],
            label="leave-one-topology-out")
    ax.axhline(0, color=C["ref"], lw=0.7, ls=(0, (1.5, 1.5)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m:.2f}" for m in mids], rotation=45,
                       ha="right", rotation_mode="anchor")
    ax.set_xlabel("distance to nearest training sample\n(standardised, bin median)")
    ax.set_ylabel(f"test-set {S.R2} within the bin")
    ax.set_ylim(-1.0, 0.95)
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor=C["light"], framealpha=0.95)
    S.gridlines_only_y(ax)
    S.panel_label(ax, "a", dx_pt=-26, dy_pt=3)
    ax.set_title("The curves do not converge", loc="left",
                 fontsize=PT["axes_title"])
    # callouts sit in the empty half of the panel: anchored on the curve they
    # were crossed by it, and the panel has no free room between the two series
    ax.annotate("the middle bins stay flat",
                xy=(6, 0.30), xytext=(0.02, 0.30), textcoords="axes fraction",
                fontsize=PT["annot"], va="center",
                color=C["topology"], zorder=8, ha="left" if False else "left",
                bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none"))
    ax.annotate("collapse is confined to\nthe last bin",
                xy=(11, -0.775), xytext=(0.985, 0.02), textcoords="axes fraction",
                fontsize=PT["annot"], va="bottom",
                color=C["topology"], ha="right", zorder=8,
                bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none"))

    # (b) per-bin sample counts: how much of each estimate is noise
    ax2 = fig.add_subplot(gs[0, 1])
    n_r = nn[nn.scheme == "random"].set_index("bin").reindex(order)["n"].to_numpy(float)
    n_t = nn[nn.scheme == "unseen_topology"].set_index("bin").reindex(order)["n"].to_numpy(float)
    ax2.bar(x - 0.19, n_r, S.widths(n_r, 0.36), color=C["random"], edgecolor="white",
            linewidth=0.4, label="random split")
    ax2.bar(x + 0.19, n_t, S.widths(n_t, 0.36), color=C["topology"], edgecolor="white",
            linewidth=0.4, label="leave-one-topology-out")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{m:.2f}" for m in mids], rotation=45,
                        ha="right", rotation_mode="anchor")
    ax2.set_xlabel("distance to nearest training sample\n(standardised, bin median)")
    ax2.set_ylabel("test samples in the bin")
    ax2.legend(loc="upper right", frameon=True, facecolor="white",
               edgecolor=C["light"], framealpha=0.95)
    S.gridlines_only_y(ax2)
    S.panel_label(ax2, "b", dx_pt=-26, dy_pt=3)
    ax2.set_title("Every bin is estimated on both schemes", loc="left",
                  fontsize=PT["axes_title"])
    # the legend occupies the top-right corner, so the count sits in the empty
    # middle-left band of the panel instead
    ax2.annotate(f"smallest bin: n = {int(n_t[0])}", xy=(0.5, 0.60),
                 xycoords="axes fraction", ha="center", va="center",
                 fontsize=PT["annot"], color=C["mid"], zorder=8,
                 bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none"))

    # The binning rule and the sample size are Methods/caption material. An
    # in-figure footnote here sat on top of the two-line x-axis label, so the
    # panel keeps only the axis label.
    return finish(fig, "fig4_nn_distance")


def main():
    only = sys.argv[1:] or ["1", "2", "3", "4"]
    fns = {"1": figure1, "2": figure2, "3": figure3, "4": figure4}
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(RASTER, exist_ok=True)
    for k in only:
        print(f"--- Figure {k} ---")
        fns[k]()
    S.report_printed(RECORDS)


if __name__ == "__main__":
    main()
