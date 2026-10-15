#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main-text Figures 5-8, rebuilt at final printed size.

Every number is read from the archived source CSVs in ../outputs.  Figure 5's
scatter is drawn from `fig4_predictions_final.csv` (25,000 rows per split), and a
deterministic subsample is used so the cloud stays legible; both splits are
subsampled at the same rate and the rule is recorded in the QA notes.
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

from pubfig import style as S

S.apply_style()
C, PT = S.C, S.PT

RECORDS = {}
SCATTER_N = 6000          # per split, deterministic
SCATTER_SEED = 42


def finish(fig, stem, **kw):
    rec = S.save_figure(fig, stem, DEST, raster_dir=RASTER, **kw)
    RECORDS[stem] = rec
    plt.close(fig)
    return rec


def _short(label: str) -> str:
    """Target labels as written in the manuscript, shortened for a figure axis."""
    repl = {
        "CO2 uptake 0.15 bar (post-combustion)": "CO\u2082 uptake 0.15 bar",
        "CO2 uptake 0.10 bar / 363 K (TSA regen)": "CO\u2082 uptake 0.10 bar/363 K",
        "CO2 uptake 0.70 bar / 413 K": "CO\u2082 uptake 0.70 bar/413 K",
        "excess CO2 uptake 0.15 bar": "excess CO\u2082 uptake",
        "CO2 uptake, CO2/N2 binary": "CO\u2082 uptake (binary)",
        "N2 uptake, CO2/N2 binary": "N\u2082 uptake (binary)",
        "working capacity (VSA)": "working cap. (VSA)",
        "working capacity (TSA)": "working cap. (TSA)",
        "heat of adsorption CO2": "heat of adsorption",
        "log10(CO2/N2 selectivity)": "log10(CO\u2082/N\u2082 selectivity)",
    }
    return repl.get(label, label)


# ==========================================================================
# Figure 5 - predictions against observations, random vs topology hold-out
# ==========================================================================
def figure5():
    d = pd.read_csv(os.path.join(OUTDIR, "fig4_predictions_final.csv"))
    rng = np.random.RandomState(SCATTER_SEED)

    fig = plt.figure(figsize=S.fig_size(74))
    gs = fig.add_gridspec(1, 2, wspace=0.20, left=0.085, right=0.988,
                          top=0.775, bottom=0.245)
    panels = [("random", "Random split (leaky)", C["random"], "a"),
              ("unseen_topology", "Leave-one-topology-out", C["topology"], "b")]
    lims = (0.0, 5.4)
    for i, (key, title, col, letter) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        sub = d[d["split"] == key]
        take = rng.choice(len(sub), size=min(SCATTER_N, len(sub)), replace=False)
        s = sub.iloc[np.sort(take)]
        r2 = 1 - ((sub.y_true - sub.y_pred) ** 2).sum() / \
            ((sub.y_true - sub.y_true.mean()) ** 2).sum()
        rmse = float(np.sqrt(((sub.y_true - sub.y_pred) ** 2).mean()))
        ax.scatter(s.y_true, s.y_pred, s=1.3, alpha=0.22, color=col,
                   edgecolors="none", rasterized=True)
        ax.plot(lims, lims, color=C["ink"], lw=0.8, ls=(0, (2.5, 2.0)),
                zorder=3)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("observed CO\u2082 uptake  [mmol/g]")
        if i == 0:
            ax.set_ylabel("predicted CO\u2082 uptake  [mmol/g]")
        else:
            ax.set_yticklabels([])
        S.gridlines_only_y(ax)
        ax.grid(axis="x", visible=True)
        S.panel_label(ax, letter, dx_pt=-26, dy_pt=3)
        ax.set_title(title, loc="left", fontsize=PT["axes_title"])
        ax.text(0.035, 0.965,
                f"{S.R2} = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(sub):,}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=PT["annot"], color=C["ink"],
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec=C["light"], lw=0.5))
        ax.text(0.965, 0.055, "y = x", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=PT["annot"], color=C["mid"])

    fig.text(0.0, 0.02,
             f"Same model and same 25,000 samples; only the split changes. "
             f"{SCATTER_N:,} points per panel shown, evenly subsampled.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "fig5_pred_vs_actual")


# ==========================================================================
# Figure 6 - the split effect and the identity effect across 10 targets
# ==========================================================================
def figure6():
    m = pd.read_csv(os.path.join(OUTDIR, "multitarget_confound.csv"))
    # The x axis is the R2 of the STRUCTURE-DERIVED DESCRIPTORS ONLY, which is
    # `R2_topology_geo` (the topology hold-out fit restricted to the eight scalar
    # descriptors). `R2_geometry_only` is a different quantity -- the full
    # feature set, identity included -- and using it gives r = -0.430 instead of
    # the r = -0.459 stated in the manuscript, so the column choice is load
    # bearing, not cosmetic.
    x = m["R2_topology_geo"].to_numpy(float)
    y1 = m["delta_R2"].to_numpy(float)
    y2 = m["identity_effect_topo"].to_numpy(float)
    labels = [_short(t) for t in m["label"]]

    fig = plt.figure(figsize=S.fig_size(80))
    gs = fig.add_gridspec(1, 2, wspace=0.26, left=0.115, right=0.988,
                          top=0.795, bottom=0.215)

    # Named anchors, placed as axes text rather than point-anchored labels: a
    # point-anchored offset at the edge of the data ran outside the axes and into
    # the neighbouring panel, and the two panels are only 0.26 apart.
    anchors = {
        0: [(0.045, 0.955, "top", "CO\u2082 uptake 0.70 bar/413 K"),
            (0.985, 0.035, "bottom", "excess CO\u2082 uptake")],
        1: [(0.045, 0.955, "top", "CO\u2082 uptake 0.10 bar/363 K"),
            (0.985, 0.035, "bottom", "N\u2082 uptake (binary)")],
    }
    for i, (y, ylab, title, letter, ylim) in enumerate([
        (y1, f"split effect: {S.DELTA}{S.R2} (random \u2212 topology)",
         "The split effect", "a", (-0.03, 0.62)),
        (y2, f"identity effect: {S.DELTA}{S.R2} (with \u2212 without identity)",
         "The identity effect", "b", (-0.135, 0.26)),
    ]):
        ax = fig.add_subplot(gs[0, i])
        r = float(np.corrcoef(x, y)[0, 1])
        n = len(x)
        t_stat = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
        ax.axhline(0, color=C["ref"], lw=0.7, ls=(0, (1.5, 1.5)))
        # least-squares reference line: direction only, no extrapolation
        k, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, k * xs + b0, color=C["mid"], lw=0.9, ls=(0, (3, 2)),
                zorder=2)
        ax.scatter(x, y, s=26, color=C["topology"], edgecolors="white",
                   linewidths=0.6, zorder=3)
        # Only two anchors carry a name. All ten targets sit in a narrow band of
        # x, so labelling every point puts a label on a neighbour whichever offset
        # is used; per-target detail is Figure 7, which lists all ten.
        for fx, fy, va, lab in anchors[i]:
            ax.text(fx, fy, lab, transform=ax.transAxes, ha="left", va=va,
                    fontsize=5.5, color=C["mid"])
        ax.set_xlim(-0.09, 1.09)
        ax.set_ylim(*ylim)
        ax.set_xlabel(f"{S.R2} of structure-derived descriptors alone\n"
                      "(topology hold-out, same split)")
        ax.set_ylabel(ylab)
        S.gridlines_only_y(ax)
        S.panel_label(ax, letter, dx_pt=-54, dy_pt=3)
        ax.set_title(title, loc="left", fontsize=PT["axes_title"])
        S.stats_box(ax, [f"Pearson r = {r:+.3f}",
                         f"p = {p:.4f}" if p >= 1e-4 else "p < 0.0001",
                         f"n = {n} targets"], loc="upper right")

    fig.text(0.0, 0.02,
             "Only the two extreme targets are named here; Figure 7 lists all ten. "
             "The targets are algebraically dependent, so this is an association.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "fig6_penalty_vs_geometry")


# ==========================================================================
# Figure 7 - 2x2 design across 10 targets
# ==========================================================================
def figure7():
    m = pd.read_csv(os.path.join(OUTDIR, "multitarget_confound.csv"))
    m = m.iloc[np.argsort(m["split_effect_geo"].to_numpy(float))]
    y = np.arange(len(m))
    labels = [_short(t) for t in m["label"]]

    fig = plt.figure(figsize=S.fig_size(104))
    gs = fig.add_gridspec(1, 2, wspace=0.10, left=0.175, right=0.988,
                          top=0.815, bottom=0.265)

    # (a) dumbbell: one row per target, split arms on a common R2 axis
    ax = fig.add_subplot(gs[0, 0])
    a = m["R2_random_geo"].to_numpy(float)
    b = m["R2_topology_geo"].to_numpy(float)
    for yi, (av, bv) in enumerate(zip(a, b)):
        ax.plot([bv, av], [yi, yi], color=C["light"], lw=1.4, zorder=1,
                solid_capstyle="round")
    ax.scatter(a, y, s=22, color=C["random"], edgecolors="white",
               linewidths=0.5, zorder=3, label="random split")
    ax.scatter(b, y, s=22, color=C["topology"], edgecolors="white",
               linewidths=0.5, zorder=3, label="topology hold-out")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.0)
    ax.set_ylim(-0.7, len(m) - 0.3)
    ax.set_xlim(-0.06, 1.06)
    ax.set_xlabel(f"{S.R2} with structure-derived descriptors only")
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor=C["light"], framealpha=0.95, fontsize=PT["annot"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    S.panel_label(ax, "a", dx_pt=-78, dy_pt=3)
    ax.set_title("The split moves R\u00b2 for every target", loc="left",
                 fontsize=PT["axes_title"])

    # (b) identity effect, same rows and same order
    ax2 = fig.add_subplot(gs[0, 1])
    idr = m["identity_effect_random"].to_numpy(float)
    idt = m["identity_effect_topo"].to_numpy(float)
    h = 0.34
    ax2.barh(y + h / 2, idr, S.widths(idr, h), color=C["random_soft"], edgecolor="white",
             linewidth=0.4, label="random split")
    ax2.barh(y - h / 2, idt, S.widths(idt, h), color=C["topology"], edgecolor="white",
             linewidth=0.4, label="topology hold-out")
    ax2.axvline(0, color=C["ink"], lw=0.7, zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_ylim(-0.7, len(m) - 0.3)
    ax2.set_xlim(-0.14, 0.26)
    ax2.set_xlabel(f"identity effect: {S.DELTA}{S.R2} (with \u2212 without)")
    ax2.legend(loc="lower right", frameon=True, facecolor="white",
               edgecolor=C["light"], framealpha=0.95, fontsize=PT["annot"])
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    S.panel_label(ax2, "b", dx_pt=-46, dy_pt=3)
    ax2.set_title("The identity effect is small and mixed", loc="left",
                  fontsize=PT["axes_title"])
    # median reference for both arms. The rules are drawn only across the rows
    # rather than the full panel height: at full height they ran straight through
    # the two labels that name them.
    for val, col in [(np.median(idr), C["random"]),
                     (np.median(idt), C["topology"])]:
        ax2.plot([val, val], [-0.45, len(m) - 0.55], color=col, lw=0.9,
                 ls=(0, (3, 2)), zorder=4, alpha=0.9)
    # the two medians are only 0.021 apart, so their labels are stacked clear of
    # the dashed rules rather than beside them
    for val, col, va, dy in [(np.median(idr), C["random"], "bottom", 0.42),
                             (np.median(idt), C["topology"], "top", -0.62)]:
        ax2.text(val + 0.010, len(m) - 0.45 + dy, f"median {val:+.3f}",
                 fontsize=5.5, color=col, ha="left", va=va, zorder=8,
                 bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none"))

    fig.text(0.0, 0.02,
             f"10 targets, 25,000 samples each, unpaired here so every target "
             f"keeps its own axes. The targets are algebraically dependent.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "fig7_multitarget_bars")


# ==========================================================================
# Figure 8 - replication in ARC-MOF
# ==========================================================================
def figure8():
    sz = pd.read_csv(os.path.join(OUTDIR, "sample_size_granularity.csv"))
    src = pd.read_csv(os.path.join(OUTDIR, "source_confound_granularity.csv"))
    arc = pd.read_csv(os.path.join(OUTDIR, "confound_arcmof.csv"))
    # the ladder values are stored as text ("25000") and the full-database row
    # carries a Chinese suffix, so both need normalising before selection
    sz["n_num"] = pd.to_numeric(sz["n_requested"], errors="coerce")
    sz["full_db"] = sz["n_num"].isna()

    fig = plt.figure(figsize=S.fig_size(80))
    gs = fig.add_gridspec(1, 3, wspace=0.40,
                          left=0.075, right=0.988, top=0.775, bottom=0.30)

    # (a) both databases, both granularities, at equal n
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    w = 0.34
    def pick(db, n, gran, frame):
        sub = frame[(frame.database == db) & (frame.n_num == n) &
                    (frame.granularity.str.startswith(gran))]
        assert len(sub) == 1, (db, n, gran, len(sub))
        return float(sub["frac_single_topology"].iloc[0])

    # panel (a) uses the same n = 25,000 ladder point on both databases, which is
    # the only comparison with an identical sample size
    hm_c, hm_f = pick("hMOF", 25000, "coarse", sz), pick("hMOF", 25000, "fine", sz)
    ar_c, ar_f = pick("ARC-MOF", 25000, "coarse", sz), pick("ARC-MOF", 25000, "fine", sz)
    ax.bar(x - w / 2, [hm_c * 100, ar_c * 100], S.widths(x, w), color=C["coarse"],
           edgecolor="white", linewidth=0.5, label="coarse key")
    ax.bar(x + w / 2, [hm_f * 100, ar_f * 100], S.widths(x, w), color=C["fine"],
           edgecolor="white", linewidth=0.5, label="fine key")
    for xi, v in zip(x - w / 2, [hm_c * 100, ar_c * 100]):
        S.value_label(ax, xi, v, f"{v:.1f}", dy_pt=1.0)
    for xi, v in zip(x + w / 2, [hm_f * 100, ar_f * 100]):
        S.value_label(ax, xi, v, f"{v:.1f}", dy_pt=1.0)
    S.zero_ref(ax, 100.0, ls=(0, (1.5, 1.5)))
    ax.set_xticks(x)
    ax.set_xticklabels(["hMOF", "ARC-MOF"])
    ax.set_ylim(0, 120)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("single-topology\nchemistry  [%]")
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor=C["light"], framealpha=0.95, fontsize=PT["annot"])
    S.gridlines_only_y(ax)
    S.panel_label(ax, "a", dx_pt=-24, dy_pt=3)
    ax.set_title("Both databases are confounded", loc="left",
                 fontsize=PT["axes_title"])

    # (b) the statistic against n, on evenly spaced ladders
    ax2 = fig.add_subplot(gs[0, 1])
    ns = [25000, 50000, 100000, 200000]
    for db, col, mk in [("hMOF", C["coarse"], "o"), ("ARC-MOF", C["fine"], "s")]:
        ys = [pick(db, n, "fine", sz) * 100 for n in ns]
        ax2.plot(range(len(ns)), ys, marker=mk, color=col, label=f"{db} (fine key)")
    ax2.set_xticks(range(len(ns)))
    ax2.set_xticklabels([f"{n//1000}k" for n in ns])
    ax2.set_xlim(-0.35, len(ns) - 0.35 + 1.05)
    ax2.set_ylim(92.0, 100.6)
    ax2.set_xlabel("sample size n\n(deliberate ladder, not a continuous axis)")
    ax2.set_ylabel("single-topology chemistry  [%]")
    ax2.legend(loc="lower left", frameon=True, facecolor="white",
               edgecolor=C["light"], framealpha=0.95, fontsize=PT["annot"])
    S.gridlines_only_y(ax2)
    S.panel_label(ax2, "b", dx_pt=-24, dy_pt=3)
    ax2.set_title("The number drifts with n", loc="left",
                  fontsize=PT["axes_title"])
    # the full-database value sits off the ladder and is drawn separately
    hm_full = float(sz[(sz.database == "hMOF") & sz.full_db &
                       (sz.granularity.str.startswith("fine"))]
                    ["frac_single_topology"].iloc[0]) * 100
    ax2.plot([len(ns) - 0.35 + 0.55], [hm_full], marker="*", ms=6,
             color=C["coarse"], clip_on=False)
    ax2.annotate("hMOF, full database\n(n = 324,426)",
                 xy=(len(ns) - 0.35 + 0.55, hm_full),
                 xytext=(2, -1), textcoords="offset points", ha="left",
                 va="top", fontsize=5.5, color=C["coarse"])

    # (c) per source sub-library
    ax3 = fig.add_subplot(gs[0, 2])
    s = src.sort_values("fine_frac_single").reset_index(drop=True)
    y3 = np.arange(len(s))
    h3 = 0.36
    ax3.barh(y3 + h3 / 2, s["coarse_frac_single"] * 100, S.widths(y3, h3),
             color=C["coarse"], edgecolor="white", linewidth=0.4,
             label="coarse key")
    ax3.barh(y3 - h3 / 2, s["fine_frac_single"] * 100, S.widths(y3, h3),
             color=C["fine"], edgecolor="white", linewidth=0.4,
             label="fine key")
    S.whiskers_h(ax3, s["fine_frac_single"] * 100, (y3 - h3 / 2),
                 s["boot_sd"] * 100, h=h3)
    ax3.axvline(100.0, color=C["ref"], lw=0.7, ls=(0, (1.5, 1.5)))
    ax3.set_yticks(y3)
    ax3.set_yticklabels([f"{r.source}\nn = {int(r.n_samples):,}" for r in s.itertuples()],
                        fontsize=5.5)
    ax3.set_xlim(0, 118)
    ax3.set_xlabel("single-topology chemistry  [%]")
    ax3.legend(loc="lower right", frameon=True, facecolor="white",
               edgecolor=C["light"], framealpha=0.95, fontsize=PT["annot"])
    ax3.grid(axis="x")
    ax3.grid(axis="y", visible=False)
    S.panel_label(ax3, "c", dx_pt=-52, dy_pt=3)
    ax3.set_title("The value tracks how each\nsub-library was built", loc="left",
                  fontsize=PT["axes_title"])

    fig.text(0.0, 0.02,
             "Panel (a) compares both databases at n = 25,000. ARC-MOF is not "
             "independent of hMOF; no ordering between them is claimed.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "fig8_cross_database",
                  exemptions=[{"panels": ["a", "b", "c"],
                               "checks": ["panel-width"],
                               "reason": "panel (a) has two categorical groups, "
                                         "(b) a numeric axis with four ladder "
                                         "points, (c) eight labelled sources; the "
                                         "content widths differ by design. The row "
                                         "check still verifies shared top/bottom "
                                         "plot-area edges and heights."}])


def main():
    only = sys.argv[1:] or ["5", "6", "7", "8"]
    fns = {"5": figure5, "6": figure6, "7": figure7, "8": figure8}
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(RASTER, exist_ok=True)
    for k in only:
        print(f"--- Figure {k} ---")
        fns[k]()
    S.report_printed(RECORDS)


if __name__ == "__main__":
    main()
