#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supplementary Figures S1-S7, rebuilt at final printed size.

Same contract as the main-text builders: figures are drawn at the 174 mm text
column, every number comes from the archived CSVs in ../outputs, and
`pubfig.style` enforces the glyph floor, the panel-alignment gate and the
rendered text-collision check before anything is written.
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
from scipy import stats

from pubfig import style as S

S.apply_style()
C, PT = S.C, S.PT

FAM_LABEL = {
    "structure-derived descriptors": "structure-derived\ndescriptors",
    "functional_groups": "functional\ngroups",
    "metal_linker": "metal\nlinker",
    "topology": "topology",
}
RECORDS = {}


def finish(fig, stem, **kw):
    rec = S.save_figure(fig, stem, DEST, raster_dir=RASTER, **kw)
    RECORDS[stem] = rec
    plt.close(fig)
    return rec


# ==========================================================================
# Figure S1 - SHAP attribution by feature family
# ==========================================================================
def figS1():
    d = pd.read_csv(os.path.join(OUTDIR, "fig5_shap_family_final.csv"))
    rand = d[d.scheme == "random"].set_index("family")
    topo = d[d.scheme == "topology"].set_index("family")
    order = list(rand.index)

    fig = plt.figure(figsize=S.fig_size(66))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.105, right=0.985,
                          top=0.775, bottom=0.245)

    for panel, (left, right, ylab, title) in enumerate([
        (rand["mean_abs_shap"], topo["mean_abs_shap"],
         "mean |SHAP| per fold", "Absolute attribution"),
        (rand["fraction"] * 100, topo["fraction"] * 100,
         "share of total attribution  [%]", "Relative attribution"),
    ]):
        ax = fig.add_subplot(gs[0, panel])
        x = np.arange(len(order))
        w = 0.36
        ax.bar(x - w / 2, [left[f] for f in order], S.widths(x, w), color=C["random"],
               edgecolor="white", linewidth=0.5, label="random split")
        ax.bar(x + w / 2, [right[f] for f in order], S.widths(x, w), color=C["topology"],
               edgecolor="white", linewidth=0.5,
               label="leave-one-topology-out")
        for xi, f in zip(x - w / 2, order):
            S.value_label(ax, xi, left[f],
                          f"{left[f]:.3f}" if panel == 0 else f"{left[f]:.0f}",
                          dy_pt=1.0, fontsize=5.5)
        for xi, f in zip(x + w / 2, order):
            S.value_label(ax, xi, right[f],
                          f"{right[f]:.3f}" if panel == 0 else f"{right[f]:.0f}",
                          dy_pt=1.0, fontsize=5.5)
        ax.set_xticks(x)
        ax.set_xticklabels([FAM_LABEL.get(f, f) for f in order], fontsize=5.5)
        ax.set_ylabel(ylab)
        if panel == 1:
            ax.set_ylim(0, 108)
        else:
            ax.set_ylim(0, 0.80)
        ax.legend(loc="upper right", frameon=True, facecolor="white",
                  edgecolor=C["light"], framealpha=0.95, fontsize=PT["annot"])
        S.gridlines_only_y(ax)
        S.panel_label(ax, "ab"[panel], dx_pt=-40, dy_pt=3)
        ax.set_title(title, loc="left", fontsize=PT["axes_title"])

    fig.text(0.0, 0.02,
             "Identity families (functional groups, metal linker, topology) move "
             "from 17.7% to 14.9%: the model does not shift onto identity.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "figS1_shap_family")


# ==========================================================================
# Figure S2 - how each geometric descriptor acts on the prediction
# ==========================================================================
def figS2():
    """Centred partial-dependence curves for the top descriptors.

    The submitted Figure S2 was a per-sample SHAP dependence plot.  Its source
    values were never archived, and re-deriving exact per-sample TreeSHAP is not
    practical here (see scripts/prepare_figS2_pdp_data.py for the measurements),
    so this panel shows centred partial dependence instead: the same question
    -- what shape does the model learn for each descriptor -- by a route that is
    cheap and exactly reproducible.  The y axis says so explicitly.
    """
    cpath = os.path.join(OUTDIR, "pdp_geometry_curves.csv")
    mpath = os.path.join(OUTDIR, "pdp_geometry_meta.csv")
    if not os.path.exists(cpath):
        raise SystemExit(
            "Figure S2 needs outputs/pdp_geometry_curves.csv. Run "
            "scripts/prepare_figS2_pdp_data.py first.")
    d = pd.read_csv(cpath)
    meta = (pd.read_csv(mpath).set_index("key")["value"].to_dict()
            if os.path.exists(mpath) else {})

    rank = (d.groupby("descriptor")["pdp_centred"]
            .apply(lambda s: float(s.max() - s.min()))
            .sort_values(ascending=False))
    picked = list(rank.index[:4])

    fig = plt.figure(figsize=S.fig_size(72))
    gs = fig.add_gridspec(1, len(picked), wspace=0.34, left=0.092, right=0.988,
                          top=0.775, bottom=0.305)
    for j, col in enumerate(picked):
        ax = fig.add_subplot(gs[0, j])
        g = d[d.descriptor == col].sort_values("value")
        ax.axhline(0, color=C["ref"], lw=0.7)
        ax.fill_between(g.value, g.pdp_centred - 1.96 * g.pdp_se,
                        g.pdp_centred + 1.96 * g.pdp_se,
                        color=C["topology_light"], alpha=0.55, linewidth=0,
                        label="95% CI of the mean")
        ax.plot(g.value, g.pdp_centred, color=C["topology"], lw=1.2,
                marker="o", ms=1.8, markeredgewidth=0)
        # the raw column names are far too long for a four-up panel; the unit is
        # kept on its own second line and the descriptor uses the short name the
        # manuscript already uses elsewhere
        short = {
            "volume [A^3]": ("volume", "[\u00c5\u00b3]"),
            "weight [u]": ("weight", "[u]"),
            "surface_area [m^2/g]": ("surface area", "[m\u00b2/g]"),
            "void_fraction": ("void fraction", "[-]"),
            "void_volume [cm^3/g]": ("void volume", "[cm\u00b3/g]"),
            "largest_free_sphere_diameter [A]": ("largest free", "sphere d [\u00c5]"),
            "largest_included_sphere_along_free_sphere_path_diameter [A]":
                ("largest incl.", "sphere path d [\u00c5]"),
            "largest_included_sphere_diameter [A]":
                ("largest incl.", "sphere d [\u00c5]"),
        }[col]
        ax.set_xlabel(f"{short[0]}\n{short[1]}")
        if j == 0:
            ax.set_ylabel(f"centred partial dependence  [mmol/g]")
        else:
            ax.set_yticklabels([])
        S.gridlines_only_y(ax)
        ax.grid(axis="x", visible=True)
        S.panel_label(ax, "abcd"[j], dx_pt=-46, dy_pt=3)
        ax.set_title(f"range = {rank[col]:.2f}", loc="left",
                     fontsize=PT["axes_title"])
        if j == 0:
            ax.legend(loc="lower left", frameon=True, facecolor="white",
                      edgecolor=C["light"], framealpha=0.95, fontsize=5.5)

    fig.text(0.0, 0.02,
             f"Centred partial dependence, random-split model, "
             f"{int(float(meta.get('n_rows_modelled', 0))):,} rows. This replaces "
             f"the per-sample SHAP view; the estimand differs.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "figS2_shap_dependence",
                  exemptions=[{"panels": ["a", "b", "c", "d"],
                               "checks": ["panel-width"],
                               "reason": "four small multiples share one grid and "
                                         "differ only in the descriptor on the x "
                                         "axis; equal widths, and the row check "
                                         "still verifies shared edges."}])


# ==========================================================================
# Figure S3 - within-topology against cross-topology R2
# ==========================================================================
def figS3():
    d = pd.read_csv(os.path.join(OUTDIR, "within_vs_cross_final.csv"))
    d = d.sort_values("n", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    fig = plt.figure(figsize=S.fig_size(70))
    gs = fig.add_gridspec(1, 1, left=0.175, right=0.978, top=0.80,
                          bottom=0.215)
    ax = fig.add_subplot(gs[0, 0])
    for yi, r in enumerate(d.itertuples()):
        ax.plot([r.R2_cross, r.R2_within], [yi, yi], color=C["light"], lw=1.4,
                zorder=1, solid_capstyle="round")
    ax.scatter(d.R2_within, y, s=24, color=C["topology"], edgecolors="white",
               linewidths=0.5, zorder=3, label="trained within the same topology")
    ax.scatter(d.R2_cross, y, s=24, color=C["random"], edgecolors="white",
               linewidths=0.5, zorder=3, label="trained on other topologies")
    ax.axvline(0, color=C["ref"], lw=0.7, ls=(0, (1.5, 1.5)))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.topology}  (n = {r.n:,})" for r in d.itertuples()],
                       fontsize=5.5)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlim(-4.9, 1.0)
    ax.set_xlabel(f"{S.R2} with the same eight geometric descriptors")
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor=C["light"], framealpha=0.95, fontsize=PT["annot"])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    S.panel_label(ax, "a", dx_pt=-64, dy_pt=3)
    ax.set_title("The geometry \u2192 uptake map is topology-specific",
                 loc="left", fontsize=PT["axes_title"])
    etb_y = int(d.index[d.topology == "etb"][0])
    ax.annotate(f"etb: {d.loc[d.topology == 'etb', 'R2_cross'].iloc[0]:.2f}",
                xy=(d.loc[d.topology == "etb", "R2_cross"].iloc[0], etb_y),
                xytext=(2, -12), textcoords="offset points", ha="left",
                va="top", fontsize=5.5, color=C["random"],
                bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none"))
    fig.text(0.0, 0.02,
             f"Median within {d.R2_within.median():.3f} against median cross "
             f"{d.R2_cross.median():.3f}. Cross-topology R\u00b2 is negative for "
             f"{int((d.R2_cross < 0).sum())} of {len(d)} topologies.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "figS3_within_vs_cross")


# ==========================================================================
# Figure S4 - multi-seed stability of the aggregation rule
# ==========================================================================
def figS4():
    d = pd.read_csv(os.path.join(OUTDIR, "multi_seed_uptake25k_fix.csv"))
    fig = plt.figure(figsize=S.fig_size(74))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.088, right=0.985,
                          top=0.775, bottom=0.245)

    # (a) three rules across five seeds
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(d))
    w = 0.26
    rules = [("loto_pooled", "pooled (all folds merged)", C["topology"]),
             ("loto_median", "median over folds", C["teal"]),
             ("loto_macro", "macro (mean over folds)", C["random"])]
    for i, (col, lab, c) in enumerate(rules):
        ax.bar(x + (i - 1) * w, d[col], S.widths(d[col], w), color=c, edgecolor="white",
               linewidth=0.4, label=lab)
    S.zero_ref(ax, 0.0, ls=(0, (1.5, 1.5)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed\n{int(s)}" for s in d.seed])
    ax.set_ylim(-1.75, 0.62)
    ax.set_ylabel(f"leave-one-topology-out {S.R2}")
    ax.legend(loc="lower left", frameon=True, facecolor="white",
              edgecolor=C["light"], framealpha=0.95, fontsize=5.5)
    S.gridlines_only_y(ax)
    S.panel_label(ax, "a", dx_pt=-24, dy_pt=3)
    ax.set_title("Same run, three aggregation rules", loc="left",
                 fontsize=PT["axes_title"])

    # (b) across-seed dispersion of each rule
    ax2 = fig.add_subplot(gs[0, 1])
    spread = []
    for col, lab, c in rules:
        v = d[col].to_numpy(float)
        spread.append((lab, 100 * v.std(ddof=1) / abs(v.mean()), c))
    yb = np.arange(len(spread))
    ax2.barh(yb, [s[1] for s in spread], S.widths(yb, 0.5),
             color=[s[2] for s in spread], edgecolor="white", linewidth=0.5)
    for yi, s in zip(yb, spread):
        ax2.annotate(f"{s[1]:.0f}%", xy=(s[1], yi), xytext=(3, 0),
                     textcoords="offset points", va="center", ha="left",
                     fontsize=PT["annot"], fontweight="bold")
    ax2.set_yticks(yb)
    ax2.set_yticklabels(["pooled", "median\nover folds", "macro"],
                        fontsize=5.5)
    ax2.set_xlim(0, 118)
    ax2.set_xlabel("across-seed dispersion  (sd / |mean|, %)")
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    S.panel_label(ax2, "b", dx_pt=-30, dy_pt=3)
    ax2.set_title("'Stable' pooled R\u00b2 is an artefact", loc="left",
                  fontsize=PT["axes_title"])

    fig.text(0.0, 0.02,
             "Five seeds, 25,000 samples each. The pooled value looks stable only "
             "because one topology dominates the pooled test set.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "figS4_aggregation_stability")


# ==========================================================================
# Figure S5 - four aggregation rules and the folds behind them
# ==========================================================================
def figS5():
    ma = pd.read_csv(os.path.join(OUTDIR, "metrics_audit.csv"))
    row = ma[ma.result.str.strip() == "LOTO / topology (100k)"].iloc[0]
    lp = pd.read_csv(os.path.join(OUTDIR, "loto_per_topology_final.csv"))

    fig = plt.figure(figsize=S.fig_size(80))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.30,
                          left=0.085, right=0.985, top=0.79, bottom=0.255)

    # (a) the four aggregates of one run
    ax = fig.add_subplot(gs[0, 0])
    names = ["pooled", "weighted", "macro", "median"]
    # bracket access: `row.median` resolves to the pandas method, not the value
    vals = [float(row['pooled_as_reported']), float(row['weighted']),
            float(row['macro']), float(row['median'])]
    cols = [C["topology"], C["chemistry"], C["random"], C["identity"]]
    x = np.arange(4)
    ax.bar(x, vals, S.widths(vals, 0.58), color=cols, edgecolor="white", linewidth=0.5)
    for xi, v in zip(x, vals):
        S.value_label(ax, xi, v, f"{v:+.3f}",
                      dy_pt=1.5 if v >= 0 else -8.5, fontsize=5.5)
    S.zero_ref(ax, 0.0, ls=(0, (1.5, 1.5)))
    ax.set_xticks(x)
    ax.set_xticklabels(["pooled", "weighted\n(by test size)", "macro",
                        "median"], fontsize=5.5)
    ax.set_ylim(-0.72, 0.50)
    ax.set_ylabel(f"leave-one-topology-out {S.R2}")
    S.gridlines_only_y(ax)
    S.panel_label(ax, "a", dx_pt=-24, dy_pt=3)
    ax.set_title("One run, four aggregates", loc="left",
                 fontsize=PT["axes_title"])

    # (b) the eleven folds behind them, with each aggregate marked
    ax2 = fig.add_subplot(gs[0, 1])
    lp = lp.sort_values("R2").reset_index(drop=True)
    y = np.arange(len(lp))
    cols2 = [C["random"] if v < 0 else C["topology"] for v in lp.R2]
    ax2.barh(y, lp.R2, S.widths(y, 0.62), color=cols2, edgecolor="white", linewidth=0.4)
    for yi, r in zip(y, lp.itertuples()):
        off = 3 if r.R2 >= 0 else -3
        ax2.annotate(f"{r.R2:+.2f}", xy=(r.R2, yi), xytext=(off, 0),
                     textcoords="offset points", ha="left" if r.R2 >= 0 else "right",
                     va="center", fontsize=5.0, color=C["ink"])
    for val, lab, c, dy in [(vals[0], "pooled", C["topology"], 0.45),
                            (vals[2], "macro", C["random"], -0.45),
                            (vals[3], "median", C["identity"], 0.45)]:
        ax2.axvline(val, color=c, lw=0.9, ls=(0, (3, 2)), zorder=4)
        # anchor inside the axes: a centred label on the macro line ran past the
        # right edge of the figure
        ax2.annotate(f"{lab} {val:+.3f}", xy=(val, len(lp) - 0.55 + dy),
                     xytext=(0, 0), textcoords="offset points",
                     ha="right" if val < 0 else "left", va="center",
                     fontsize=5.5, color=c, zorder=5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{r.group}  (n = {int(r.n_test):,})"
                         for r in lp.itertuples()], fontsize=5.0)
    ax2.set_ylim(-0.6, len(lp) - 0.4)
    ax2.set_xlim(-3.5, 1.05)
    ax2.set_xlabel(f"per-topology held-out {S.R2}")
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    S.panel_label(ax2, "b", dx_pt=-72, dy_pt=3)
    ax2.set_title("The eleven folds behind those numbers", loc="left",
                  fontsize=PT["axes_title"])

    fig.text(0.0, 0.02,
             f"Pooling is dominated by pcu ({lp.loc[lp.group == 'pcu', 'n_test'].iloc[0]:,} "
             f"of {int(lp.n_test.sum()):,} test rows).",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "figS5_aggregation_rules",
                  exemptions=[{"panels": ["a", "b"], "checks": ["panel-width"],
                               "reason": "panel (a) holds four bars, panel (b) "
                                         "eleven labelled folds that need the "
                                         "wider column. The row check still "
                                         "verifies shared top/bottom edges."}])


# ==========================================================================
# Figure S6 - per-source confounding in ARC-MOF
# ==========================================================================
def figS6():
    src = pd.read_csv(os.path.join(OUTDIR, "source_confound_granularity.csv"))
    nd = pd.read_csv(os.path.join(OUTDIR, "null_topology_cardinality.csv"))
    d = src.merge(nd[["source", "observed", "null_mean", "b"]], on="source",
                  how="left", validate="one_to_one")
    d["excess_pp"] = (d["observed"] - d["null_mean"]) * 100.0

    fig = plt.figure(figsize=S.fig_size(76))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28,
                          left=0.082, right=0.985, top=0.775, bottom=0.265)

    # (a) both real granularities per source
    ax = fig.add_subplot(gs[0, 0])
    s = d.sort_values("fine_frac_single").reset_index(drop=True)
    x = np.arange(len(s))
    w = 0.36
    ax.bar(x - w / 2, s["coarse_frac_single"] * 100, S.widths(x, w), color=C["coarse"],
           edgecolor="white", linewidth=0.4, label="coarse key")
    ax.bar(x + w / 2, s["fine_frac_single"] * 100, S.widths(x, w), color=C["fine"],
           edgecolor="white", linewidth=0.4, label="fine key")
    S.whiskers_v(ax, (x + w / 2), s["fine_frac_single"] * 100,
                 s["boot_sd"] * 100, w=w)
    ax.axhline(100.0, color=C["ref"], lw=0.7, ls=(0, (1.5, 1.5)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.source}\nn = {int(r.n_samples):,}"
                        for r in s.itertuples()], fontsize=5.0)
    ax.set_ylim(0.78, 1.075)
    ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"])
    ax.set_ylabel("compositions in a single topology\n(fraction)")
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor=C["light"], framealpha=0.95, fontsize=5.5)
    S.gridlines_only_y(ax)
    S.panel_label(ax, "a", dx_pt=-46, dy_pt=3)
    ax.set_title("Confounding differs by source", loc="left",
                 fontsize=PT["axes_title"])
    for i, r in enumerate(s.itertuples()):
        if r.n_topologies <= 1:
            ax.annotate("1 topology\n(degenerate)", xy=(i + w / 2, 1.0),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.0,
                        color=C["degen"])

    # (b) excess over the source's own permutation null
    ax2 = fig.add_subplot(gs[0, 1])
    sub = d[d.n_topologies > 1].copy()
    xv = sub["H_T"].to_numpy(float)
    yv = sub["excess_pp"].to_numpy(float)
    r, p = stats.pearsonr(xv, yv)
    rho, sp = stats.spearmanr(xv, yv)
    ax2.scatter(xv, yv, s=26, color=C["topology"], edgecolors="white",
                linewidths=0.6, zorder=3)
    k, b0 = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 20)
    ax2.plot(xs, k * xs + b0, color=C["mid"], lw=0.9, ls=(0, (3, 2)), zorder=2)
    ax2.axhline(0.0, color=C["ref"], lw=0.7, ls=(0, (1.5, 1.5)))
    S.annotate_no_overlap(ax2, list(zip(xv, yv)),
                          list(sub["source"].astype(str)), fontsize=5.5,
                          mask=True)
    ax2.set_xlim(0.0, 6.6)
    ax2.set_ylim(-3.0, 46.0)
    ax2.set_xlabel("topology entropy H(T) of the source  [bit]")
    ax2.set_ylabel("excess over its own\npermutation null  [pp]")
    S.gridlines_only_y(ax2)
    ax2.grid(axis="x", visible=True)
    S.panel_label(ax2, "b", dx_pt=-26, dy_pt=3)
    ax2.set_title("Topology richness does not\nsettle the spread", loc="left",
                  fontsize=PT["axes_title"])
    S.stats_box(ax2, [f"Pearson r = {r:+.2f}  (p = {p:.2f})",
                      f"Spearman \u03c1 = {rho:+.2f}  (p = {sp:.2f})",
                      f"n = {len(sub)} sources (DB6 dropped: 1 topology)"],
                loc="upper left", fontsize=5.5)

    fig.text(0.0, 0.02,
             "Panel (b) drops DB6, whose single topology makes the statistic "
             "degenerate. DB12 alone carries the fitted slope; the relation is "
             "suggestive, not established.",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "figS6_source_confound",
                  exemptions=[{"panels": ["a", "b"], "checks": ["panel-width"],
                               "reason": "panel (a) carries eight labelled sources, "
                                         "panel (b) is a seven-point scatter. The "
                                         "row check still verifies shared "
                                         "top/bottom edges."}])


# ==========================================================================
# Figure S7 - does the unseen-chemistry rate order the hold-out cost?
# ==========================================================================
def figS7():
    d = pd.read_csv(os.path.join(OUTDIR, "s_g_curve.csv"))
    keep = d[d.n_test >= 100].copy()
    fig = plt.figure(figsize=S.fig_size(76))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28,
                          left=0.082, right=0.985, top=0.775, bottom=0.265)

    # (a) unseen-chemistry rate against held-out R2
    ax = fig.add_subplot(gs[0, 0])
    neg = keep[keep.R2_pooled < 0]
    pos = keep[keep.R2_pooled >= 0]
    ax.axhline(0.0, color=C["ref"], lw=0.7, ls=(0, (1.5, 1.5)))
    ax.scatter(pos.S_miss_unseen, pos.R2_pooled, s=26, color=C["topology"],
               edgecolors="white", linewidths=0.6, zorder=3,
               label=f"{S.R2} \u2265 0")
    ax.scatter(neg.S_miss_unseen, neg.R2_pooled, s=26, color=C["random"],
               edgecolors="white", linewidths=0.6, zorder=3,
               label=f"{S.R2} < 0")
    r, p = stats.pearsonr(keep.S_miss_unseen, keep.R2_pooled)
    rho, sp = stats.spearmanr(keep.S_miss_unseen, keep.R2_pooled)
    ax.set_xlim(-0.06, 1.12)
    ax.set_ylim(-3.4, 1.15)
    ax.set_xlabel(f"{S.S_MISS} \u2014 fraction of the held-out topology's\n"
                  f"structures with no chemical counterpart in training")
    ax.set_ylabel(f"per-topology held-out {S.R2}")
    ax.legend(loc="lower right", frameon=True, facecolor="white",
              edgecolor=C["light"], framealpha=0.95, fontsize=5.5)
    S.gridlines_only_y(ax)
    ax.grid(axis="x", visible=True)
    S.panel_label(ax, "a", dx_pt=-26, dy_pt=3)
    ax.set_title("The unseen-chemistry rate does not order the cost",
                 loc="left", fontsize=PT["axes_title"])
    S.stats_box(ax, [f"Pearson r = {r:+.2f}  (p = {p:.2f})",
                     f"Spearman \u03c1 = {rho:+.2f}  (p = {sp:.2f})",
                     f"n = {len(keep)} topologies, n_test \u2265 100"],
                loc="upper left", fontsize=5.5)

    # (b) why: the pooled R2 normalises by each fold's own variance
    ax2 = fig.add_subplot(gs[0, 1])
    dd = d.sort_values("sd_y_test").reset_index(drop=True)
    y = np.arange(len(dd))
    cols = [C["random"] if t == "pcu" else C["topology"] for t in dd.held_out_topology]
    ax2.barh(y, dd.sd_y_test, S.widths(y, 0.62), color=cols, edgecolor="white", linewidth=0.4)
    for yi, r_ in zip(y, dd.itertuples()):
        ax2.annotate(f"{r_.R2_pooled:+.2f}", xy=(r_.sd_y_test, yi),
                     xytext=(3, 0), textcoords="offset points",
                     va="center", ha="left", fontsize=5.0, color=C["ink"])
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{r_.held_out_topology}\n(n = {int(r_.n_test):,})"
                         for r_ in dd.itertuples()], fontsize=5.0)
    ax2.set_ylim(-0.6, len(dd) - 0.4)
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("sd of the target within the held-out topology  [mmol/g]")
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)
    S.panel_label(ax2, "b", dx_pt=-46, dy_pt=3)
    ax2.set_title("Why: each fold's own variance", loc="left",
                  fontsize=PT["axes_title"])
    pcu_i = int(dd.index[dd.held_out_topology == "pcu"][0])
    ax2.annotate("pcu: 63% of the samples and\nthe largest variance, so it sets\n"
                 "the pooled value",
                 xy=(dd.sd_y_test.iloc[pcu_i], pcu_i), xytext=(-8, 14),
                 textcoords="offset points", ha="right", va="bottom",
                 fontsize=5.0, color=C["random"],
                 arrowprops=dict(arrowstyle="-|>", color=C["random"], lw=0.7))

    fig.text(0.0, 0.02,
             "Eleven topologies, one run. Numbers beside the bars in (b) are the "
             "same held-out R\u00b2 as in (a).",
             fontsize=PT["foot"], color=C["mid"], ha="left", va="bottom")
    return finish(fig, "figS7_sg_curve",
                  exemptions=[{"panels": ["a", "b"], "checks": ["panel-width"],
                               "reason": "panel (a) is a labelled scatter with a long "
                                         "two-line x label, panel (b) a ranked bar "
                                         "chart. The row check still verifies "
                                         "shared top/bottom edges."}])


def main():
    only = sys.argv[1:] or ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    fns = {"S1": figS1, "S2": figS2, "S3": figS3, "S4": figS4,
           "S5": figS5, "S6": figS6, "S7": figS7}
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(RASTER, exist_ok=True)
    for k in only:
        print(f"--- Figure {k} ---")
        fns[k]()
    S.report_printed(RECORDS)


if __name__ == "__main__":
    main()
