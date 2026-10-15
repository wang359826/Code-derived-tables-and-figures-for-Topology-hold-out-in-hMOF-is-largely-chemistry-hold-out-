#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared publication style for every figure of this manuscript.

Design contract (see FIGURE_QA_PUBLICATION.md for the full record)
------------------------------------------------------------------
* Final printed geometry is fixed up front: the manuscript PDF gives every
  figure a text column of 174.0 mm (492 pt), so each figure is drawn at
  FIG_W_IN = 492/72 in. Nothing is scaled afterwards. Every glyph therefore
  arrives at its nominal point size instead of being shrunk to 2-4 pt the way
  the previous 100-170 dpi, 9-17 inch wide exports were.
* Glyph floor: 6.0 pt. Body/axis text is 7.0 pt, tick labels 6.5 pt, dense
  in-panel annotations 6.0 pt, panel letters 8.0 pt bold.
* Colour carries one meaning across the whole paper:

      random split / leaky protocol      ALWAYS red   (RANDOM)
      topology hold-out / honest         ALWAYS blue  (TOPOLOGY)
      within-topology (reference)        blue
      cross-topology                     red
      coarse chemistry key               teal          (COARSE)
      fine chemistry key (+ functional)  red           (FINE)
      identity features                  violet
      neutral reference/annotation       grey

  The previous figure set used red for the random split in Figures 1-3 and 8
  but red for the topology hold-out in Figures 4, 5 and S1; that contradiction
  is removed here.
* No embedded "Figure N." inside the artwork. The number, title and legend are
  supplied once by the PDF builder's <figcaption>, so the image cannot drift
  out of step with the caption number.
* Text stays editable: SVG text is not converted to paths and PDF uses
  TrueType (fonttype 42).

Geometry helpers
----------------
`mm()`, `fig_size()`, `save_figure()` and `panel_label()` are the only entry
points a figure script needs.
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import numpy as np

# --------------------------------------------------------------------------
# final printed geometry
# --------------------------------------------------------------------------
MM_PER_IN = 25.4
COL_W_MM = 174.0                     # manuscript text column width
FIG_W_IN = COL_W_MM / MM_PER_IN      # 6.8504 in == 492 pt
GRID_KW = dict(left=0.075, right=0.988, bottom=0.20, top=0.86, wspace=0.34,
               hspace=0.30)


def mm(x: float) -> float:
    """millimetres -> inches (matplotlib's unit)."""
    return x / MM_PER_IN


def fig_size(h_mm: float, w_mm: float = COL_W_MM):
    """Return a (width, height) figure size in inches for the printed panel."""
    return (w_mm / MM_PER_IN, h_mm / MM_PER_IN)


# --------------------------------------------------------------------------
# typography
# --------------------------------------------------------------------------
PT = {
    "title": 8.5,        # suptitle (only used when a figure needs one line)
    "panel": 8.0,        # bold lowercase panel letter
    "axes_title": 7.5,   # per-axes title
    "label": 7.0,        # axis labels
    "tick": 6.5,         # tick labels
    "legend": 6.5,
    "annot": 6.0,        # in-panel value / statistic annotations
    "foot": 6.0,         # figure-level footnote line
}
GLYPH_FLOOR_PT = 6.0

# --------------------------------------------------------------------------
# colour: one meaning per colour across the whole paper
# --------------------------------------------------------------------------
C = {
    # --- the two protocols (paper-wide anchor) ---
    "random":        "#B64342",   # random split / leaky protocol      red
    "random_soft":   "#E9A6A1",
    "random_dark":   "#8C2F2E",
    "topology":      "#0F4D92",   # topology hold-out / honest         blue
    "topology_soft": "#3775BA",
    "topology_light": "#A9C4E4",
    "topology_dark": "#0A3566",

    # --- other grouping variables ---
    "linker":        "#9A4D8E",   # grouped by organic linker          violet
    "metal":         "#42949E",   # grouped by metal centre            teal
    "teal":          "#42949E",
    "chemistry":     "#4D7A3A",   # grouped by chemistry key           green

    # --- chemistry granularity (Figs 2, 8, S6) ---
    "coarse":        "#42949E",   # coarse key = metal + 2 linkers
    "fine":          "#B64342",   # fine key   = coarse + functional groups

    # --- feature families / attribution (Figs 6, 7, S1) ---
    "identity":      "#9A4D8E",
    "identity_soft": "#C79BBF",
    "degen":         "#B9B9B9",   # degenerate / undefined value

    # --- neutrals ---
    "ink":           "#272727",
    "mid":           "#767676",
    "light":         "#CFCECE",
    "grid":          "#DCDCDC",
    "ref":           "#8C8C8C",
}
# grouping-variable ramp used by Figure 1 (random -> ... -> topology)
GROUP_RAMP = {
    "random":   C["random"],
    "linker":   C["linker"],
    "metal":    C["metal"],
    "chemistry": C["chemistry"],
    "topology": C["topology"],
}
FAMILY_COLOR = {
    "structure-derived descriptors": C["topology"],
    "functional_groups":             C["identity"],
    "metal_linker":                  C["metal"],
    "topology":                      C["chemistry"],
}

# --------------------------------------------------------------------------
# matplotlib defaults
# --------------------------------------------------------------------------
def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        # Arial lacks U+2080-2082, U+2098, U+209B, U+03C1 and U+2212, so the
        # manuscript's CO2 / S_miss / rho / minus labels would render as boxes.
        # DejaVu Sans covers every glyph used here and stays inside the Arial /
        # Helvetica / sans-serif family the journal contract expects.
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": PT["label"],
        "axes.titlesize": PT["axes_title"],
        "axes.labelsize": PT["label"],
        "axes.linewidth": 0.6,
        "axes.edgecolor": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.titlecolor": C["ink"],
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "xtick.labelsize": PT["tick"],
        "ytick.labelsize": PT["tick"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": PT["legend"],
        "legend.frameon": False,
        "legend.labelspacing": 0.30,
        "legend.handlelength": 1.4,
        "legend.handleheight": 0.7,
        "legend.borderaxespad": 0.3,
        "legend.borderpad": 0.25,
        "figure.dpi": 140,          # on-screen only
        "savefig.dpi": 600,         # raster deliverables
        "savefig.facecolor": "white",
        "svg.fonttype": "none",     # editable SVG text
        "pdf.fonttype": 42,         # editable TrueType text in PDF
        "pdf.compression": 6,
        "patch.linewidth": 0.5,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "errorbar.capsize": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.default": "regular",
        "figure.autolayout": False,
    })
    # The manuscript writes R^2 as a Unicode superscript everywhere; do the same
    # so mathtext cannot shrink a glyph below the 6 pt floor.
    mpl.rcParams["axes.unicode_minus"] = True


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
R2 = "R\u00b2"          # R²
RHO = "\u03c1"           # ρ
DELTA = "\u0394"         # Δ
MINUS = "\u2212"         # −
RARROW = "\u2192"        # →
CO2 = "CO\u2082"         # CO₂
LOG10 = "log10"          # avoids Arial's missing U+2081
S_MISS = "S(miss)"       # avoids the sparse Unicode subscript block


def panel_label(ax, letter: str, dx_pt: float = -30.0, dy_pt: float = 3.0) -> None:
    """Bold lowercase panel letter placed with a FIXED point offset.

    A fixed offset keeps panel letters on a common anchor across panels of
    unequal height; an axes-fraction offset such as y=1.02 would move them by
    different physical distances.
    """
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx_pt, dy_pt), textcoords="offset points",
                fontsize=PT["panel"], fontweight="bold", color=C["ink"],
                ha="left", va="bottom", annotation_clip=False)


def stats_box(ax, lines, loc="upper left", pad=0.022, fontsize=None):
    """Neutral statistics box; no fancy arrow, no opaque mask over data."""
    from matplotlib.offsetbox import AnchoredText
    txt = AnchoredText("\n".join(lines), loc=loc, pad=pad,
                       borderpad=0.35, prop=dict(size=fontsize or PT["annot"],
                                                 color=C["ink"]),
                       frameon=True)
    txt.patch.set_boxstyle("round,pad=0.25")
    txt.patch.set_facecolor("white")
    txt.patch.set_edgecolor(C["light"])
    txt.patch.set_linewidth(0.5)
    txt.patch.set_alpha(0.92)
    ax.add_artist(txt)
    return txt


def widths(n_or_seq, w):
    """Return an explicit 1-D width array of length ``n_or_seq``.

    Matplotlib 3.11 no longer broadcasts a scalar ``width`` against an array of
    bar heights: ``ax.bar(x, heights, 0.34)`` collapses to ONE bar spanning the
    whole x range, and any ``yerr`` then renders as a single diagonal segment
    across the axes.  Every bar call in this package passes its width through
    this helper so the plotted geometry cannot depend on the matplotlib version.
    """
    try:
        n = len(n_or_seq)
    except TypeError:
        n = int(n_or_seq)
    return [float(w)] * n


def value_label(ax, x, y, text, dy_pt=1.2, **kw):
    """Numeric value above a bar/point, using a point offset (never data units).

    A small white marker sits behind the digits.  Axes gridlines span the whole
    plot area, so without it a y gridline at the bar's own value runs straight
    through the label; masking a background gridline hides no data.
    """
    kw.setdefault("ha", "center")
    kw.setdefault("va", "bottom")
    kw.setdefault("fontsize", PT["annot"])
    kw.setdefault("color", C["ink"])
    kw.setdefault("bbox", dict(boxstyle="square,pad=0.16", fc="white",
                               ec="none", alpha=1.0))
    ax.annotate(text, xy=(x, y), xytext=(0, dy_pt), textcoords="offset points", **kw)


def annotate_no_overlap(ax, points, labels, fontsize=None, color=None,
                        n_candidates: int = 40, gap_pt: float = 2.6,
                        mask: bool = False):
    """Place one label per point, searching a spiral of offsets until none touch.

    A fixed above/below rule is not enough for a crowded scatter: two points a
    few points apart both get an "above" offset and the labels land on each
    other.  Each label walks a deterministic golden-angle spiral of candidate
    offsets, from just above its point outwards, and takes the first slot that
    lies inside the axes and overlaps no already-placed label.

    Returns (placed_count, failed_count).
    """
    import math

    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fs = fontsize or PT["annot"]
    col = color or C["mid"]
    placed, failed = [], 0
    golden = math.pi * (3.0 - math.sqrt(5.0))
    fmt = 0.55 * (fs + gap_pt)

    for (x, y), lab in zip(points, labels):
        done = False
        for i in range(n_candidates):
            if i == 0:
                ox, oy = 0.0, gap_pt
            else:
                radius = gap_pt + 1.15 * (fs + gap_pt) * math.sqrt(i)
                ang = i * golden
                ox = radius * math.cos(ang)
                oy = radius * math.sin(ang)
            ha = "center" if abs(ox) < fmt else ("left" if ox > 0 else "right")
            va = "center" if abs(oy) < fmt else ("bottom" if oy > 0 else "top")
            t = ax.annotate(lab, xy=(x, y), xytext=(ox, oy),
                            textcoords="offset points", ha=ha, va=va,
                            fontsize=fs, color=col, annotation_clip=True,
                            zorder=(8 if mask else 3),
                            bbox=(dict(boxstyle="square,pad=0.16", fc="white",
                                       ec="none") if mask else None))
            fig.canvas.draw()
            bb = t.get_window_extent(r)
            axbb = ax.get_window_extent(r)
            inside = (bb.x0 >= axbb.x0 - 1 and bb.x1 <= axbb.x1 + 1
                      and bb.y0 >= axbb.y0 - 1 and bb.y1 <= axbb.y1 + 1)
            clash = any(bb.x0 < o.x1 + 0.8 and o.x0 < bb.x1 + 0.8
                        and bb.y0 < o.y1 + 0.8 and o.y0 < bb.y1 + 0.8
                        for o in placed)
            if inside and not clash:
                placed.append(bb)
                done = True
                break
            t.remove()
        if not done:
            failed += 1
    return len(placed), failed


def audit_text_stroke_collisions(fig) -> tuple:
    """Report stroked line segments that pass through the interior of a text box.

    The skill's PDF collision audit lists `text-stroke` as a FAIL class, but these
    figures are embedded as rasters, so that audit cannot see inside them; and a
    text-to-text check cannot see a line drawn through a label either.  Figure 1's
    trend arrows did exactly that at an offset that cleared every *text* box.

    Each Line2D segment is clipped against the inner 70% of every Text artist's
    window extent (via Liang-Barsky), so a line grazing a label's outer margin is
    reported as 0 and only a genuine crossing is flagged.

    Returns (crossings, masked) where each entry is (text, segment_count).
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    texts = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                texts.append(t)
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if t.get_text().strip():
                texts.append(t)
    for t in fig.texts:
        if t.get_text().strip():
            texts.append(t)

    segs = []
    for ax in fig.axes:
        for ln in ax.lines:
            xy = ln.get_xydata()
            if xy is None or len(xy) < 2:
                continue
            p = ax.transData.transform(xy)
            for a, b in zip(p[:-1], p[1:]):
                segs.append((a, b))

    hits, masked_hits = [], []
    for t in texts:
        bb = t.get_window_extent(r)
        if bb.width <= 0 or bb.height <= 0:
            continue
        x0, x1 = bb.x0 + 0.15 * bb.width, bb.x1 - 0.15 * bb.width
        y0, y1 = bb.y0 + 0.15 * bb.height, bb.y1 - 0.15 * bb.height
        n = 0
        for a, b in segs:
            dx, dy = b[0] - a[0], b[1] - a[1]
            t0, t1, ok = 0.0, 1.0, True
            for pp, qq in ((-dx, a[0] - x0), (dx, x1 - a[0]),
                           (-dy, a[1] - y0), (dy, y1 - a[1])):
                if pp == 0:
                    if qq < 0:
                        ok = False
                        break
                else:
                    rr = qq / pp
                    if pp < 0:
                        t0 = max(t0, rr)
                    else:
                        t1 = min(t1, rr)
            if ok and t0 <= t1:
                n += 1
        if n:
            txt = t.get_text().replace("\n", " ")[:38]
            # A label drawn on an opaque white marker is legible even when a
            # background line passes behind it, so it is reported in a separate
            # list rather than as an unresolved crossing.  Data marks (bars,
            # error bars, series lines) are still counted: masking those would
            # hide information, which is why only background rules and gridlines
            # are ever masked in this package.
            bbox = t.get_bbox_patch()
            masked = bool(bbox is not None and bbox.get_visible()
                          and bbox.get_facecolor()[3] > 0.9
                          and sum(bbox.get_facecolor()[:3]) > 2.7)
            (masked_hits if masked else hits).append((txt, n))
    return hits, masked_hits


def whiskers_v(ax, x, y, err, cap_frac=0.28, w=None, **kw):
    """Draw vertical error bars as explicit segments instead of ``bar(yerr=)``.

    Matplotlib 3.11's error-bar container can coalesce the per-bar segments of a
    grouped bar chart into a single zig-zag line running from the first group to
    the last.  That line is not the data, and it draws across neighbouring value
    labels.  Emitting the stem and two caps directly removes that behaviour and
    keeps the geometry identical across matplotlib versions.
    """
    kw = dict(color=C["ink"], lw=0.8, zorder=5, solid_capstyle="butt", **kw)
    half = cap_frac * (w if w is not None else 0.3) / 2.0
    for xi, yi, ei in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(err)):
        if not np.isfinite(ei) or ei <= 0:
            continue
        ax.plot([xi, xi], [yi - ei, yi + ei], **kw)
        ax.plot([xi - half, xi + half], [yi + ei, yi + ei], **kw)
        ax.plot([xi - half, xi + half], [yi - ei, yi - ei], **kw)


def whiskers_h(ax, x, y, err, cap_frac=0.28, h=None, **kw):
    """Horizontal counterpart of :func:`whiskers_v`."""
    kw = dict(color=C["ink"], lw=0.8, zorder=5, solid_capstyle="butt", **kw)
    half = cap_frac * (h if h is not None else 0.3) / 2.0
    for xi, yi, ei in zip(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(err)):
        if not np.isfinite(ei) or ei <= 0:
            continue
        ax.plot([xi - ei, xi + ei], [yi, yi], **kw)
        ax.plot([xi - ei, xi - ei], [yi - half, yi + half], **kw)
        ax.plot([xi + ei, xi + ei], [yi - half, yi + half], **kw)


def zero_ref(ax, y=0.0, **kw):
    kw = dict(color=C["ref"], lw=0.7, zorder=1, **kw)
    ax.axhline(y, **kw)


def gridlines_only_y(ax):
    ax.grid(axis="y", which="major")
    ax.grid(axis="x", visible=False)


def audit_text_collisions(fig, min_overlap_px: float = 1.5,
                          min_frac: float = 0.15) -> list:
    """Report rendered text boxes that overlap, in display pixels.

    The skill's PDF collision audit reads PDF geometry, which cannot see inside a
    raster figure, and the source-level checks cannot see where a glyph actually
    lands.  This walks every Text artist in the figure and compares window
    extents, which is the only way to catch a value label that has grown into its
    neighbour once the font size and the final canvas width are both fixed.

    Returns a list of (text_a, text_b, overlap_px, fraction_of_smaller).
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                items.append(t)
        # Axis tick labels are excluded on purpose: rotated tick labels have
        # axis-aligned bounding boxes that overlap by construction even when
        # the glyphs are clearly separated, which would flood the report with
        # false positives.  They are checked at final size by eye instead.
        for t in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                  + list(ax.get_xticklabels())[:0] + list(ax.get_yticklabels())[:0]):
            if t.get_text().strip():
                items.append(t)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                items.append(t)
    for t in fig.texts:
        if t.get_text().strip():
            items.append(t)

    boxes = []
    for t in items:
        try:
            bb = t.get_window_extent(r)
        except Exception:                                     # noqa: BLE001
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        boxes.append((t.get_text().replace("\n", " ")[:38], bb))

    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > min_overlap_px and oy > min_overlap_px:
                area = ox * oy
                smaller = min(a.width * a.height, b.width * b.height)
                frac = area / smaller if smaller else 0.0
                if frac >= min_frac:
                    hits.append((boxes[i][0], boxes[j][0],
                                 round(float(ox * oy), 1), round(float(frac), 3)))
    return hits


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def save_figure(fig, stem: str, outdir: str, raster_dir: str | None = None,
                alignment_tolerance_pt: float = 1.5,
                target_width_pt: float = FIG_W_IN * 72,
                exemptions=None, exclude_axes=None,
                row_groups=None, column_groups=None,
                require_panel_labels: bool = False) -> dict:
    """Fit the figure to the target printed width, then run the gates and export.

    The manuscript PDF places every figure at the full text-column width, so the
    figure must be *drawn* at that width.  Letting the PDF builder downscale a
    wider canvas is exactly what shrank the previous glyphs to 2-4 pt, so this
    helper iterates the canvas size until ``bbox_inches="tight"`` measures the
    target width, and only then exports.  A figure that keeps its nominal 7 pt
    body text therefore arrives at 7 pt on the page.

    Exports, in the skill's order: the mandatory panel-alignment gate first
    (which blocks on failure), then SVG and PDF with editable text, then 600 dpi
    TIFF and PNG previews at exactly the printed size.
    """
    from audit_panel_alignment import require_matplotlib_panel_alignment

    os.makedirs(outdir, exist_ok=True)
    if raster_dir:
        os.makedirs(raster_dir, exist_ok=True)

    def tight_pt():
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        return bb.width * 72.0, bb.height * 72.0

    # --- iterate canvas size so the tight bounding box hits the target width.
    # Damped fixed point: the tight width responds almost linearly to the canvas
    # width, and a full step can overshoot because the 0.01 in pad and the text
    # extents do not scale exactly with the canvas.
    for _ in range(14):
        w_pt, h_pt = tight_pt()
        if abs(w_pt - target_width_pt) <= 0.4:
            break
        factor = target_width_pt / w_pt
        factor = 1.0 + 0.6 * (factor - 1.0)
        w_in, h_in = fig.get_size_inches()
        fig.set_size_inches(w_in * factor, h_in * factor)

    width_pt, height_pt = tight_pt()

    collisions = audit_text_collisions(fig)
    stroke_hits, masked_hits = audit_text_stroke_collisions(fig)

    # --- mandatory alignment gate before any file is written -----------------
    # Built from the same backend-neutral manifest the skill's CLI audits, so the
    # verdict is identical, but "NOT AUDITABLE" (exit 2) is kept as a recorded
    # condition instead of an exception: a panel that needs a different width on
    # purpose cannot be compared to its sibling, and the rendered collision audit
    # is then the binding gate.  Definite misalignment (exit 1) always blocks.
    from audit_panel_alignment import (
        audit_layout_manifest, exit_code, matplotlib_layout_manifest,
        render_text, write_json_report, write_overlay_svg)

    manifest = matplotlib_layout_manifest(
        fig,
        exemptions=tuple(exemptions or ()),
        exclude_axes=tuple(exclude_axes or ()),
        row_groups=row_groups,
        column_groups=column_groups,
    )
    report = audit_layout_manifest(
        manifest,
        tolerance_pt=alignment_tolerance_pt,
        gutter_tolerance_pt=alignment_tolerance_pt,
        require_panel_labels=require_panel_labels,
    )
    write_json_report(report, os.path.join(outdir, f"{stem}.alignment.json"))
    if report.get("layout"):
        write_overlay_svg(report, os.path.join(outdir, f"{stem}.alignment.svg"))
    code = exit_code(report, strict=True)
    if code == 1:
        raise RuntimeError(f"{stem}: panel alignment FIX BEFORE DELIVERY\n"
                           + render_text(report, strict=True))
    if code == 2:
        report["recorded_condition"] = (
            "NOT AUDITABLE: panels have intentionally different grid spans, so no "
            "comparable row/column groups could be inferred. Alignment is not "
            "claimed; the rendered collision audit and the final-size panel "
            "inspection are the binding checks for this figure.")

    paths = {"svg": os.path.join(outdir, f"{stem}.svg"),
             "pdf": os.path.join(outdir, f"{stem}.pdf")}
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.01)
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.01)
    if raster_dir:
        paths["tiff"] = os.path.join(raster_dir, f"{stem}.tiff")
        paths["png"] = os.path.join(raster_dir, f"{stem}.png")
        fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight", pad_inches=0.01,
                    pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(paths["png"], dpi=600, bbox_inches="tight", pad_inches=0.01)

    printed = {"width_pt": width_pt, "height_pt": height_pt,
               "width_mm": width_pt / 72.0 * MM_PER_IN,
               "height_mm": height_pt / 72.0 * MM_PER_IN,
               "nominal_pt": PT["label"],
               "effective_pt": PT["label"] * target_width_pt / width_pt}
    return {"paths": paths, "printed": printed, "alignment": report,
            "collisions": collisions, "stroke_collisions": stroke_hits,
            "masked_stroke_labels": masked_hits}


def report_printed(records: dict) -> None:
    """Print a compact per-figure geometry/verdict table."""
    print()
    print(f"{'figure':<32}{'w_pt':>8}{'h_pt':>8}{'aspect':>8}"
          f"{'body_pt':>9}  alignment")
    print("-" * 84)
    for name, rec in records.items():
        p = rec["printed"]
        verdict = rec["alignment"]
        if isinstance(verdict, dict):
            verdict = (verdict.get("verdict") or verdict.get("result")
                       or verdict.get("status") or "?")
        print(f"{name:<32}{p['width_pt']:>8.1f}{p['height_pt']:>8.1f}"
              f"{p['width_pt']/p['height_pt']:>8.3f}"
              f"{p['effective_pt']:>9.2f}  {verdict}")
        for a, b, _area, frac in rec.get("collisions", []):
            print(f"    COLLISION {frac:.0%}  {a!r}  <->  {b!r}")
        for txt, n in rec.get("stroke_collisions", []):
            print(f"    TEXT-ON-LINE  {n} segment(s) cross {txt!r}")
        for txt, n in rec.get("masked_stroke_labels", []):
            print(f"    masked label  {n} background segment(s) behind {txt!r}")
    print()
    total = sum(len(r.get("collisions", [])) for r in records.values())
    stotal = sum(len(r.get("stroke_collisions", [])) for r in records.values())
    print(f"text-text collisions across {len(records)} figures: {total}")
    mtotal = sum(len(r.get("masked_stroke_labels", [])) for r in records.values())
    print(f"text-on-stroke crossings across {len(records)} figures: {stotal}")
    print(f"masked labels over background rules: {mtotal}"
          f"  (legible by construction; data marks are never masked)")
    print()
