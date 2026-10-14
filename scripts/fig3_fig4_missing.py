#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1 / Figure S5 —— 补画两张正文图（2026-10-11 重编号后为 Figure 1 与 Figure S5）。

背景
----
投稿版 PDF 里这两张图曾是红色占位框（见图件清单）。本脚本按已核实的数据源补画，
**不重算任何数字**，只做绘图。

Figure S5  留一拓扑 R² 的四种聚合口径
    数据：outputs/metrics_audit.csv（第 0 行 = LOTO / topology 100k）
          第 1 行 = 剔除最大组 pcu 后的同一实验
          outputs/loto_per_topology_final.csv（逐拓扑 R²，用于折叠条带）
    要点：pooled +0.381  vs  macro −0.541  vs  weighted −0.059  vs  median +0.278
          符号随口径翻转；去掉 pcu（63% 样本）后加权降到 −0.678

Figure 1  同一模型、同一数据库、只换目标变量
    数据：outputs/leakage_results_final.csv        （CO2 吸附量，0.15 bar）
          outputs/leakage_results_logsel100k.csv   （log10 CO2/N2 选择性）
    要点：吸附量 随机 0.7529 → 拓扑 0.3786，ΔR² = 0.374
          选择性 随机 0.8357 → 拓扑 0.7211，ΔR² = 0.115
          泄漏幅度是**目标变量**的属性

运行
----
    python fig3_fig4_missing.py --outdir outputs
（本脚本是独立重绘步骤，不在 run_all.py 的任何 level 内；level 2 的
 plot_main_figure.py 亦写 Figure 1 同名文件，以本脚本的版本为最终版。）
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# ---- 与论文其余图一致的配色（fig_cross_database.py / plot_main_figure.py）
C_H = "#2F5D8C"      # hMOF / 蓝（主色）
C_A = "#B4543A"      # 警示 / 红棕
C_E = "#3E7A5E"      # 绿
C_M = "#8A6BBE"      # 紫
GREY = "#5F5E5A"
RED = "#A32D2D"
BLUE = "#185FA5"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "figure.dpi": 150,
})


def f(outdir: str, name: str) -> str:
    return os.path.join(outdir, name)


# ============================================================ Figure S5
def figure_s5(outdir: str) -> str:
    """四种聚合口径：同一组折叠 R²，聚合方式不同 → 符号翻转。"""
    audit = pd.read_csv(f(outdir, "metrics_audit.csv"), encoding="utf-8-sig")
    perfold = pd.read_csv(f(outdir, "loto_per_topology_final.csv"),
                          encoding="utf-8-sig")

    r_all = audit.iloc[0]
    r_nopcu = audit.iloc[1]

    # 四种口径（全 11 组）
    rules = [
        ("pooled\n(all folds merged)", float(r_all["pooled_as_reported"]), C_H),
        ("weighted\n(by test-set size)", float(r_all["weighted"]), C_M),
        ("macro\n(mean over folds)", float(r_all["macro"]), C_A),
        ("median\n(over folds)", float(r_all["median"]), C_E),
    ]
    # 去掉 pcu 后（10 组）
    rules_np = {
        "weighted": float(r_nopcu["weighted"]),
        "macro": float(r_nopcu["macro"]),
        "median": float(r_nopcu["median"]),
    }

    fig = plt.figure(figsize=(13.4, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.62], wspace=0.26)

    # -------- (a) 四种口径的柱子
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(rules))
    vals = [v for _, v, _ in rules]
    cols = [c for _, _, c in rules]
    bars = ax.bar(x, vals, 0.58, color=cols, edgecolor="white", lw=0.8)
    for b, v in zip(bars, vals):
        off = 0.035 if v >= 0 else -0.075
        ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold", color="#1a1a1a")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _, _ in rules], fontsize=8.6)
    ax.set_ylabel("leave-one-topology-out  R²")
    ax.set_ylim(-0.70, 0.50)
    ax.set_title("(a) One experiment, four aggregation rules",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    # 标注：符号不一致
    ax.annotate("sign flips\nwith the rule",
                xy=(2, -0.40), xytext=(2.42, -0.60),
                fontsize=8.6, color=GREY, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1,
                                connectionstyle="arc3,rad=0.22"))

    # -------- (b) 逐拓扑 R² 折叠条带 + 聚合位置
    ax2 = fig.add_subplot(gs[0, 1])
    d = perfold.sort_values("R2", ascending=True).reset_index(drop=True)
    n = len(d)
    ypos = np.arange(n)
    bar_cols = [C_H if v >= 0 else C_A for v in d["R2"]]
    ax2.barh(ypos, d["R2"], height=0.62, color=bar_cols,
             edgecolor="white", lw=0.7)
    # 拓扑名 + n 放在左侧固定栏（不随柱长移动，避免遮挡）
    for yi, (g, v, nt) in enumerate(zip(d["group"], d["R2"], d["n_test"])):
        ax2.text(-3.62, yi + 0.16, f"{g}", va="center", ha="left", fontsize=8.4,
                 fontweight="bold", color="#1a1a1a")
        ax2.text(-3.62, yi - 0.24, f"n={nt:,}", va="center", ha="left",
                 fontsize=6.9, color="#666666")
        # 数值放在柱端外侧；过长（rht −3.01）时改放柱内右侧，避免与左侧名字重叠
        if v < -2.5:
            ax2.text(v + 0.06, yi, f"{v:+.2f}", va="center", ha="left",
                     fontsize=7.6, color="#ffffff", fontweight="bold")
        else:
            ax2.text(v + (0.05 if v >= 0 else -0.05), yi, f"{v:+.2f}",
                     va="center", ha="left" if v >= 0 else "right",
                     fontsize=7.6, color="#333333")
    ax2.axvline(0, color="k", lw=1.0)
    # 四种聚合口径的竖线（标签分两层：远柱端在上，近零在下，避免互相遮挡）
    marks = [
        (float(r_all["pooled_as_reported"]), C_H, "pooled\n+0.381", 2.35),
        (float(r_all["median"]), C_E, "median\n+0.278", 0.95),
        (float(r_all["weighted"]), C_M, "weighted\n−0.059", 2.35),
        (float(r_all["macro"]), C_A, "macro\n−0.541", 0.95),
    ]
    for xv, c, lab, ytag in marks:
        ax2.axvline(xv, color=c, lw=1.5, ls="--", alpha=0.9)
        ax2.text(xv, n - 0.9 + ytag, lab, color=c, fontsize=7.4,
                 ha="center", va="center", fontweight="bold", linespacing=1.2,
                 path_effects=[pe.withStroke(linewidth=2.6,
                                             foreground="white")])
    ax2.set_yticks([])
    ax2.set_ylim(-0.8, n + 2.1)
    ax2.set_xlim(-4.55, 1.05)
    ax2.set_xlabel("per-topology held-out  R²")
    ax2.set_title("(b) The 11 folds behind those four numbers — "
                  "the aggregate is a choice, not a fact",
                  loc="left", fontweight="bold", fontsize=10)
    ax2.grid(axis="x", alpha=0.25, lw=0.6)
    ax2.set_axisbelow(True)

    fig.suptitle("Figure S5.  Leave-one-topology-out R² under four aggregation "
                 "rules — the sign of the aggregate depends on the rule",
                 fontsize=11.6, fontweight="bold", y=1.005)
    fig.text(0.008, -0.035,
             "Same run, same folds (100,000 samples, 8 structure-derived "
             "descriptors). pooling concatenates test predictions and is "
             "dominated by pcu (63% of the data); macro/median weight every "
             "topology equally. Excluding pcu, the weighted value falls to "
             "−0.678 and the macro value to −0.625.  Source: "
             "outputs/metrics_audit.csv, outputs/loto_per_topology_final.csv.",
             fontsize=7.2, color="#555555", ha="left")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.17, wspace=0.13)

    p = f(outdir, "figS5_aggregation_rules.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)
    print(f"  pooled {rules[0][1]:+.4f} | weighted {rules[1][1]:+.4f} | "
          f"macro {rules[2][1]:+.4f} | median {rules[3][1]:+.4f}")
    print(f"  no-pcu: weighted {rules_np['weighted']:+.4f} | "
          f"macro {rules_np['macro']:+.4f} | median {rules_np['median']:+.4f}")
    return p


# ============================================================ Figure 1
def figure1(outdir: str) -> str:
    """同一模型、同一库、只换目标：泄漏幅度是目标变量的属性。"""
    up = pd.read_csv(f(outdir, "leakage_results_final.csv"),
                     encoding="utf-8-sig")
    se = pd.read_csv(f(outdir, "leakage_results_logsel100k.csv"),
                     encoding="utf-8-sig")

    ORDER = [
        ("随机划分", "random\n(most papers)"),
        ("按有机连接体分组", "grouped by\norganic linker"),
        ("按金属中心分组", "grouped by\nmetal centre"),
        ("按拓扑分组", "grouped by\ntopology"),
    ]

    def pick(d, prefix):
        for s, r in zip(d["strategy"].astype(str), d["R2"]):
            if s.startswith(prefix):
                return float(r)
        raise KeyError(prefix)

    up_v = [pick(up, p) for p, _ in ORDER]
    se_v = [pick(se, p) for p, _ in ORDER]
    labs = [lab for _, lab in ORDER]

    d_up = up_v[0] - up_v[-1]
    d_se = se_v[0] - se_v[-1]
    rel_up = d_up / max(up_v[-1], 1e-9) * 100
    rel_se = d_se / max(se_v[-1], 1e-9) * 100

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1.85, 1.0]})

    x = np.arange(len(ORDER))
    w = 0.36
    b1 = ax1.bar(x - w / 2, up_v, w, color=RED,
                 label="CO$_2$ uptake (0.15 bar)")
    b2 = ax1.bar(x + w / 2, se_v, w, color=BLUE,
                 label="log$_{10}$(CO$_2$/N$_2$ selectivity)")
    for bars, vals in ((b1, up_v), (b2, se_v)):
        for rect, v in zip(bars, vals):
            ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.012,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8.2)

    def drop(ax, x0, y0, x1, y1, color, text, dx):
        ax.annotate("", xy=(x1 + dx, y1), xytext=(x0 + dx, y0),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.6,
                                    shrinkA=1, shrinkB=1))
        ax.text(x1 + dx, (y0 + y1) / 2, text, color=color,
                fontsize=9.4, fontweight="bold", va="center", ha="center",
                path_effects=[pe.withStroke(linewidth=3.0,
                                            foreground="white")])

    drop(ax1, 0, up_v[0], 3, up_v[-1], RED,
         f"$\\Delta R^2$ = {d_up:.3f}\n(+{rel_up:.0f}%)", -0.46)
    drop(ax1, 0, se_v[0], 3, se_v[-1], BLUE,
         f"$\\Delta R^2$ = {d_se:.3f}\n(+{rel_se:.0f}%)", +0.46)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labs, fontsize=8.8)
    ax1.set_ylabel("cross-validated  R²")
    ax1.set_ylim(0, 1.06)
    ax1.set_xlim(-0.72, 3.98)
    ax1.set_title("Same database, same model — only the target changes",
                  fontsize=10.8)
    ax1.legend(fontsize=8.8, loc="lower left")
    ax1.grid(axis="y", alpha=0.25, lw=0.5)
    ax1.set_axisbelow(True)

    y = np.arange(2)[::-1]
    vals = [rel_se, rel_up]
    cols = [BLUE, RED]
    labs2 = ["log$_{10}$(CO$_2$/N$_2$ selectivity)\n(geometry-dominated)",
             "CO$_2$ uptake\n(chemistry-dominated)"]
    ax2.barh(y, vals, height=0.5, color=cols)
    for yi, v in zip(y, vals):
        ax2.text(v + 1.8, yi, f"{v:.0f}%", va="center", fontsize=11,
                 fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labs2, fontsize=8.8)
    ax2.set_xlim(0, 122)
    ax2.set_ylim(-0.65, 1.65)
    ax2.set_xlabel("relative inflation of R²  [%]")
    ax2.set_title("How badly random splitting lies\ndepends on what you predict",
                  fontsize=10.8)
    ax2.grid(axis="x", alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)
    ax2.text(30, 0.5, f"{rel_up / max(rel_se, 1e-9):.1f}× worse",
             ha="left", va="center", fontsize=11.5,
             color=GREY, fontweight="bold")
    ax2.annotate("", xy=(26, 0.62), xytext=(6, 0.38),
                 arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2,
                                 connectionstyle="arc3,rad=-0.25"))

    fig.suptitle("Figure 1.  Leakage magnitude is a property of the target — "
                 "same model, same database",
                 fontsize=11.6, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    p = f(outdir, "fig1_leakage_two_targets.png")
    fig.savefig(p, dpi=220)
    plt.close(fig)
    print("saved:", p)
    print(f"  uptake  随机 {up_v[0]:.4f} → 拓扑 {up_v[-1]:.4f}  "
          f"ΔR²={d_up:.4f}  相对虚高 {rel_up:.0f}%")
    print(f"  logsel  随机 {se_v[0]:.4f} → 拓扑 {se_v[-1]:.4f}  "
          f"ΔR²={d_se:.4f}  相对虚高 {rel_se:.0f}%")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    outdir = os.path.abspath(args.outdir)
    figure_s5(outdir)
    figure1(outdir)


if __name__ == "__main__":
    main()
