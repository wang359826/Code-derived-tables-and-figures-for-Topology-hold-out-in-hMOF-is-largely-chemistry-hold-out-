#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1（论文主图）：同一模型、同一份数据，仅改划分策略
                      —— 数据泄漏的严重程度取决于目标变量的物理本质

左：CO2 吸附量（10 万样本）与 log10(CO2/N2 选择性)（10 万样本）在四种划分下的 R²
右：虚高幅度 ΔR² 与相对虚高的对比 —— 吸附量 99% vs 选择性 16%，相差约 6 倍

数据来源（均为已实跑结果，本脚本不重算，只绘图）：
    outputs/leakage_results_final.csv        （吸附量，10 万样本，RandomForest）
    outputs/leakage_results_logsel100k.csv   （log10 选择性，10 万样本）

运行
----
    python plot_main_figure.py --outdir outputs
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

RED = "#A32D2D"        # 随机划分（虚高 / 警示）
BLUE = "#185FA5"       # 严格划分
GREY = "#5F5E5A"

# 横轴：划分严格程度由弱到强
# （用前缀匹配，因为脚本在不同开关下会给同一策略不同的后缀，
#   例如「按拓扑分组（GroupKFold）」与「按拓扑分组（未见拓扑）」）
ORDER = [
    ("随机划分", "random\n(most papers)"),
    ("按有机连接体分组", "grouped by\norganic linker"),
    ("按金属中心分组", "grouped by\nmetal centre"),
    ("按拓扑分组", "grouped by\ntopology"),
]
ORDER_SEL = ORDER


def load(path: str, order) -> pd.DataFrame:
    d = pd.read_csv(path)
    idx = [str(s) for s in d["strategy"]]
    rows = []
    for prefix, label in order:
        hit = [i for i, s in enumerate(idx) if s.startswith(prefix)]
        if not hit:
            raise KeyError(f"{path} 中找不到以「{prefix}」开头的策略；"
                           f"实际有: {idx}")
        rows.append({"label": label,
                     "R2": float(d.iloc[hit[0]]["R2"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uptake", default="outputs/leakage_results_final.csv")
    ap.add_argument("--selectivity", default="outputs/leakage_results_logsel100k.csv")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--out", default=None,
                    help="输出文件名（仅文件名），默认 fig1_leakage_two_targets.png")
    args = ap.parse_args()

    up = load(args.uptake, ORDER)
    se = load(args.selectivity, ORDER_SEL)

    d_up = up["R2"].iloc[0] - up["R2"].iloc[-1]        # 吸附量 ΔR²
    d_se = se["R2"].iloc[0] - se["R2"].iloc[-1]        # 选择性 ΔR²
    rel_up = d_up / max(up["R2"].iloc[-1], 1e-9) * 100
    rel_se = d_se / max(se["R2"].iloc[-1], 1e-9) * 100

    x = np.arange(len(up))
    w = 0.36

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1.85, 1.0]}
    )

    # ---------------- 左：两种目标变量在四种划分下的 R² ----------------
    b1 = ax1.bar(x - w / 2, up["R2"], w, color=RED,
                 label="CO$_2$ uptake (10$^5$ MOFs)")
    b2 = ax1.bar(x + w / 2, se["R2"], w, color=BLUE,
                 label="log$_{10}$(CO$_2$/N$_2$ selectivity), 10$^5$ MOFs")

    for bars, vals in ((b1, up["R2"]), (b2, se["R2"])):
        for rect, v in zip(bars, vals):
            ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.012,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8.2)

    # 用双箭头标出各自的 ΔR²
    def annotate_drop(ax, x0, y0, x1, y1, color, text, dx, ha="left"):
        ax.annotate("", xy=(x1 + dx, y1), xytext=(x0 + dx, y0),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.6,
                                    shrinkA=1, shrinkB=1))
        ax.text(x1 + dx + (0.08 if ha == "left" else -0.08),
                (y0 + y1) / 2, text, color=color,
                fontsize=9.4, fontweight="bold", va="center", ha=ha,
                # 白描边：文字即使压在柱子上也清晰可读
                path_effects=[pe.withStroke(linewidth=3.0,
                                            foreground="white")])

    annotate_drop(ax1, 0, up["R2"].iloc[0], 3, up["R2"].iloc[-1], RED,
                  f"$\\Delta R^2$ = {d_up:.3f}\n(+{rel_up:.0f}%)", -0.46)
    annotate_drop(ax1, 0, se["R2"].iloc[0], 3, se["R2"].iloc[-1], BLUE,
                  f"$\\Delta R^2$ = {d_se:.3f}\n(+{rel_se:.0f}%)", +0.46)

    ax1.set_xticks(x)
    ax1.set_xticklabels(up["label"], fontsize=8.8)
    ax1.set_ylabel("cross-validated  R²")
    ax1.set_ylim(0, 1.06)
    ax1.set_xlim(-0.72, 3.98)
    ax1.set_title("Same model, same data — only the split protocol changes",
                  fontsize=10.8)
    ax1.legend(fontsize=8.8, loc="lower left")
    ax1.grid(axis="y", alpha=0.25, lw=0.5)
    ax1.set_axisbelow(True)

    # ---------------- 右：虚高幅度对比 ----------------
    y = np.arange(2)[::-1]
    vals = [rel_se, rel_up]
    cols = [BLUE, RED]
    labs = ["log$_{10}$(CO$_2$/N$_2$ selectivity)\n(geometry-dominated)",
            "CO$_2$ uptake\n(chemistry-dominated)"]
    ax2.barh(y, vals, height=0.5, color=cols)
    for yi, v in zip(y, vals):
        ax2.text(v + 1.8, yi, f"{v:.0f}%", va="center", fontsize=11,
                 fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labs, fontsize=8.8)
    ax2.set_xlim(0, 122)
    ax2.set_ylim(-0.65, 1.65)
    ax2.set_xlabel("relative inflation of R²  [%]")
    ax2.set_title("How badly random splitting lies\ndepends on what you predict",
                  fontsize=10.8)
    ax2.grid(axis="x", alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)
    # 两柱之间的空白带放倍数说明，避免压住柱子
    ax2.text(30, 0.5, f"{rel_up / max(rel_se, 1e-9):.1f}× worse",
             ha="left", va="center", fontsize=11.5,
             color=GREY, fontweight="bold")
    ax2.annotate("", xy=(26, 0.62), xytext=(6, 0.38),
                 arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2,
                                 connectionstyle="arc3,rad=-0.25"))

    fig.suptitle("Data leakage in MOF CO₂-capture machine learning: "
                 "chemistry-bound targets are far more vulnerable than geometry-bound ones",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(args.outdir,
                       args.out or "fig1_leakage_two_targets.png")
    os.makedirs(args.outdir, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print(f"已保存: {out}")
    print(f"\nCO2 吸附量   : 随机 {up['R2'].iloc[0]:.4f} → 未见拓扑 "
          f"{up['R2'].iloc[-1]:.4f}   ΔR²={d_up:.4f}  相对虚高 {rel_up:.0f}%")
    print(f"log10 选择性 : 随机 {se['R2'].iloc[0]:.4f} → 未见拓扑 "
          f"{se['R2'].iloc[-1]:.4f}   ΔR²={d_se:.4f}  相对虚高 {rel_se:.0f}%")
    print(f"两种目标的虚高倍数比 = {rel_up / max(rel_se, 1e-9):.2f}×")


if __name__ == "__main__":
    main()
