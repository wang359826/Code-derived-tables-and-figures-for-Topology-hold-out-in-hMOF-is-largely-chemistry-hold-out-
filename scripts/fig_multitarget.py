#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
出 Figure 6 / 7：10 个目标的跨目标规律
=====================================

Figure 6   ΔR²（左）与惩罚（右）对"结构描述符单独解释力"的散点图
           —— 核心信息：惩罚不是常数，且可被 R²(仅结构描述符) 预测。
Figure 7   10 个目标 × 三种设定（随机 / 拓扑留出 / 仅结构描述符）的 R² 并列柱状图
           —— 核心信息：拓扑留出的柱子**全部低于**仅结构描述符的柱子（10/10）。

注：2026-10-11 图序修订后，这两张图由原 Figure 10 / 11 变为 Figure 6 / 7。

用法
----
    python fig_multitarget.py

输出
----
    outputs/fig6_penalty_vs_geometry.png
    outputs/fig7_multitarget_bars.png
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

# 图上标签用英文，避免中文字体缺失
SHORT = {
    "CO2 uptake 0.15 bar (post-combustion)": "CO2 uptake 0.15bar/298K",
    "CO2 uptake 0.10 bar / 363 K (TSA regen)": "CO2 uptake 0.10bar/363K",
    "CO2 uptake 0.70 bar / 413 K": "CO2 uptake 0.70bar/413K",
    "excess CO2 uptake 0.15 bar": "excess CO2 uptake",
    "CO2 uptake, CO2/N2 binary": "CO2 uptake (binary)",
    "N2 uptake, CO2/N2 binary": "N2 uptake (binary)",
    "working capacity (VSA)": "working cap. (VSA)",
    "working capacity (TSA)": "working cap. (TSA)",
    "heat of adsorption CO2": "heat of adsorption",
    "log10(CO2/N2 selectivity)": "log10 selectivity",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(os.path.join(OUT, "multitarget_confound.csv"))
    d["short"] = d["label"].map(SHORT).fillna(d["label"])
    # ⚠️ R10：旧版把"随机划分+仅结构"与"拓扑留出+身份"对比，同时换了两个因素。
    #    现在改用**同划分**的基线：R2_topology_geo（拓扑留出 + 仅结构）。
    d["penalty"] = d["R2_topology_geo"] - d["R2_topology_full"]   # = −identity_effect_topo
    BASE = "R2_topology_geo"

    # ---------------- Figure 10 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    for ax, ycol, title in (
        (axes[0], "split_effect_full", "Split effect: dR2 (random - topology), same features"),
        (axes[1], "identity_effect_topo", "Identity effect: with - without identity, same split"),
    ):
        r, p = pearsonr(d[BASE], d[ycol])
        ax.scatter(d[BASE], d[ycol], s=68,
                   facecolor="#4C72B0", edgecolor="#1f3a63", zorder=3)
        # 最小二乘趋势线
        b, a = np.polyfit(d[BASE], d[ycol], 1)
        xs = np.linspace(d[BASE].min() - 0.02,
                         d[BASE].max() + 0.02, 50)
        ax.plot(xs, a + b * xs, "--", color="#C44E52", lw=1.6, zorder=2)
        ax.set_xlabel("R2 of structure-derived descriptors only "
                   "(topology hold-out, same split)")
        ax.set_ylabel(title)
        ax.set_title(f"{title}\nPearson r = {r:+.3f}  (p = {p:.4f})", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_facecolor("#ffffff")
        for s in ("bottom", "top", "left", "right"):
            ax.spines[s].set_color("#888888")

    # 只标两个最有代表性的点，避免互相遮挡
    for ax in axes:
        for _, row in d.iterrows():
            if row["short"] in ("excess CO2 uptake", "CO2 uptake 0.70bar/413K",
                                "log10 selectivity"):
                ax.annotate(row["short"],
                            (row["R2_geometry_only"], row["delta_R2"]
                             if ax is axes[0] else row["penalty"]),
                            textcoords="offset points", xytext=(6, 6), fontsize=8,
                            color="#222222")

    fig.tight_layout()
    p10 = os.path.join(OUT, "fig6_penalty_vs_geometry.png")
    fig.savefig(p10, dpi=170)
    print("saved:", p10)

    # ---------------- Figure 7 ----------------
    d2 = d.sort_values("R2_random", ascending=True)
    idx = np.arange(len(d2))
    w = 0.21
    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    ax.bar(idx - 1.5 * w, d2["R2_random_full"], w,
           label="random split, with identity", color="#4C72B0")
    ax.bar(idx - 0.5 * w, d2["R2_random_geo"], w,
           label="random split, descriptors only", color="#8DA8D6")
    ax.bar(idx + 0.5 * w, d2["R2_topology_full"], w,
           label="topology hold-out, with identity", color="#C44E52")
    ax.bar(idx + 1.5 * w, d2["R2_topology_geo"], w,
           label="topology hold-out, descriptors only", color="#55A868")
    ax.set_xticks(idx)
    ax.set_xticklabels(d2["short"], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("R2")
    ax.set_ylim(0, 1.05)
    ax.set_title("2x2 design: the SPLIT moves R2 (median dR2 = 0.324); "
                 "identity features do not (median -0.011, p = 0.28)", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_facecolor("#ffffff")
    for s in ("bottom", "top", "left", "right"):
        ax.spines[s].set_color("#888888")
    fig.tight_layout()
    p11 = os.path.join(OUT, "fig7_multitarget_bars.png")
    fig.savefig(p11, dpi=170)
    print("saved:", p11)


if __name__ == "__main__":
    main()
