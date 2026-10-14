#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure S7：逐拓扑的「化学未见面率 → 留出代价」散点（S_g 曲线）

数据：`outputs/s_g_curve.csv`（由 `scripts/s_g_curve.py` 在 100,000 抽样上
逐拓扑留出生成；每个点 = 一个被留出的拓扑）。

⚠️ 结果是否证性的，出图时必须与数字一起呈现，不得只画趋势线：
  · 因变量 R² 在各点上是**不同测试集**上的值，不构成可比性主张；
  · S_miss 与 R² 的相关不显著（见下文终端输出）；
  · 唯一显著的结构是 §3.15 已报告的那一条：pcu 的 sd_y 约为其余拓扑的
    2–4 倍，池化 R² 因此被它主导（去掉 pcu 后相关升到 r = +0.53, p = 0.11，
    仍不显著）。
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
FIGDIR = os.path.join(HERE, "..", "..", "01_manuscript", "figures")

C_PT = "#2F6E8C"
C_PCU = "#B4423C"
C_TREND = "#8C9BB5"

plt.rcParams.update({"font.size": 9.5, "axes.labelsize": 10,
                     "axes.titlesize": 11, "figure.dpi": 150,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def main():
    d = pd.read_csv(os.path.join(OUT, "s_g_curve.csv"))
    d = d.sort_values("S_miss_unseen").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    # ---------- (a) 每个拓扑的 S_miss 与其留出代价（哑铃图）----------
    # 7 个拓扑共享 S_miss = 1.000，却在 R² 上从 −2.55 跨到 +0.47。散点图在这种
    # 重叠下无法标注，改用哑铃/棒棒糖图：横轴 = S_miss，颜色 = 该折 R² 的符号与大小，
    # 于是"同一 S_miss 上代价差异巨大"直接可见，且无一标签重叠。
    ax = axes[0]
    oa = d.sort_values(["S_miss_unseen", "R2_pooled"]).reset_index(drop=True)
    ypos = np.arange(len(oa))
    import matplotlib.cm as cm
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=min(-1.0, oa["R2_pooled"].min()),
                        vcenter=0.0, vmax=max(1.0, oa["R2_pooled"].max()))
    for i, (_, r) in enumerate(oa.iterrows()):
        ax.plot([0, r["S_miss_unseen"]], [i, i], color="#D8D8D8", lw=2.4,
                solid_capstyle="round", zorder=1)
        ax.scatter(r["S_miss_unseen"], i, s=86, marker="o",
                   c=[cm.RdBu(norm(r["R2_pooled"]))],
                   edgecolor="#333333", linewidth=0.6, zorder=3)
        ax.text(1.12, i, f"R² {r['R2_pooled']:+.2f}", fontsize=7.0,
                va="center", ha="left", color="#333333")
    ax.axvline(1.0, color="#999999", lw=0.8, ls=":", zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{t}  (n={int(n):,})"
                        for t, n in zip(oa["held_out_topology"], oa["n_test"])],
                       fontsize=7.6)
    ax.set_ylim(-1.6, len(oa) - 0.4)
    ax.set_xlim(-0.04, 1.42)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("S$_{miss}$ — fraction of the held-out topology's\n"
                  "structures with no chemical counterpart in training")
    ax.set_title("(a) Rows ordered by S$_{miss}$; the seven rows at\n"
                 "S$_{miss}$ = 1.00 span R² from −2.55 to +0.47",
                 loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    sub = d[d["n_test"] >= 100]
    if len(sub) >= 4:
        x, y = sub["S_miss_unseen"].to_numpy(float), sub["R2_pooled"].to_numpy(float)
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        ax.text(0.10, 0.95,
                f"n = {len(sub)} topologies with n_test ≥ 100\n"
                f"Pearson r = {pr:+.2f} (p = {pp:.2f})\n"
                f"Spearman ρ = {sr:+.2f} (p = {sp:.2f})\n"
                f"→ no detectable relation",
                transform=ax.transAxes, fontsize=7.2, va="top", ha="left",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.32", fc="#FFFFFF",
                          ec="#CCCCCC", lw=0.7))
    ax.set_xlabel("S$_{miss}$ — fraction of the held-out topology's\n"
                  "structures with no chemical counterpart in training")
    ax.set_ylabel("held-out topology (rows ordered by S$_{miss}$)")
    ax.set_title("(a) The unseen-chemistry rate does not order the\n"
                 "per-topology hold-out cost", loc="left", fontweight="bold")
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.10)

    # ---------- (b) 为什么：pcu 主导池化值 ----------
    ax = axes[1]
    order_b = d.sort_values("sd_y_test").reset_index(drop=True)
    ypos = np.arange(len(order_b))
    ax.barh(ypos, order_b["sd_y_test"],
            color=[C_PCU if t == "pcu" else C_PT
                   for t in order_b["held_out_topology"]])
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{t}  (n={int(n):,})"
                        for t, n in zip(order_b["held_out_topology"],
                                        order_b["n_test"])], fontsize=7.8)
    ax.set_xlabel("sd of the target within the held-out topology (mmol/g)")
    ax.set_title("(b) Why: R² normalises by each fold's own variance,\n"
                 "and pcu's is 2–4× the others'", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(order_b["sd_y_test"]) * 1.28)
    ip = int(order_b.index[order_b.held_out_topology == "pcu"][0])
    ax.annotate("pcu: 63% of the samples,\nthe largest fold variance,\n"
                "so it dominates the pooled value",
                xy=(order_b.loc[ip, "sd_y_test"], ip),
                xytext=(0.52, 0.34), textcoords="axes fraction",
                fontsize=7.3, color=C_PCU, va="center",
                arrowprops=dict(arrowstyle="->", color=C_PCU, lw=0.9))

    fig.suptitle("S_g curve: across topologies, a higher unseen-chemistry rate "
                 "does not predict a larger hold-out cost",
                 fontsize=11.0, y=1.02)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for p in (os.path.join(FIGDIR, "figS7_sg_curve.png"),
              os.path.join(OUT, "figS7_sg_curve.png")):
        fig.savefig(p)
        print("已保存:", p)

    # ---------- 终端汇总 ----------
    from scipy.stats import pearsonr as pr_
    print("\n" + "=" * 72)
    print("S_g 曲线汇总（100,000 抽样，逐拓扑留出）")
    print("=" * 72)
    print(d[["held_out_topology", "n_test", "S_miss_unseen", "r_train",
             "sd_y_test", "R2_pooled"]].round(3).to_string(index=False))
    for name, s in (("all", d),
                    ("n_test>=100", d[d.n_test >= 100]),
                    ("no pcu", d[d.held_out_topology != "pcu"]),
                    ("n_test>=100 & no pcu",
                     d[(d.n_test >= 100) & (d.held_out_topology != "pcu")])):
        x, y = s["S_miss_unseen"].to_numpy(float), s["R2_pooled"].to_numpy(float)
        if len(s) >= 4 and np.std(x) > 0 and np.std(y) > 0:
            r, p = pr_(x, y)
            print(f"  {name:22s} n={len(s):2d}  Pearson r={r:+.3f} (p={p:.3f})")


if __name__ == "__main__":
    main()
