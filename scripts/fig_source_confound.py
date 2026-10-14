# -*- coding: utf-8 -*-
"""
Figure S6：ARC-MOF 内部 8 个来源子库的化学–拓扑混淆程度

(a) 两种**真实**化学粒度（coarse = mc+flig；fine = mc+flig+func）下的
    单拓扑化学组成占比，按 fine 值排序，误差棒为 bootstrap sd。
(b) 混淆程度**超出自身置换零模型**的部分 vs 拓扑熵 H(T)——
    用于说明来源间的差异**不能**由拓扑多样性（richness）解释。
    （main text §3.15 第 2 条读数的图，回应审计：正文引用的
     "does not correlate significantly with H(T)" 原先指向一个与 H(T)
     无关的面板，(b) 现在就是那个面板。）

零模型：`null_model_controls.py` 的 source 内部逐样本拓扑标签置换，
`outputs/common_support/null_topology_cardinality.csv`（coarse 粒度、
common support），b = 1,000。

数据来源：`outputs/source_confound_granularity.csv`
（生产脚本为 `source_confound_granularity.py`；本脚本只读该 CSV 出图）。
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

C_COARSE = "#8C9BB5"
C_FINE = "#B4423C"
C_ACC = "#2F6E8C"
DEGEN = "#9A9A9A"     # 退化来源（只有 1 个拓扑）

plt.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def main():
    src = os.path.join(OUT, "source_confound_granularity.csv")
    d = pd.read_csv(src)
    need = ["coarse_frac_single", "fine_frac_single"]
    if not all(c in d.columns for c in need):
        raise SystemExit("✗ CSV 缺少 coarse_frac_single / fine_frac_single 列——"
                         "请用最新版 source_confound_granularity.py 重新生成")
    if d[need].isna().any().any():
        raise SystemExit("✗ CSV 中有 NaN——先跑完再出图")

    # ---- 面板 (b) 所需的零模型（source 内部逐样本置换，coarse，common support）----
    null_p = os.path.join(OUT, "common_support", "null_topology_cardinality.csv")
    if not os.path.exists(null_p):
        null_p = os.path.join(OUT, "null_topology_cardinality.csv")
    nd = pd.read_csv(null_p)[["source", "observed", "null_mean", "b"]]
    d = d.merge(nd, on="source", how="left", validate="one_to_one")
    d["excess_pp"] = (d["observed"] - d["null_mean"]) * 100.0

    d = d.sort_values("fine_frac_single").reset_index(drop=True)
    labels = d["source"].astype(str)
    n = len(d)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---------------- (a) 两级真实粒度 ----------------
    ax = axes[0]
    x = np.arange(n)
    w = 0.38
    b_c = ax.bar(x - w / 2, d["coarse_frac_single"], w,
                 color=C_COARSE, label="coarse chemistry (metal-centre + linker)")
    b_f = ax.bar(x + w / 2, d["fine_frac_single"], w,
                 color=C_FINE, label="fine chemistry (+ functional groups)")
    if "boot_sd" in d.columns:
        ax.errorbar(x + w / 2, d["fine_frac_single"], yerr=d["boot_sd"],
                    fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2.5)

    # 标注退化来源（拓扑数为 1，指标恒等于 1）
    if "n_topologies" in d.columns:
        for i, nt in enumerate(d["n_topologies"]):
            if nt <= 1:
                b_f[i].set_color(DEGEN)
                ax.text(i + w / 2, 1.006, "1 topology\n(degenerate)",
                        ha="center", va="bottom", fontsize=7, color=DEGEN)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nn={int(v):,}" for s, v in
                        zip(labels, d["n_samples"])], fontsize=8.2)
    ax.set_ylabel("fraction of compositions occurring in a single topology")
    ax.set_ylim(0.75, 1.075)
    ax.set_title("(a) Confounding differs by source, at both real granularities",
                 loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    if "boot_sd" in d.columns:
        ax.text(0.02, 0.97, "fine bars: error bars = bootstrap sd (10 resamples)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.4, color="#555555")

    # ---------------- (b) 超出自身置换零模型的部分 vs 拓扑熵 H(T) ----------------
    # 直接检验 main text §3.15 第 2 条读数："来源间差异不是拓扑丰富度
    # （topology richness）的机械投影"。若差异只是"拓扑越多、单拓扑占比越低"
    # 的算术后果，excess over null 就应随 H(T) 系统变化；实测不显著。
    ax = axes[1]
    sub = d[d["n_topologies"] > 1].copy()          # 丢掉退化的 DB6
    xv = sub["H_T"].to_numpy(dtype=float)
    yv = sub["excess_pp"].to_numpy(dtype=float)
    ax.scatter(xv, yv, s=62, color=C_ACC, marker="o",
               edgecolor="white", linewidth=0.8, zorder=3)
    for xi, yi, s in zip(xv, yv, sub["source"].astype(str)):
        ax.annotate(s, (xi, yi), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.6, color="#444444")
    ax.axhline(0.0, color="#999999", lw=0.9, ls="--", zorder=1)
    txt = []
    if len(sub) >= 4 and np.std(xv) > 0 and np.std(yv) > 0:
        pr, pp = pearsonr(xv, yv)
        sr, sp = spearmanr(xv, yv)
        txt.append(f"Pearson r = {pr:+.2f} (p = {pp:.2f})")
        txt.append(f"Spearman ρ = {sr:+.2f} (p = {sp:.2f})")
        # 画一条最小二乘参考线（仅示方向，不做外推）
        k, b0 = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 50)
        ax.plot(xs, k * xs + b0, color=C_ACC, lw=1.0, alpha=0.55, zorder=2)
    ax.set_xlabel("topology entropy H(T) of the source (bit)")
    ax.set_ylabel("excess over its own permutation null\n"
                  "(% points above null)")
    ax.set_title("(b) Topology richness does NOT explain the spread",
                 loc="left", fontweight="bold")
    ax.margins(x=0.10, y=0.14)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    if txt:
        ax.text(0.03, 0.96, "\n".join(txt) + f"\n(n = {len(sub)} sources)",
                transform=ax.transAxes, fontsize=7.4, va="top", ha="left",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF",
                          ec="#CCCCCC", lw=0.7))

    fig.suptitle("ARC-MOF: chemistry–topology confounding is a property of "
                 "how each source sub-library was constructed",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    p = os.path.join(FIGDIR, "figS6_source_confound.png")
    fig.savefig(p)
    print("已保存:", p)
    # 同时保留 outputs/ 下的存档副本（与其它 SI 图一致）
    p2 = os.path.join(OUT, "figS6_source_confound.png")
    fig.savefig(p2)
    print("已保存:", p2)

    # ---------------- 终端摘要 ----------------
    print("\n" + "=" * 72)
    print("Figure S6 数据摘要")
    print("=" * 72)
    print(d[["source", "n_samples", "n_topologies", "H_T",
             "coarse_r", "coarse_frac_single",
             "fine_r", "fine_frac_single",
             "null_mean", "excess_pp"]].round(4).to_string(index=False))
    sr, sp = spearmanr(d["coarse_frac_single"], d["fine_frac_single"])
    print(f"\n两级粒度之间的 Spearman ρ = {sr:+.3f} (p = {sp:.4f})")
    print(f"fine 范围: {d['fine_frac_single'].min():.4f} ~ "
          f"{d['fine_frac_single'].max():.4f}")
    print(f"\n(b) excess over null 范围: {sub['excess_pp'].min():+.1f} ~ "
          f"{sub['excess_pp'].max():+.1f} pp（n = {len(sub)}）")


if __name__ == "__main__":
    main()
