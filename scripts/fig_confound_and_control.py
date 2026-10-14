# -*- coding: utf-8 -*-
"""
出 Figure 2（化学–拓扑混淆结构）与 Figure 3（受控等网状对照）。
（2026-10-11 图序修订后，这两张图由原 Figure 1 / 2 变为 Figure 2 / 3。）

设计纪律（遵守本项目方法学铁律）：
- Figure 3 的主指标用 MAE / RMSE / MAE÷sd，**不用 R²**。
  原因：pcu 与 nbo 两个测试集的 sd(y) 分别是 0.472 与 0.106，差 4.4 倍，
  方差不可比时 R² 的分母不同，直接相减会得到假信号（见 §2.7）。
- Figure 2 的 full / elite 两栏取自同一份 elite_subset_confound.csv，
  保证是同一种子、同一批样本上的可比数字，不与 chem_topo_confound.csv 混用。

图内标签一律英文（matplotlib 中文会乱码）；正文说明用中文。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

# ---------------------------------------------------------------- 公共样式
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})
C_MAIN = "#2F5D8C"     # 深蓝：主指标
C_ALT = "#B4543A"      # 砖红：对照指标
C_ELITE = "#3E7A5E"    # 绿：精英子集
C_GREY = "#9AA5B1"


def _f(name):
    return os.path.join(OUT, name)


# ================================================================ Figure 2
def figure2():
    conf = pd.read_csv(_f("elite_subset_confound.csv"), encoding="utf-8-sig")
    jac = pd.read_csv(_f("chem_topo_jaccard.csv"), encoding="utf-8-sig")

    # 只取 coarse 粒度做矩阵（与正文 §3.2 的"除 pcu↔nbo 外全为 0"一致）
    jac = jac[jac["key"].astype(str).str.contains("coarse")]
    topos = sorted(set(jac["topo_a"]).union(set(jac["topo_b"])))
    idx = {t: i for i, t in enumerate(topos)}
    M = np.full((len(topos), len(topos)), np.nan)
    for _, r in jac.iterrows():
        a, b = idx[r["topo_a"]], idx[r["topo_b"]]
        M[a, b] = M[b, a] = r["jaccard"]
    np.fill_diagonal(M, 1.0)   # 自身与自身完全重叠，纯展示用

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5),
                             gridspec_kw={"width_ratios": [1.0, 1.05]})

    # ---- (a) 两个混淆标量：full 25k vs elite 8,206
    ax = axes[0]
    order = ["full_25k_coarse", "full_25k_fine", "elite_coarse", "elite_fine"]
    pretty = ["full 25k\ncoarse", "full 25k\nfine", "elite 8,206\ncoarse", "elite 8,206\nfine"]
    conf = conf.set_index("label")
    frac = np.array([conf.loc[o, "frac_single_topology"] for o in order])
    expl = np.array([1.0 - conf.loc[o, "cond_frac"] for o in order])

    x = np.arange(len(order))
    w = 0.38
    b1 = ax.bar(x - w / 2, frac * 100, w, color=C_MAIN,
                label="chemistry mapping to a single topology (%)")
    b2 = ax.bar(x + w / 2, expl * 100, w, color=C_ALT,
                label="topology entropy explained by chemistry (%)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(100, color="k", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=8.5)
    ax.set_ylabel("%")
    ax.set_ylim(0, 118)
    ax.set_title("(a) Chemistry determines topology", loc="left", fontweight="bold")
    ax.legend(fontsize=7.6, loc="lower right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    # 精英子集的底色提示
    ax.axvspan(1.5, 3.5, color=C_ELITE, alpha=0.07, zorder=0)

    # ---- (b) 拓扑对化学集合重叠矩阵
    ax = axes[1]
    Mask = np.where(np.isnan(M), 1.0, M)
    im = ax.imshow(Mask, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(topos)))
    ax.set_yticks(range(len(topos)))
    ax.set_xticklabels(topos, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(topos, fontsize=8)
    ax.set_title("(b) Chemical-set overlap between topology pairs (Jaccard)",
                 loc="left", fontweight="bold")
    # 在非零格子上标数值
    n = len(topos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j]
            if not np.isnan(v) and v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.5 else "black",
                        fontweight="bold")
            else:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=7, color="#B0B7BE")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Jaccard overlap", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("Figure 2. Chemistry–topology confounding in hMOF",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.008, -0.02,
             "Only pcu and nbo share any chemistry (J = 0.36); all other topology pairs have "
             "disjoint chemical sets.  Right panel: coarse key = metal + two linkers; "
             "diagonal set to 1 for reference.",
             fontsize=7.6, color="#555555", ha="left")
    fig.tight_layout()
    p = _f("fig2_confound_structure.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


# ================================================================ Figure 3
def figure3():
    seeds = pd.read_csv(_f("isoreticular_control_seeds.csv"), encoding="utf-8-sig")
    ctrl = pd.read_csv(_f("isoreticular_control.csv"), encoding="utf-8-sig")

    # 测试集目标标准差（同一测试集在两种 regime 下相同）——用来证明为何不用 R²
    sd_map = {}
    for _, r in ctrl.iterrows():
        sd_map[r["test_topology"]] = r["sd_y_test"]

    topos = ["pcu", "nbo"]

    from matplotlib.patches import Patch
    handles = [Patch(color=C_MAIN, label="training set INCLUDES the partner topology"),
               Patch(color=C_ALT, label="training set EXCLUDES the partner topology")]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.05, 1.0]})

    def _bars(ax, groups, mcol_selector, ylab, title, fmt="{:.3f}"):
        """groups: list of (label, filter_mask). 每个组画 with/without 两根柱。"""
        x = np.arange(len(groups))
        w = 0.34
        all_v = []
        for j, reg in enumerate(["with_pair", "without_pair"]):
            mu, sd = [], []
            for _, m in groups:
                s = seeds[m & (seeds["regime"] == reg)][mcol_selector]
                mu.append(s.mean()); sd.append(s.std(ddof=1))
            all_v.extend(mu)
            col = C_MAIN if reg == "with_pair" else C_ALT
            b = ax.bar(x + (j - 0.5) * w, mu, w, yerr=sd, capsize=3, color=col,
                       error_kw={"lw": 1.0})
            for bb, v in zip(b, mu):
                ax.text(bb.get_x() + bb.get_width() / 2,
                        bb.get_height() + max(all_v) * 0.02,
                        fmt.format(v), ha="center", va="bottom", fontsize=7.4)
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(all_v) * 1.34)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    # ---- (a) 归一化误差：MAE / sd(y)  —— 主指标（两边分母对齐）
    groups = [(f"pcu\n(sd={sd_map['pcu']:.3f})", (seeds["test_topo"] == "pcu")),
              (f"nbo\n(sd={sd_map['nbo']:.3f})", (seeds["test_topo"] == "nbo"))]
    _bars(axes[0], groups, "MAE_over_sd", "MAE / sd(y_test)",
          "(a) Normalised error")
    axes[0].legend(handles=handles, fontsize=7.4, loc="upper left", framealpha=0.95)

    # ---- (b) 绝对误差：MAE 与 RMSE 同量级，可共轴
    ax = axes[1]
    labels, vals_w, errs_w, vals_o, errs_o = [], [], [], [], []
    for t in topos:
        for mcol in ["MAE", "RMSE"]:
            a = seeds[(seeds["test_topo"] == t) & (seeds["regime"] == "with_pair")][mcol]
            b = seeds[(seeds["test_topo"] == t) & (seeds["regime"] == "without_pair")][mcol]
            labels.append(f"{t}\n{mcol}")
            vals_w.append(a.mean()); errs_w.append(a.std(ddof=1))
            vals_o.append(b.mean()); errs_o.append(b.std(ddof=1))
    xb = np.arange(len(labels)); w = 0.34
    b1 = ax.bar(xb - w / 2, vals_w, w, yerr=errs_w, capsize=3,
                color=C_MAIN, error_kw={"lw": 1.0})
    b2 = ax.bar(xb + w / 2, vals_o, w, yerr=errs_o, capsize=3,
                color=C_ALT, error_kw={"lw": 1.0})
    for b, v in list(zip(b1, vals_w)) + list(zip(b2, vals_o)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(xb); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("absolute error  (mmol/g)")
    ax.set_ylim(0, max(vals_o) * 1.32)
    ax.set_title("(b) Absolute error", loc="left", fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    # ---- (c) 相对恶化 %
    ax = axes[2]
    rows = []
    for t in topos:
        for mcol in ["MAE", "RMSE"]:
            a = seeds[(seeds["test_topo"] == t) & (seeds["regime"] == "with_pair")][mcol]
            b = seeds[(seeds["test_topo"] == t) & (seeds["regime"] == "without_pair")][mcol]
            rows.append({"topo": t, "metric": mcol,
                         "pct": (b.mean() / a.mean() - 1) * 100,
                         "lo": (b.min() / a.max() - 1) * 100,
                         "hi": (b.max() / a.min() - 1) * 100})
    rel = pd.DataFrame(rows)
    xb = np.arange(len(rel))
    cols = [C_MAIN if r.topo == "pcu" else C_ALT for r in rel.itertuples()]
    bars = ax.bar(xb, rel["pct"], 0.55, color=cols,
                  yerr=[rel["pct"] - rel["lo"], rel["hi"] - rel["pct"]],
                  capsize=4, error_kw={"lw": 1.0})
    ax.axhline(0, color="k", lw=0.9)
    for b, r in zip(bars, rel.itertuples()):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + (6 if r.pct >= 0 else -12),
                f"{r.pct:+.1f}%", ha="center",
                va="bottom" if r.pct >= 0 else "top",
                fontsize=9, fontweight="bold")
    ax.set_xticks(xb)
    ax.set_xticklabels([f"{r.topo}\n{r.metric}" for r in rel.itertuples()], fontsize=9)
    ax.set_ylabel("change when the partner topology is removed (%)")
    ax.set_title("(c) Penalty", loc="left", fontweight="bold", fontsize=10.5)
    ax.set_ylim(-25, 105)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.03,
            "blue = pcu test set\nbrown = nbo test set",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#555555")

    fig.suptitle("Figure 3. Controlled isoreticular comparison (pcu / nbo), 5 seeds",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.008, -0.02,
             "R2 is deliberately NOT used: sd(y) differs 4.4x between the two test sets "
             "(pcu 0.472 vs nbo 0.106), so R2 differences would be an artefact of the denominator.",
             fontsize=7.6, color="#555555", ha="left")
    fig.tight_layout()
    p = _f("fig3_isoreticular_control.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)

    # 顺手把用到的数值打印出来，便于核对稿件
    print("\n核对数值（5 种子 mean±sd）:")
    for t in topos:
        for mcol in ["MAE", "RMSE", "MAE_over_sd"]:
            a = seeds[(seeds["test_topo"] == t) & (seeds["regime"] == "with_pair")][mcol]
            b = seeds[(seeds["test_topo"] == t) & (seeds["regime"] == "without_pair")][mcol]
            print(f"  {t:4s} {mcol:12s} with {a.mean():.4f}±{a.std(ddof=1):.4f} "
                  f"-> without {b.mean():.4f}±{b.std(ddof=1):.4f}  "
                  f"({(b.mean()/a.mean()-1)*100:+.1f}%)")


if __name__ == "__main__":
    figure2()
    figure3()
