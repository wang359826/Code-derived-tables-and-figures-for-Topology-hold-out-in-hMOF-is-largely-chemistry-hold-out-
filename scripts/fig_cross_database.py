# -*- coding: utf-8 -*-
"""
Figure 8：跨数据库复现 —— 化学–拓扑混淆不是 hMOF 独有。
Figure S4：聚合口径的稳定性 —— pooled 的"稳"是假象。

（2026-10-11 图序修订后，这两张图由原 Figure 12 / 13 变为 Figure 8 / Figure S4。）

Figure 8 的数据来自三处：
  outputs/confound_arcmof.csv        （ARC-MOF 混淆标量）
  outputs/sample_size_granularity.csv（同 n 同粒度类型的直接对照，取代已作废的随机压缩对齐）
  outputs/confound_by_source_arcmof.csv（按来源分层）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5,
                     "axes.labelsize": 9.5, "figure.dpi": 150})
C_H = "#2F5D8C"      # hMOF
C_A = "#B4543A"      # ARC-MOF
C_E = "#3E7A5E"      # 枚举来源
C_M = "#8A6BBE"      # 混合/实验来源


def f(n):
    return os.path.join(OUT, n)


# ============================================================ Figure 8
def figure8():
    cf = pd.read_csv(f("confound_arcmof.csv"), encoding="utf-8-sig")
    sg = pd.read_csv(f("sample_size_granularity.csv"), encoding="utf-8-sig")
    src = pd.read_csv(f("source_confound_granularity.csv"), encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5),
                             gridspec_kw={"width_ratios": [0.85, 1.15, 1.0]})

    # ---- (a) 两库都混淆（hMOF 用**全库**口径）
    ax = axes[0]
    h_cf = pd.read_csv(f("confound_full.csv"), encoding="utf-8-sig")
    h_fine = float(h_cf.loc[h_cf["granularity"].astype(str).str.contains("fine"),
                            "frac_single_topology"].iloc[0])
    h_coarse = float(h_cf.loc[h_cf["granularity"].astype(str).str.contains("coarse"),
                              "frac_single_topology"].iloc[0])
    a_sub = cf[(cf["topo_col"] == "Crystalnet") & (cf["subset"] == "all")]
    a_fine = float(a_sub[a_sub["granularity"].astype(str).str.contains("fine")]
                   ["frac_single_topology"].iloc[0])
    a_coarse = float(a_sub[a_sub["granularity"].astype(str).str.contains("coarse")]
                     ["frac_single_topology"].iloc[0])
    n_a = int(a_sub["n_samples"].iloc[0]) if "n_samples" in a_sub.columns else 264225
    hv = [h_coarse * 100, h_fine * 100]
    av = [a_coarse * 100, a_fine * 100]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2, hv, w, color=C_H, label="hMOF (full, 324,426)")
    b2 = ax.bar(x + w / 2, av, w, color=C_A, label="ARC-MOF ({:,})".format(n_a))
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(100, color="k", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["coarse key", "fine key"])
    ax.set_ylabel("chemistry mapped to ONE topology (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(a) Both databases are confounded", loc="left",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7.4, loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    # ---- (b) 样本量依赖 + 同 n 同粒度类型的直接对照（不做随机重组）
    ax = axes[1]
    for db, col in [("hMOF", C_H), ("ARC-MOF", C_A)]:
        g = sg[(sg["database"] == db)
               & (sg["granularity"].astype(str).str.contains("fine"))].copy()
        g["n"] = g["n_samples"].astype(float)
        g = g.sort_values("n")
        ax.plot(g["n"], g["frac_single_topology"] * 100, "o-",
                color=col, lw=1.8, ms=4.5, label=db + " (fine)")
    ax.set_xscale("log")
    ax.set_xlabel("sample size n (log)")
    ax.set_ylabel("single-topology chemistry (%)")
    ax.set_title("(b) ...but the number drifts with n", loc="left",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.annotate("no random re-compression:/nsame n, same granularity type",
                xy=(0.03, 0.05), xycoords="axes fraction",
                fontsize=7.4, color="#333333", va="bottom")
    ax.set_ylim(88, 100.4)

    # ---- (c) 来源分层（两级真实粒度 + bootstrap 误差棒）
    ax = axes[2]
    s = src.sort_values("fine_frac_single")
    xs = np.arange(len(s)); w = 0.38
    vals_c = [float(v) * 100 for v in s["coarse_frac_single"]]
    vals_f = [float(v) * 100 for v in s["fine_frac_single"]]
    ax.bar(xs - w / 2, vals_c, w, color="#B0A6C8", label="coarse")
    ax.bar(xs + w / 2, vals_f, w, color=C_A, label="fine")
    if "boot_sd" in s.columns:
        sd = [0.0 if pd.isna(v) else float(v) * 100 for v in s["boot_sd"]]
        ax.errorbar(xs + w / 2, vals_f, yerr=sd, fmt="none",
                    ecolor="#333333", elinewidth=0.9, capsize=2.5)
    ax.axhline(h_fine * 100, color=C_H, ls="--", lw=1.4)
    ax.text(len(s) - 0.4, h_fine * 100 + 0.7, "hMOF (fine, full)", color=C_H,
            fontsize=7.4, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r.source}\n(n={int(r.n_samples):,})" for r in s.itertuples()],
                       fontsize=7.2)
    ax.set_ylim(70, 112)
    ax.set_ylabel("single-topology chemistry (%)")
    ax.set_title("(c) confound tracks how each subset was built", loc="left",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7.6)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    fig.suptitle("Figure 8. The confound replicates in a second database (ARC-MOF)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.text(0.008, -0.02,
             "ARC-MOF is NOT independent of hMOF: it spans all public MOF databases and "
             "shares authors with hMOF (Result 0); leave-one-source-out bounds the overlap "
             "at 98.6-99.8%. Panel (c): sub-libraries are ordered by degree only - the "
             "official DB-to-source mapping is unavailable, so no "
             "'enumerated vs experimental' claim is made.",
             fontsize=7.2, color="#555555", ha="left")
    fig.tight_layout()
    p = f("fig8_cross_database.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)

    # 打印核对
    print(f"  (a) hMOF coarse {h_coarse*100:.2f}%  fine {h_fine*100:.2f}%")
    print(f"      ARC  coarse {a_coarse*100:.2f}%  fine {a_fine*100:.2f}%")


# ============================================================ Figure S4
def figure_s4():
    d = pd.read_csv(f("multi_seed_uptake25k_fix.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ---- (a) 每个种子的三种口径
    ax = axes[0]
    seeds = d["seed"].astype(str).tolist()
    x = np.arange(len(seeds)); w = 0.26
    series = [("loto_pooled", "pooled (all folds concatenated)", C_H),
              ("loto_median", "median over folds", C_E),
              ("loto_macro", "macro (mean over folds)", C_A)]
    for i, (col, lab, c) in enumerate(series):
        ax.bar(x + (i - 1) * w, d[col], w, color=c, label=lab)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels([f"seed\n{s}" for s in seeds], fontsize=8)
    ax.set_ylabel("leave-one-topology-out  R²")
    ax.set_ylim(-1.9, 0.65)
    ax.set_title("(a) Same run, three aggregation rules", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    # ---- (b) 稳定性对比：sd / |mean|
    ax = axes[1]
    lab, val, col = [], [], []
    for colname, name, c in series:
        v = d[colname]
        lab.append(name.split(" (")[0])
        val.append(v.std(ddof=1) / abs(v.mean()) * 100)
        col.append(c)
    bars = ax.bar(range(len(lab)), val, 0.5, color=col)
    for b, v in zip(bars, val):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
    ax.set_ylabel("across-seed dispersion  (sd / |mean|, %)")
    ax.set_title("(b) 'Stable' pooled R² is an artefact", loc="left",
                 fontweight="bold", fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    fig.suptitle("Figure S4. Pooled R² looks stable only because one topology dominates",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.text(0.008, -0.02,
             "5 seeds × 25,000 samples.  One fold had a degenerate (zero-variance) test set "
             "in 3 of 5 seeds and is excluded from the macro/median; 10–11 of 11 folds remain.",
             fontsize=7.4, color="#555555", ha="left")
    fig.tight_layout()
    p = f("figS4_aggregation_stability.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)
    for col, name, _ in series:
        v = d[col]
        print(f"  {name:36s} {v.mean():+.4f} ± {v.std(ddof=1):.4f}   "
              f"sd/|mean| = {v.std(ddof=1)/abs(v.mean())*100:.1f}%")


if __name__ == "__main__":
    figure8()
    figure_s4()
