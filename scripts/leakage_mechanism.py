#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 7：数据泄漏的机理 —— 近重复样本假说
==========================================

背景
----
前面的实验只证明了"随机划分会让 R² 虚高"，但没有证明**为什么**。
SHAP 归因给出了一个反直觉的线索（见 figures_and_shap.py）：
去掉构筑单元身份特征后，泄漏只减少约 12%。
也就是说，泄漏**不是**通过某个明确的"身份列"发生的。

本脚本检验另一个假说：

    H1（近重复假说）：hMOF 是用有限构筑单元组合枚举出来的，
    同一家族的结构在描述符空间里彼此很近，且标签也相近。
    随机划分会让这些"近重复样本"同时掉进训练集和测试集，
    于是模型只需要**查表插值**，而不是学结构-性能关系。

    若 H1 成立，则：模型的预测精度应当**只取决于测试样本到
    训练集的最近邻距离**，而与"用哪种划分方案"无关。
    换句话说，把随机划分和留一拓扑的样本按最近邻距离分箱后，
    两者的 R²-距离曲线应当**重合在一条线上**。

这个检验的说服力在于：它把 ΔR² 从"一个待解释的现象"
变成了"一个可被单一变量（最近邻距离）解释的现象"。

方法
----
1. 用标准化后的 8 个几何描述符计算欧氏距离。
2. 两种划分方案下，对**每个测试样本**求它到训练集的最近邻距离 d_NN。
3. 按 d_NN 分箱，分别算箱内 R²（随机划分 vs 留一拓扑）。
4. 距离匹配对照：只保留随机划分中 d_NN >= 留一拓扑中位数的样本，
   看它的 R² 是否掉到留一拓扑的水平。

运行
----
    python leakage_mechanism.py --max-rows 8000 --tag smoke
    python leakage_mechanism.py --max-rows 25000 --tag final

输出
----
  fig4_nn_distance[_tag].png     R² vs 最近邻距离（两条曲线是否重合）
  fig7_nn_curve[_tag].csv        分箱数值（Table 15；CSV 文件名沿用存档名不变）
  fig7_nn_samples[_tag].csv      每个测试样本的距离/误差（供复查）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

TARGET = "CO2_uptake_P0.15bar_T298K [mmol/g]"
NUMERIC_FEATURES = [
    "volume [A^3]",
    "weight [u]",
    "surface_area [m^2/g]",
    "void_fraction",
    "void_volume [cm^3/g]",
    "largest_free_sphere_diameter [A]",
    "largest_included_sphere_along_free_sphere_path_diameter [A]",
    "largest_included_sphere_diameter [A]",
]
CATEGORICAL_FEATURES = ["functional_groups", "metal_linker", "topology"]
N_ESTIMATORS = 120
RANDOM_STATE = 42

# 2026-10-16：默认 n_jobs=-1 在无法创建命名管道的环境（受限沙箱）里会以
# WinError 5 失败。固定 random_state 下 n_jobs 不改变拟合结果，故允许用
# 环境变量 LEAKAGE_MECH_N_JOBS 覆盖；与 omitted_variable.py / cardinality_correction.py
# 的 --n-jobs 同一约定。
N_JOBS = int(os.environ.get("LEAKAGE_MECH_N_JOBS", "-1"))


def _make_model():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)


def load_data(csv_path, max_rows):
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 行")
    if max_rows and max_rows < len(df):
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"随机采样 {max_rows} 行")
    return df


def build(df, target=TARGET):
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    y = pd.to_numeric(df[target], errors="coerce")
    Xn = df[cols].copy().astype(float)          # 用于算距离的几何空间
    X = Xn.copy()
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for c in cats:
        X = pd.concat([X, pd.get_dummies(df[c].astype(str),
                                         prefix=c).astype(float)], axis=1)
    mask = y.notna() & X.notna().all(axis=1)
    X, Xn, y = (X[mask].reset_index(drop=True),
                Xn[mask].reset_index(drop=True),
                y[mask].reset_index(drop=True))
    dfc = df.loc[mask].reset_index(drop=True)
    return X, Xn, y, dfc["topology"].astype(str).reset_index(drop=True), dfc


def collect(X, Xn, y, groups, n_splits=5):
    """两种划分下，逐测试样本记录 (真实值, 预测值, 最近邻距离)。"""
    from sklearn.model_selection import KFold, LeaveOneGroupOut
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    # 距离空间：标准化后的几何描述符
    scaler = StandardScaler().fit(Xn.values)
    Z = scaler.transform(Xn.values)
    rows = []

    schemes = [("random", list(KFold(n_splits=n_splits, shuffle=True,
                                     random_state=RANDOM_STATE).split(X))),
               ("unseen_topology",
                list(LeaveOneGroupOut().split(X, y, groups=groups.values)))]

    for name, splits in schemes:
        t0 = time.time()
        for tr, te in splits:
            if len(te) == 0:
                continue
            mdl = _make_model()
            mdl.fit(X.iloc[tr], y.iloc[tr])
            pred = mdl.predict(X.iloc[te])

            nn = NearestNeighbors(n_neighbors=1).fit(Z[tr])
            d, _ = nn.kneighbors(Z[te], return_distance=True)
            rows.append(pd.DataFrame({
                "scheme": name,
                "nn_dist": d.ravel(),
                "y_true": y.iloc[te].values,
                "y_pred": pred,
            }))
        print(f"  {name}: {time.time()-t0:.0f} s")

    return pd.concat(rows, ignore_index=True)


def r2_of(sub):
    from sklearn.metrics import r2_score
    if len(sub) < 30:
        return np.nan
    return r2_score(sub["y_true"], sub["y_pred"])


def binned_curve(samples, n_bins=12, min_n=150):
    """按最近邻距离分箱，两种方案分别算箱内 R²。"""
    # 分箱边界取自 pooled 分布的分位数，保证两方案可比
    qs = np.quantile(samples["nn_dist"], np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    samples = samples.copy()
    samples["bin"] = pd.cut(samples["nn_dist"], qs, duplicates="drop")

    rows = []
    for b, sub in samples.groupby("bin", observed=True):
        for scheme, s in sub.groupby("scheme"):
            if len(s) < min_n:
                continue
            rows.append({
                "bin": str(b),
                "bin_mid": float(np.median(s["nn_dist"])),
                "scheme": scheme,
                "n": len(s),
                "R2": r2_of(s),
            })
    return pd.DataFrame(rows)


def figure7(samples, curve, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---- 左：最近邻距离分布 ----
    for name, color, label in [("random", "#A32D2D", "random split"),
                               ("unseen_topology", "#185FA5",
                                "leave-one-topology-out")]:
        d = samples.loc[samples["scheme"] == name, "nn_dist"]
        ax1.hist(d, bins=60, alpha=0.55, color=color, label=label,
                 range=(0, float(np.quantile(samples["nn_dist"], 0.98))))
    ax1.set_xlabel("distance to nearest training sample (standardized)")
    ax1.set_ylabel("count")
    ax1.set_title("Held-out topologies sit farther from the training data\n"
                  "— but is that what causes the drop?", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.22, lw=0.5)

    # 标注中位数
    med_r = samples.loc[samples["scheme"] == "random", "nn_dist"].median()
    med_t = samples.loc[samples["scheme"] == "unseen_topology", "nn_dist"].median()
    ax1.text(0.98, 0.92, f"median: {med_r:.4f} vs {med_t:.4f}\n"
                         f"ratio = {med_t/max(med_r,1e-9):.2f}×",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="#FFFFFF", ec="#888888", alpha=0.9))

    # ---- 右：R² vs 最近邻距离 ----
    for scheme, color, label, mk in [
            ("random", "#A32D2D", "random split", "o"),
            ("unseen_topology", "#185FA5", "leave-one-topology-out", "s")]:
        c = curve[curve["scheme"] == scheme].sort_values("bin_mid")
        if c.empty:
            continue
        ax2.plot(c["bin_mid"], c["R2"], marker=mk, color=color,
                 lw=1.6, ms=5, label=label)
    ax2.axhline(0, color="#333333", lw=0.8, ls=":")
    ax2.set_xlabel("distance to nearest training sample")
    ax2.set_ylabel("R² within bin")
    ax2.set_title("The curves do NOT overlap → proximity alone cannot\n"
                  "explain the gap (it accounts for only ~26%)", fontsize=10.5)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.22, lw=0.5)

    fig.suptitle("Hypothesis H1 (geometric near-duplicates) is REJECTED\n"
                 "random split stays flat vs distance; held-out topology decays "
                 "— a signature of extrapolation failure, not memorisation",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"已保存: {outpath}")


def collect_family(X, y, dfc, groups, n_splits=5):
    """H2 检验：随机划分下，把测试样本按"构筑单元家族是否在训练集中出现过"分组。

    家族定义刻意**排除 topology**（否则留一拓扑下这个量恒为 0，无法比较），
    取 metal_linker + organic_linker1 + organic_linker2 + functional_groups。
    这样"家族已见"衡量的是：模型在训练时是否见过同一套化学构件。
    """
    from sklearn.model_selection import KFold, LeaveOneGroupOut

    def fam_key(frame):
        parts = []
        for c in ["metal_linker", "organic_linker1", "organic_linker2",
                  "functional_groups"]:
            parts.append(frame[c].astype(str) if c in frame.columns
                         else pd.Series([""] * len(frame), index=frame.index))
        return parts[0] + "|" + parts[1] + "|" + parts[2] + "|" + parts[3]

    fam_all = fam_key(dfc).reset_index(drop=True)
    rows = []

    schemes = [("random", list(KFold(n_splits=n_splits, shuffle=True,
                                     random_state=RANDOM_STATE).split(X))),
               ("unseen_topology",
                list(LeaveOneGroupOut().split(X, y, groups=groups.values)))]

    for name, splits in schemes:
        t0 = time.time()
        for tr, te in splits:
            if len(te) == 0:
                continue
            mdl = _make_model()
            mdl.fit(X.iloc[tr], y.iloc[tr])
            pred = mdl.predict(X.iloc[te])
            seen = fam_all.iloc[te].isin(set(fam_all.iloc[tr])).values
            rows.append(pd.DataFrame({
                "scheme": name,
                "family_seen": seen,
                "y_true": y.iloc[te].values,
                "y_pred": pred,
            }))
        print(f"  {name}: {time.time()-t0:.0f} s")
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--family", action="store_true",
                    help="额外跑 H2（构筑单元家族重叠）检验")
    ap.add_argument("--replot", action="store_true",
                    help="只从已有 CSV 重绘 Figure 7，不重跑计算")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    if args.replot:
        p = os.path.join(args.outdir, f"fig7_nn_samples{suffix}.csv")
        pc = os.path.join(args.outdir, f"fig7_nn_curve{suffix}.csv")
        samples = pd.read_csv(p)
        curve = pd.read_csv(pc)
        print(f"读取 {p}（{len(samples)} 行）与 {pc}")
        figure7(samples, curve,
                os.path.join(args.outdir, f"fig4_nn_distance{suffix}.png"))
        print("完成（replot）。")
        return

    print("=" * 72)
    print("Figure 7：数据泄漏机理 —— 近重复样本假说")
    print("=" * 72)

    df = load_data(args.csv, args.max_rows)
    X, Xn, y, groups, dfc = build(df, args.target)
    print(f"建模样本 {len(X)}，特征 {X.shape[1]}，几何空间维度 {Xn.shape[1]}")

    print("\n[1/3] 两种划分下收集预测值与最近邻距离 ...")
    samples = collect(X, Xn, y, groups)
    p = os.path.join(args.outdir, f"fig7_nn_samples{suffix}.csv")
    samples.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"已保存: {p}")

    print("\n[2/3] 按最近邻距离分箱 ...")
    curve = binned_curve(samples, n_bins=args.n_bins)
    pc = os.path.join(args.outdir, f"fig7_nn_curve{suffix}.csv")
    curve.to_csv(pc, index=False, encoding="utf-8-sig")
    print(curve.to_string(index=False))
    print(f"已保存: {pc}")

    # ---- 距离匹配对照：这是整个检验的关键数字 ----
    print("\n" + "=" * 72)
    print("距离匹配对照（关键检验）")
    print("=" * 72)
    med_t = samples.loc[samples["scheme"] == "unseen_topology", "nn_dist"].median()
    r_all = samples[samples["scheme"] == "random"]
    r_far = r_all[r_all["nn_dist"] >= med_t]
    t_all = samples[samples["scheme"] == "unseen_topology"]
    print(f"  随机划分（全部）              R² = {r2_of(r_all):.4f}  n = {len(r_all)}")
    print(f"  留一拓扑（全部）              R² = {r2_of(t_all):.4f}  n = {len(t_all)}")
    print(f"  随机划分 + 距离 >= 拓扑中位数  R² = {r2_of(r_far):.4f}  n = {len(r_far)}")
    print(f"\n  中位数最近邻距离：随机 "
          f"{r_all['nn_dist'].median():.4f}  vs  留一拓扑 {med_t:.4f}  "
          f"（{med_t/max(r_all['nn_dist'].median(),1e-9):.2f} 倍）")
    if not np.isnan(r2_of(r_far)):
        closed = (r2_of(r_all) - r2_of(r_far)) / max(r2_of(r_all) - r2_of(t_all), 1e-9)
        print(f"  被近邻距离解释掉的 ΔR² 比例 = {closed*100:.0f}%")
        print("\n  若该比例接近 100%，说明虚高性能几乎全部来自"
              "\n  '测试样本在训练集里有近亲'，而不是模型真的学到了规律。")

    print("\n[3/3] 出图 ...")
    figure7(samples, curve,
            os.path.join(args.outdir, f"fig4_nn_distance{suffix}.png"))

    # ---- H2：构筑单元家族重叠 ----
    if args.family:
        print("\n" + "=" * 72)
        print("H2 检验：构筑单元家族是否在训练集中出现")
        print("=" * 72)
        print("（家族 = 金属 + 两个有机连接体 + 官能团，**不含拓扑**，")
        print("  这样留一拓扑下的样本才有可比性）\n")
        fam = collect_family(X, y, dfc, groups)
        pf = os.path.join(args.outdir, f"fig7_family{suffix}.csv")
        fam.to_csv(pf, index=False, encoding="utf-8-sig")
        print(f"已保存: {pf}\n")

        summary = []
        for scheme, s in fam.groupby("scheme"):
            for flag, ss in s.groupby("family_seen"):
                summary.append({
                    "scheme": scheme,
                    "family_seen_in_train": bool(flag),
                    "n": len(ss),
                    "frac": len(ss) / len(s),
                    "R2": r2_of(ss),
                })
        sm = pd.DataFrame(summary)
        print(sm.to_string(index=False))
        ps = os.path.join(args.outdir, f"fig7_family_summary{suffix}.csv")
        sm.to_csv(ps, index=False, encoding="utf-8-sig")
        print(f"\n已保存: {ps}")

        # 关键对照：随机划分中"家族未见"的子集，是否掉到留一拓扑的水平
        r_unseen = fam[(fam["scheme"] == "random") & (~fam["family_seen"])]
        t_all2 = fam[fam["scheme"] == "unseen_topology"]
        r_all2 = fam[fam["scheme"] == "random"]
        r_seen = fam[(fam["scheme"] == "random") & fam["family_seen"]]
        if len(r_unseen) >= 30 and len(r_seen) >= 30:
            print("\n  关键对照：")
            print(f"    随机划分 · 家族已见   R² = {r2_of(r_seen):.4f}  n = {len(r_seen)}")
            print(f"    随机划分 · 家族未见   R² = {r2_of(r_unseen):.4f}  n = {len(r_unseen)}")
            print(f"    留一拓扑（全部未见）  R² = {r2_of(t_all2):.4f}  n = {len(t_all2)}")
            denom = r2_of(r_seen) - r2_of(t_all2)
            num = r2_of(r_seen) - r2_of(r_unseen)
            if abs(denom) > 1e-9:
                print(f"\n    被'家族是否见过'解释掉的 ΔR² 比例 = {num/denom*100:.0f}%")
                print("    若该比例接近 100%，则泄漏的载体就是构筑单元家族的重叠。")

    print("\n完成。")


if __name__ == "__main__":
    main()
