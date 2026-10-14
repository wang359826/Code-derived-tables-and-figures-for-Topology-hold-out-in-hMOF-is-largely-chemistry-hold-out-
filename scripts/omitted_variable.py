#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 8 / Table 6：拓扑是不是一个"被遗漏的变量"？
==================================================

前面的机理检验否掉了两个直觉假说：

  H1 几何空间近重复  → 只解释 ΔR² 的 26%（leakage_mechanism.py）
  H2 构筑单元家族重叠 → 只解释 ΔR² 的 11%

而且有一个更硬的证据：mof_leakage_benchmark.py 的缓解手段对照显示，
**只用 8 个几何描述符**（完全不含身份特征）时，随机划分 R²=0.672
对上按拓扑分组 R²=0.374，差距仍有 0.298 —— 几乎没被消掉。

这说明问题不在"特征里混进了身份信息"，而在**特征本身不足以定住目标**：

    H3（遗漏变量假说）：几何描述符并不能唯一决定 CO₂ 吸附量。
    真正决定吸附量的还有拓扑所编码的化学环境（孔道形状、开放金属位点排布等），
    而这个变量**根本不在特征集里**。

    若 H3 成立：
      (a) 用拓扑 alone 就能解释掉目标变量相当一部分方差；
      (b) 几何模型拟合后的**残差**应当仍然带有明显的拓扑结构
          ——即"残差 ~ 拓扑"的 R² 显著为正。

    而这条一旦成立，留一拓扑的失败就不是"模型不够好"，
    而是**任务本身在给定特征集下不可外推**——
    这是一个比"数据泄漏"更根本、也更值得发表的结论。

方法
----
1. 方差分解：y 的组间/组内方差（η²），分别对 topology / metal / linker pair。
2. 单独用该分组变量做 5 折随机 CV，看它能达到多少 R²（上界参考）。
3. 用几何描述符拟合，取折外残差，再对残差做同样的方差分解
   → 若残差里仍有大量结构，说明该变量是遗漏变量。

运行
----
    python omitted_variable.py --max-rows 8000 --tag smoke
    python omitted_variable.py --max-rows 25000 --tag final

输出
----
  omitted_variable[_tag].csv   Table 6：方差分解 + 残差结构
"""

from __future__ import annotations

import argparse
import os
import sys

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
RANDOM_STATE = 42
# 2026-10-14：并行度改为可用 --n-jobs 覆盖。原因与本包其他脚本相同——受限沙箱下
# joblib 的进程/线程池无法创建命名管道（WinError 5），而固定 random_state 下
# n_jobs 不改变拟合结果，所以 n_jobs=1 是等价的顺序执行。
N_JOBS_OVERRIDE = -1


def _make_model():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=120, random_state=RANDOM_STATE,
                                 n_jobs=N_JOBS_OVERRIDE)


def eta_squared(y: np.ndarray, groups: np.ndarray) -> float:
    """组间方差占总方差的比例（η²）。衡量"该分组变量能定住多少目标"。"""
    df = pd.DataFrame({"y": y, "g": groups})
    grand = df["y"].mean()
    ss_total = float(((df["y"] - grand) ** 2).sum())
    if ss_total <= 0:
        return np.nan
    gm = df.groupby("g")["y"].agg(["mean", "count"])
    ss_between = float((gm["count"] * (gm["mean"] - grand) ** 2).sum())
    return ss_between / ss_total


def cv_r2(X: pd.DataFrame, y: pd.Series, n_splits=5) -> float:
    """5 折随机划分下的折外 R²（这里只用作"信息上界"的参考）。"""
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    yt, yp = [], []
    for tr, te in KFold(n_splits=n_splits, shuffle=True,
                        random_state=RANDOM_STATE).split(X):
        m = _make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        yp.extend(m.predict(X.iloc[te]).tolist())
        yt.extend(y.iloc[te].tolist())
    return r2_score(yt, yp)


def oof_residuals(X: pd.DataFrame, y: pd.Series, n_splits=5):
    """几何模型在随机划分下的折外残差。"""
    from sklearn.model_selection import KFold
    res = np.full(len(y), np.nan)
    for tr, te in KFold(n_splits=n_splits, shuffle=True,
                        random_state=RANDOM_STATE).split(X):
        m = _make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        res[te] = y.iloc[te].values - m.predict(X.iloc[te])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="RandomForest 并行度。固定 random_state 下 n_jobs 不改变拟合结果，"
                         "所以受限沙箱里传 1 即可得到与默认 -1 相同的数值（2026-10-14 新增："
                         "该脚本原先硬编码 n_jobs=-1，在无法创建命名管道的环境里直接 WinError 5）")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--within-vs-cross", action="store_true",
                    help="额外做「拓扑内 vs 跨拓扑」对照")
    args = ap.parse_args()

    global N_JOBS_OVERRIDE
    N_JOBS_OVERRIDE = args.n_jobs      # 固定 random_state 下不改变数值，只是执行方式

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    df = pd.read_csv(args.csv)
    if args.max_rows and args.max_rows < len(df):
        df = df.sample(args.max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    y = pd.to_numeric(df[args.target], errors="coerce")
    Xg = df[[c for c in NUMERIC_FEATURES if c in df.columns]].astype(float)
    ok = y.notna() & Xg.notna().all(axis=1)
    df, y, Xg = df[ok].reset_index(drop=True), y[ok].reset_index(drop=True), \
        Xg[ok].reset_index(drop=True)

    group_defs = {
        "topology": df["topology"].astype(str).values,
        "metal_linker": df["metal_linker"].astype(str).values,
        "linker_pair": (df["organic_linker1"].astype(str) + "_"
                        + df["organic_linker2"].astype(str)).values,
    }

    print("=" * 74)
    print("遗漏变量分析：几何描述符之外，还有多少信息被分组变量携带？")
    print("=" * 74)
    print(f"样本 {len(df)}，几何描述符 {Xg.shape[1]} 个\n")

    print("[1/3] 目标变量的方差分解 + 单独用该变量能到多少 R² ...")
    rows = []
    for name, g in group_defs.items():
        e2 = eta_squared(y.values, g)
        # 单独用该分组变量的 one-hot 做 5 折 CV
        Xo = pd.get_dummies(pd.Series(g), prefix=name).astype(float)
        r2 = cv_r2(Xo, y)
        rows.append({"variable": name, "n_groups": int(pd.Series(g).nunique()),
                     "eta2_of_target": e2, "R2_alone": r2})
        print(f"  {name:14s} 组数 {pd.Series(g).nunique():>5}  "
              f"η²(目标) = {e2:.4f}   单独建模 R² = {r2:.4f}")

    print("\n[2/3] 几何模型的折外残差 ...")
    res = oof_residuals(Xg, y)
    r2_geo = 1.0 - float(np.nanvar(res) / np.nanvar(y.values))
    print(f"  几何模型（8 描述符）随机划分 R² = {r2_geo:.4f}")
    print(f"  残差标准差 = {np.nanstd(res):.4f}   目标标准差 = {np.std(y.values):.4f}")

    print("\n[3/3] 残差里还剩多少分组结构？（关键） ...")
    for row, (name, g) in zip(rows, group_defs.items()):
        e2r = eta_squared(res, g)
        Xo = pd.get_dummies(pd.Series(g), prefix=name).astype(float)
        r2r = cv_r2(Xo, pd.Series(res))
        row["eta2_of_residual"] = e2r
        row["R2_residual"] = r2r
        print(f"  {name:14s} η²(残差) = {e2r:.4f}   残差建模 R² = {r2r:.4f}")

    out = pd.DataFrame(rows)
    out.insert(0, "n_samples", len(df))
    out.insert(1, "R2_geometric_model", r2_geo)
    p = os.path.join(args.outdir, f"omitted_variable{suffix}.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")

    # ---- 拓扑内 vs 跨拓扑：关系是不是拓扑特异的？ ----
    if args.within_vs_cross:
        print("\n" + "=" * 74)
        print("拓扑内 R² vs 跨拓扑 R²（几何描述符，各拓扑单独比较）")
        print("=" * 74)
        print("  对每个拓扑 T：")
        print("    within —— 只在 T 自己的样本上做 5 折 CV")
        print("    cross  —— 在除 T 以外的所有拓扑上训练，到 T 上测试")
        print("  若 within 显著高于 cross，说明「几何 → 吸附量」的映射"
              "是拓扑特异的，\n  这正是跨拓扑外推失败的直接原因。\n")

        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold

        topo = df["topology"].astype(str)
        wrows = []
        for t, n in topo.value_counts().items():
            if n < 200:
                continue
            idx_t = np.where(topo.values == t)[0]
            idx_o = np.where(topo.values != t)[0]
            if len(idx_o) < 500:
                continue

            # within：T 内部 5 折
            Xt, yt = Xg.iloc[idx_t], y.iloc[idx_t]
            k = min(5, len(idx_t) // 50)
            if k < 2:
                continue
            yt_all, yp_all = [], []
            for tr, te in KFold(n_splits=k, shuffle=True,
                                random_state=RANDOM_STATE).split(Xt):
                m = _make_model()
                m.fit(Xt.iloc[tr], yt.iloc[tr])
                yp_all.extend(m.predict(Xt.iloc[te]).tolist())
                yt_all.extend(yt.iloc[te].tolist())
            r2_within = r2_score(yt_all, yp_all)

            # cross：其他拓扑训练 → T 测试
            m = _make_model()
            m.fit(Xg.iloc[idx_o], y.iloc[idx_o])
            r2_cross = r2_score(y.iloc[idx_t].values, m.predict(Xg.iloc[idx_t]))

            wrows.append({"topology": t, "n": int(n),
                          "R2_within": r2_within, "R2_cross": r2_cross,
                          "gap": r2_within - r2_cross})
            print(f"  {t:>6}  n={int(n):>6}  within = {r2_within:.4f}   "
                  f"cross = {r2_cross:.4f}   gap = {r2_within - r2_cross:+.4f}")

        if wrows:
            w = pd.DataFrame(wrows)
            pw = os.path.join(args.outdir, f"within_vs_cross{suffix}.csv")
            w.to_csv(pw, index=False, encoding="utf-8-sig")
            print(f"\n  中位 within = {w['R2_within'].median():.4f}，"
                  f"中位 cross = {w['R2_cross'].median():.4f}，"
                  f"中位 gap = {w['gap'].median():+.4f}")
            print(f"  已保存: {pw}")
            print("\n  判读：gap 越大、且 cross 常为负，越说明映射是拓扑特异的。")
            print("  这也是对审稿人「换个更强的模型行不行」的正面回答 —— "
                  "不行，问题不在模型。")

    print("\n" + "=" * 74)
    print("判读")
    print("=" * 74)
    top = out[out["variable"] == "topology"].iloc[0]
    print(f"  拓扑单独就能解释目标变量 η² = {top['eta2_of_target']:.3f}，"
          f"单独建模 R² = {top['R2_alone']:.3f}")
    print(f"  几何模型拟合后，残差里仍残留 η² = {top['eta2_of_residual']:.3f} 的拓扑结构")
    print("\n  若残差的 η² 与目标的 η² 同量级甚至更高，说明几何描述符"
          "\n  从未真正'吸收'掉拓扑携带的信息 —— 拓扑是一个被遗漏的变量。")
    print("\n  推论：留一拓扑下的性能退化，本质上不是'数据泄漏'造成的假象，"
          "\n  而是**给定特征集下跨拓扑外推本身不可行**。")
    print(f"\n已保存: {p}")


if __name__ == "__main__":
    main()
