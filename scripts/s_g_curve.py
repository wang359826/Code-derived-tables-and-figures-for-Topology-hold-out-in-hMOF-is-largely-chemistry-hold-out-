#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S_g 曲线：把「已见化学率 → 留出代价」从单点升级为连续轴
（第二轮审查 O5 末段 / 原 SI §S4-3 的待办）

之前的证据是两个端点：
  · Table 9  给出每个拓扑被留出时"化学在训练集中无对应"的**结构占比**（纯计数）；
  · Table 13 给出 100,000 样本上每个拓扑留出的**池化 R²**（训练模型）；
但两者从未画在一起，也从未逐拓扑配对。

本脚本在**同一个 100,000 抽样、同一个模型、同一种折法**下同时产出：
  S_g(T)      留出拓扑 T 时，其测试样本中"化学在训练折里有对应"的比例（纯计数）
  S_miss(T)   = 1 − S_g(T)，"化学未见面"比例
  r          = n/M，该折样本/化学组成比（粒度自校准的混淆强度）
  R²_pooled  该折的池化 R²
  然后给出 S_miss → R² 的相关与回归，**同时给出合并奥肯定律修正**
  （Table 12 已证明留一拓扑的池化 R² 被最大组 pcu 主导，所以另给
   去掉 pcu 后重算的版本）。

⚠️ 纪律：R² 只在**同一支撑集**内可比。本脚本 report 的逐折 R² 是不同
   测试集上的值，**不构成可比性主张**；这里的因变量是"每个拓扑各自的
   留出代价"，用法与 §3.13 的目标间面板同类（描述性敏感性分析）。

用法
----
  python s_g_curve.py --csv data/all_MOFs_screening_data.csv
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

TARGET = "CO2_uptake_P0.15bar_T298K [mmol/g]"
NUMERIC_FEATURES = [
    "volume [A^3]", "weight [u]", "surface_area [m^2/g]", "void_fraction",
    "void_volume [cm^3/g]", "largest_free_sphere_diameter [A]",
    "largest_included_sphere_along_free_sphere_path_diameter [A]",
    "largest_included_sphere_diameter [A]",
]
CATS = ["functional_groups", "metal_linker", "topology"]
H_COARSE = ["metal_linker", "organic_linker1", "organic_linker2"]
N_EST = 100           # 与正文十目标面板一致（§2.6）
SEED = 42
MIN_TEST = 100        # 测试折小于此值时 R² 不可靠，标记但不删除


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        HERE, "..", "data", "all_MOFs_screening_data.csv"))
    ap.add_argument("--max-rows", type=int, default=100000)
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    df = pd.read_csv(args.csv)
    if args.max_rows and args.max_rows < len(df):
        df = df.sample(args.max_rows, random_state=SEED).reset_index(drop=True)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X = df[NUMERIC_FEATURES].copy().astype(float)
    for c in CATS:
        X = pd.concat([X, pd.get_dummies(df[c].astype(str),
                                         prefix=c).astype(float)], axis=1)
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    d = df.loc[mask].reset_index(drop=True)
    print(f"建模样本 {len(X):,}  特征 {X.shape[1]}")

    topo = d["topology"].astype(str)
    # 组成键与"每个组成横跨几个拓扑"——用 numpy 编码，避免 groupby.apply（在 10 万行上极慢）
    ck_vals = (d[H_COARSE].fillna("NA").astype(str)
               .agg("|".join, axis=1).to_numpy())
    ck_code, ck_inv = np.unique(ck_vals, return_inverse=True)
    tp_code, tp_inv = np.unique(topo.to_numpy(), return_inverse=True)
    n_tp, n_ck = len(tp_code), len(ck_code)
    pair = ck_inv.astype(np.int64) * n_tp + tp_inv.astype(np.int64)
    pair = np.unique(pair)
    k_per_comp = np.zeros(n_ck, dtype=np.int64)
    np.add.at(k_per_comp, pair // n_tp, 1)
    is_multi = k_per_comp[ck_inv] > 1          # 逐行：其组成是否横跨 >1 个拓扑
    n_comp_per_row = k_per_comp[ck_inv]
    print(f"共 {n_ck:,} 个组成，其中 {int((k_per_comp > 1).sum()):,} 个横跨 >1 个拓扑")

    rows = []
    for t in topo.value_counts().index:
        te = np.where(topo.to_numpy() == t)[0]
        tr = np.where(topo.to_numpy() != t)[0]
        if len(te) == 0:
            continue
        sg = float(is_multi[te].mean())
        m = RandomForestRegressor(n_estimators=N_EST, random_state=SEED,
                                  n_jobs=1)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        r2 = float(r2_score(y.iloc[te], p))
        rows.append({
            "held_out_topology": t,
            "n_train": len(tr), "n_test": len(te),
            "n_compositions_test": int(np.unique(ck_inv[te]).size),
            "S_g_seen": sg, "S_miss_unseen": 1 - sg,
            "r_train": len(tr) / max(1, np.unique(ck_inv[tr]).size),
            "sd_y_test": float(y.iloc[te].std()),
            "R2_pooled": r2,
        })
        print(f"  {t:5s} n_test={len(te):7,d}  S_miss={1-sg:.3f}  "
              f"r={rows[-1]['r_train']:.2f}  R²={r2:+.3f}", flush=True)

    out = pd.DataFrame(rows).sort_values("S_miss_unseen")
    p1 = os.path.join(args.outdir, "s_g_curve.csv")
    out.to_csv(p1, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p1}")

    # ---- 相关：S_miss → R²（全部拓扑 / 去掉最小测试折 / 去掉 pcu）----
    from scipy.stats import pearsonr, spearmanr
    def corr(sub, name):
        if len(sub) < 3:
            return
        x, yv = sub["S_miss_unseen"].to_numpy(float), sub["R2_pooled"].to_numpy(float)
        if np.std(x) == 0 or np.std(yv) == 0:
            return
        pr, pp = pearsonr(x, yv)
        sr, sp = spearmanr(x, yv)
        print(f"  {name:34s} n={len(sub):2d}  Pearson r={pr:+.3f} (p={pp:.3f})"
              f"  Spearman ρ={sr:+.3f} (p={sp:.3f})")
    print("\n=== S_miss vs pooled R² (descriptive; different support sets) ===")
    corr(out, "all topologies")
    corr(out[out["n_test"] >= MIN_TEST], f"n_test >= {MIN_TEST}")
    corr(out[out["held_out_topology"] != "pcu"], "excluding pcu")
    corr(out[(out["n_test"] >= MIN_TEST) &
            (out["held_out_topology"] != "pcu")], f"n_test>={MIN_TEST} & no pcu")


if __name__ == "__main__":
    main()
