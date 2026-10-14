#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quasi_independent_targets.py —— P3-1: 十目标面板的准独立目标族重算

回应 R2-M6 / S-B5：§3.13 声称 "10/10 targets, median +0.324"，
但十个目标中 uptake / excess uptake / 两个 working capacity / selectivity
互为代数函数，真正准独立的远少于 10。

本脚本：
  1. 读取 outputs/multitarget_confound.csv
  2. 计算目标间的 Spearman 相关矩阵
  3. 识别代数等价组（|ρ| > 0.999）
  4. 按准独立目标族重算 "positive fraction" 与 Wilcoxon p 值
  5. 输出 multitarget_confound_quasi_independent.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(OUT, "multitarget_confound.csv"))
    ap.add_argument("--out", default=os.path.join(OUT, "multitarget_confound_quasi_independent.csv"))
    ap.add_argument("--thr", type=float, default=0.999,
                    help="Spearman |ρ| 超过此值视为代数等价")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    # 只保留有完整 delta_R2 的行
    d = d.dropna(subset=["delta_R2"]).reset_index(drop=True)
    targets = d["target"].tolist()

    # 相关矩阵
    rho_mat = pd.DataFrame(np.eye(len(targets)), index=targets, columns=targets)
    for i, ti in enumerate(targets):
        for j, tj in enumerate(targets):
            if i < j:
                r, _ = spearmanr(d[d["target"] == ti]["delta_R2"].values,
                                 d[d["target"] == tj]["delta_R2"].values)
                rho_mat.loc[ti, tj] = r
                rho_mat.loc[tj, ti] = r

    # 等价组（贪心合并）
    equiv = {}
    group_id = 0
    for i, ti in enumerate(targets):
        if ti in equiv:
            continue
        grp = [ti]
        for j, tj in enumerate(targets):
            if j <= i or tj in equiv:
                continue
            if abs(rho_mat.loc[ti, tj]) > args.thr:
                grp.append(tj)
        for t in grp:
            equiv[t] = group_id
        group_id += 1

    d["quasi_independent_group"] = d["target"].map(equiv)
    n_groups = d["quasi_independent_group"].nunique()

    # 每族的 Wilcoxon 检验（H0: median delta_R2 <= 0）
    rows = []
    for gid, g in d.groupby("quasi_independent_group"):
        pos = int((g["delta_R2"] > 0).sum())
        tot = len(g)
        median = g["delta_R2"].median()
        if tot >= 3:
            try:
                _, p = wilcoxon(g["delta_R2"], alternative="greater")
            except Exception:
                p = np.nan
        else:
            p = np.nan
        rows.append({
            "quasi_independent_group": gid,
            "targets": ", ".join(g["target"].tolist()),
            "n_targets": tot,
            "n_positive": pos,
            "fraction_positive": pos / tot if tot else np.nan,
            "median_delta_R2": median,
            "wilcoxon_p_greater": p,
        })

    out_rows = pd.DataFrame(rows)
    out_rows.to_csv(args.out, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print("准独立目标族重算")
    print("=" * 72)
    print(f"原始目标数 : {len(targets)}")
    print(f"准独立族数 : {n_groups}")
    print()
    print(out_rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 全局统计：按族数而非目标数
    global_pos = int((d.groupby("quasi_independent_group")["delta_R2"].median() > 0).sum())
    global_median = d.groupby("quasi_independent_group")["delta_R2"].median().median()
    print()
    print(f"按族计：{global_pos}/{n_groups} 族 median > 0，跨族 median = {global_median:+.4f}")

    print(f"\n已保存: {args.out}")


if __name__ == "__main__":
    import argparse  # noqa: E402
    main()
