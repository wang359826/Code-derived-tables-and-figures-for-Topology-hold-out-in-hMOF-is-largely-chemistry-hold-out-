#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
derive_multitarget_stats.py —— §3.13 的派生统计量（相关系数与其 p 值）。

为什么单独有这个脚本（R1-W10）：
    主稿 §3.13 引用了若干派生统计（+0.975 / +0.976、−0.910、−0.860 及各自的
    p 值），但先前没有任何脚本产出它们——`multitarget_confound.py` 只写逐目标
    的 R² 表，`fig_multitarget.py` 只把相关系数画进图题。这与主稿"每个数字都由
    `scripts/run_all.py` 重新生成"的声明相抵触。本脚本补上这一环，并把口径
    写死：**两个统计量必须取自同一列对**。

口径（2026-10-11 定）：
    主稿 §3.13 item 3 与 item 4 的论据都是"与拓扑留出同一划分下的描述符 R²"，
    因此统一使用 `R2_topology_geo` 列（拓扑划分 + 仅结构描述符），而不是
    `R2_random_geo`（随机划分 + 仅结构描述符）。先前版本把 item 4 的 Pearson
    取自前者、Spearman 取自后者，二者不可互相复算。

输入：outputs/multitarget_confound.csv（由 multitarget_confound.py 生成）
输出：outputs/multitarget_derived_stats.csv，并把结果打印到 stdout。
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

# ---- p 值：优先用 scipy（与主稿其余部分一致）；缺失时用 t 分布近似回退 ----
try:
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None


def _betacf(a, b, x, itmax=200, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def p_two_sided_t(r, n):
    """相关系数的双侧 p 值（t 近似，df = n - 2）——与 scipy 对 Pearson/Spearman
    采用的近似一致。scipy 可用时不用本函数。"""
    if n < 3 or abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t = r * math.sqrt(df / (1.0 - r * r))
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def corr_block(df, xcol, ycol, tag, log_target=True):
    """log_target=True 时剔除 log 目标（selectivity），与主稿 'nine targets' 表
    的 Spearman 口径保持可比 —— 由调用方决定是否剔除 extreme 行。"""
    x = df[xcol].to_numpy(float)
    y = df[ycol].to_numpy(float)
    n = len(df)
    if _st is not None:
        r, pr = _st.pearsonr(x, y)
        rho, prho = _st.spearmanr(x, y)
    else:
        r = float(np.corrcoef(x, y)[0, 1])
        pr = p_two_sided_t(r, n)
        rho = float(df[xcol].corr(df[ycol], method="spearman"))
        prho = p_two_sided_t(rho, n)
    return {"stat": tag, "n": n, "pearson_r": round(r, 3), "pearson_p": pr,
            "spearman_rho": round(rho, 3), "spearman_p": prho,
            "x": xcol, "y": ycol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/multitarget_confound.csv")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"找不到 {args.csv}；请先运行 multitarget_confound.py 生成它。")
    d = pd.read_csv(args.csv)
    no_excess = d[~d["label"].str.contains("excess")]

    rows = [
        # item 3：同一拓扑划分下，全特征 R² 与"仅描述符"R² 的追踪（全部 10 个目标）
        corr_block(d, "R2_topology_full", "R2_topology_geo",
                   "item3: R2_topology(full) vs R2_topology(geo-only)"),
        corr_block(no_excess, "R2_topology_full", "R2_topology_geo",
                   "item3: same, excluding the excess target"),
        # item 4：划分效应 vs 描述符可解释的份额（同一拓扑划分口径）
        corr_block(d, "split_effect_full", "R2_topology_geo",
                   "item4: split_effect vs R2_topology(geo-only)"),
        corr_block(no_excess, "split_effect_full", "R2_topology_geo",
                   "item4: same, excluding the excess target"),
        # 对照：若误用随机划分的描述符 R²，会得到不同（且先前被混用）的值
        corr_block(d, "split_effect_full", "R2_random_geo",
                   "control: split_effect vs R2_random(geo-only)"),
        corr_block(no_excess, "split_effect_full", "R2_random_geo",
                   "control: same, excluding the excess target"),
    ]
    out = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    dest = os.path.join(args.outdir, "multitarget_derived_stats.csv")
    out.to_csv(dest, index=False)

    pd.set_option("display.width", 200)
    print(f"派生统计量（来源 {os.path.basename(args.csv)}）"
          + ("" if _st is not None else "  [scipy 不可用：p 值用 t 近似回退]"))
    print(out[["stat", "n", "pearson_r", "pearson_p",
               "spearman_rho", "spearman_p"]].to_string(index=False))
    print(f"\n已写出 {dest}")
    print("\n口径提示：本脚本固定使用 R2_topology_geo 列（拓扑留出 + 仅结构描述符）。"
          "\n先前的 item 4 把 Pearson 取自该列、Spearman 取自 R2_random_geo 列，"
          "两者不可互相复算——control 行给出后者，供对照。")


if __name__ == "__main__":
    main()
