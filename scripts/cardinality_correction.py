# -*- coding: utf-8 -*-
"""
Table 15（§3.8 遗漏变量分析）的**基数校正**：ω² + 置换零模型
================================================================

动机
----
§3.8 的 Table 15 用 η² 与「折外 R²」衡量 topology / metal_linker /
linker_pair 三个分类变量能解释多少目标方差。linker_pair 有 **587 组**、
平均每组约 43 个样本，因此正文写了一条「Magnitude caveat」：这两个统计量
都被基数抬高，并说「a cardinality-corrected (ω²) or grouped-CV estimate
would be the way to price it」。

本脚本就是那个 price：把「被抬高多少」从一句话变成数字。

做了什么
--------
1. **ω²**（omega-squared）——η² 的常用无偏化版本：

       ω² = (SS_between − (k−1)·MS_within) / (SS_total + MS_within)

   其中 MS_within = SS_within / (N − k)。k 越大、每组越小，惩罚越大。
   对 η²(target) 与 η²(residual) 各算一遍，再用 ω² 重算「absorbed」。

2. **η² 的置换零模型**：把分组标签随机重排（保持组大小分布），B 次，
   得到 η² 在「无信号」下的分布。理论期望是 (k−1)/(N−1)，这里用实测确认。

3. **折外 R² 的置换零模型**：把分组标签打乱后重跑同一套 5 折折外 R²
   （同一个模型、同一个 RANDOM_STATE），看 587 组的 one-hot 在纯噪声上
   能拿到多少 R²。折外协议本身应当把基数抬高挡在门外，这一项就是验证它。

数据口径与 `omitted_variable.py` **完全一致**：同一个 `--max-rows 25000`
抽样（random_state=42）、同一个目标列、同一套 8 个描述符、同一个
RandomForest 配置，因此 η² / R² 应逐位复现 `outputs/omitted_variable_final.csv`。

用法
----
    python cardinality_correction.py --max-rows 25000 --perm 10
输出
----
    outputs/cardinality_correction.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from omitted_variable import (  # noqa: E402
    TARGET, NUMERIC_FEATURES, RANDOM_STATE, eta_squared, cv_r2,
    oof_residuals,
)
import omitted_variable as ov  # noqa: E402
# 2026-10-14：本脚本的随机森林来自 omitted_variable._make_model，其并行度由该模块的
# N_JOBS_OVERRIDE 决定。受限沙箱下 joblib 的进程池无法创建命名管道（WinError 5），
# 所以这里把它接到 --n-jobs 上，使本脚本在管道受限的环境里也能顺序跑完。

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass


def omega_squared(y: np.ndarray, groups: np.ndarray) -> float:
    """ω²：η² 的基数无偏化版本（k 组、N 样本）。"""
    y = np.asarray(y, dtype=float)
    g = np.asarray(groups)
    N = len(y)
    k = pd.Series(g).nunique()
    if N - k <= 1 or k < 2:
        return np.nan
    grand = y.mean()
    ss_tot = float(((y - grand) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    gm = pd.DataFrame({"y": y, "g": g}).groupby("g")["y"].agg(["mean", "count"])
    ss_b = float((gm["count"] * (gm["mean"] - grand) ** 2).sum())
    ss_w = ss_tot - ss_b
    ms_w = ss_w / (N - k)
    return float((ss_b - (k - 1) * ms_w) / (ss_tot + ms_w))


def eta2_null_bound(k: int, N: int) -> float:
    """无信号时 η² 的期望上界：E[η²] = (k−1)/(N−1)。"""
    return (k - 1) / (N - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "..", "data",
                                                  "all_MOFs_screening_data.csv"))
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "outputs"))
    ap.add_argument("--perm", type=int, default=3,
                    help="折外 R² 置换零模型的重复次数（每次 5 折 RF）。"
                         "linker_pair 的 one-hot 有 587 列，单次折外拟合约 1 分钟，"
                         "故默认 B=3（每变量共 6 次拟合）；设 0 则完全跳过置换零。")
    ap.add_argument("--r2-vars", default="linker_pair",
                    help="为哪些变量跑折外 R² 置换零（逗号分隔）；"
                         "RF 在 587 列上很贵，默认只跑基数最高的 linker_pair")
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="底层 RandomForest 的并行度（透传给 omitted_variable."
                         "N_JOBS_OVERRIDE）。固定 random_state 下 n_jobs 不改变拟合结果，"
                         "所以受限沙箱里传 1 可得到同样的数值（2026-10-14 新增）")
    args = ap.parse_args()

    ov.N_JOBS_OVERRIDE = args.n_jobs

    df = pd.read_csv(args.csv)
    if args.max_rows and args.max_rows < len(df):
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)

    y = pd.to_numeric(df[TARGET], errors="coerce")
    Xg = df[[c for c in NUMERIC_FEATURES if c in df.columns]].astype(float)
    ok = y.notna() & Xg.notna().all(axis=1)
    df, y, Xg = (df[ok].reset_index(drop=True), y[ok].reset_index(drop=True),
                 Xg[ok].reset_index(drop=True))

    group_defs = {
        "topology": df["topology"].astype(str).values,
        "metal_linker": df["metal_linker"].astype(str).values,
        "linker_pair": (df["organic_linker1"].astype(str) + "_"
                        + df["organic_linker2"].astype(str)).values,
    }

    N = len(df)
    print("=" * 78)
    print("Table 15 的基数校正（ω² + 置换零模型）")
    print(f"样本 N = {N:,}；模型与抽样同 omitted_variable.py")
    print("=" * 78)

    rng = np.random.default_rng(args.seed)

    out_path = os.path.join(args.outdir, "cardinality_correction.csv")

    def dump(tag=""):
        """边算边存：任何一步被中断，已完成的部分都不会丢。"""
        o = pd.DataFrame(rows)
        o.insert(0, "n_samples_geometric_model", N)
        o.insert(1, "R2_geometric_model", r2_geo)
        os.makedirs(args.outdir, exist_ok=True)
        o.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"    ↳ 已落盘 {os.path.basename(out_path)}{tag}")

    print("\n[1/4] 几何模型的折外残差（用于 η²(residual) / R²(residual)）…")
    res = oof_residuals(Xg, y)
    r2_geo = 1.0 - float(np.nanvar(res) / np.nanvar(y.values))
    print(f"    几何模型随机划分 R² = {r2_geo:.4f}（期望 0.6721）")
    yv, resv = y.values, res

    rows = []
    print("\n[2/4] η² / ω² 与 η² 的置换零…")
    for name, g in group_defs.items():
        k = int(pd.Series(g).nunique())
        e2t = eta_squared(yv, g)
        e2r = eta_squared(resv, g)
        w2t = omega_squared(yv, g)
        w2r = omega_squared(resv, g)
        # η² 的置换零（无模型拟合，很便宜）
        null_t, null_r = np.empty(2000), np.empty(2000)
        for b in range(2000):
            gp = rng.permutation(g)
            null_t[b] = eta_squared(yv, gp)
            null_r[b] = eta_squared(resv, gp)
        print(f"  {name:12s} k={k:>4}  η²(tgt)={e2t:.4f} ω²={w2t:.4f}  "
              f"η²(res)={e2r:.4f} ω²={w2r:.4f}  "
              f"E[η²|null]={(k-1)/(N-1):.5f} (perm mean "
              f"{null_t.mean():.5f} / {null_r.mean():.5f})")
        rows.append(dict(
            variable=name, n_groups=k, n_samples=N,
            eta2_target=e2t, omega2_target=w2t,
            eta2_residual=e2r, omega2_residual=w2r,
            absorbed_eta2=(1 - e2r / e2t) if e2t > 0 else np.nan,
            absorbed_omega2=(1 - w2r / w2t) if w2t > 0 else np.nan,
            eta2_null_expect=(k - 1) / (N - 1),
            eta2_null_perm_mean_target=float(null_t.mean()),
            eta2_null_perm_p95_target=float(np.percentile(null_t, 95)),
            eta2_null_perm_mean_residual=float(null_r.mean()),
            eta2_null_perm_p95_residual=float(np.percentile(null_r, 95)),
            granularity="table15_25000",
        ))
    dump("（ω² 阶段完成）")

    print("\n[3/4] 折外 R² 及其置换零（组标签打乱后重跑同一 5 折 RF）…")
    r2_vars = set() if args.perm <= 0 else {
        s.strip() for s in args.r2_vars.split(",") if s.strip()}
    for row, (name, g) in zip(rows, group_defs.items()):
        Xo = pd.get_dummies(pd.Series(g), prefix=name).astype(float)
        r2a = cv_r2(Xo, y)
        r2r = cv_r2(Xo, pd.Series(res))
        row["R2_alone"] = r2a
        row["R2_residual"] = r2r
        if name in r2_vars:
            n_t, n_r = [], []
            import time as _t
            t0 = _t.time()
            for b in range(args.perm):
                gp = rng.permutation(g)
                Xp = pd.get_dummies(pd.Series(gp), prefix=name).astype(float)
                n_t.append(cv_r2(Xp, y))
                n_r.append(cv_r2(Xp, pd.Series(res)))
            row["R2_alone_excess_over_null"] = r2a - float(np.mean(n_t))
            row["R2_residual_excess_over_null"] = r2r - float(np.mean(n_r))
            row["R2_null_perm_mean_alone"] = float(np.mean(n_t))
            row["R2_null_perm_mean_residual"] = float(np.mean(n_r))
            row["R2_null_perm_n"] = args.perm
            print(f"  {name:12s} R²(alone)={r2a:+.4f} (null {np.mean(n_t):+.4f}, "
                  f"B={args.perm}, {_t.time()-t0:.0f}s)   "
                  f"R²(resid)={r2r:+.4f} (null {np.mean(n_r):+.4f})")
        else:
            row["R2_alone_excess_over_null"] = np.nan
            row["R2_residual_excess_over_null"] = np.nan
            row["R2_null_perm_mean_alone"] = np.nan
            row["R2_null_perm_mean_residual"] = np.nan
            row["R2_null_perm_n"] = 0
            print(f"  {name:12s} R²(alone)={r2a:+.4f}   R²(resid)={r2r:+.4f}"
                  f"   [未做置换零：k={row['n_groups']} 太小，基数无可抬]")
        dump(f"（{name} 完成）")

    print("\n[4/4] 保存 …")
    print("saved:", out_path)

    print("\n" + "=" * 78)
    print("判读")
    print("=" * 78)
    for row in rows:
        print(f"  {row['variable']:12s} k={row['n_groups']:>4}  "
              f"absorbed: η² 版 {row['absorbed_eta2']*100:.1f}%  →  "
              f"ω² 版 {row['absorbed_omega2']*100:.1f}%")
    print("  若 ω² 版与 η² 版同量级，则 §3.8 的 caveat 是保守的、方向不变。")


if __name__ == "__main__":
    main()
