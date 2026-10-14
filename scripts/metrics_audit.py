#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
指标口径审计：pooled / macro / weighted / median 一律全报
====================================================

为什么要有这个脚本（2026-10-09）
------------------------------
`mof_leakage_benchmark.py` 的 `evaluate_loto()` 返回的是
**把 11 折预测拼接后算一次的 pooled R²**，而逐折 R² 另存到 CSV。
这两个数不是一回事，而且可以差得非常远：

    留一拓扑（10 万样本）
      pooled   R² = +0.381   ← 把所有折的预测拼起来，一次算完
      weighted R² = -0.059   ← 逐拓扑 R² 按测试集大小加权平均
      macro    R² = -0.541   ← 逐拓扑 R² 简单平均（每种拓扑同等权重）
      median   R² = +0.278   ← 逐拓扑 R² 中位数

pooled 之所以为正，是因为 pcu（63,447 个样本、R²=+0.298）在池子里占 63%，
把 etb(−2.49)、fof(−1.92)、nbo(−0.94) 的灾难性失败稀释掉了。
**用 pooled 去代表"未见拓扑下的典型表现"是错的**，方向甚至是反的：
剔除 pcu 后其余 10 组的加权平均是 −0.678。

本脚本从已落盘的 per-group / per-fold CSV 重算四种口径，
输出一份"再也不能拿 pooled 冒充"的对照表（Table 2 修正版）。

用法
----
  python metrics_audit.py --outdir outputs
"""

import argparse
import os

import numpy as np
import pandas as pd


def four_ways(d, r2col="R2", ncol="n_test"):
    """对逐组 R² 计算四种聚合口径。"""
    v = d[r2col].to_numpy(dtype=float)
    n = d[ncol].to_numpy(dtype=float) if ncol in d.columns else np.ones(len(v))
    return {
        "macro": float(v.mean()),          # 每组权重相同
        "weighted": float(np.average(v, weights=n)),  # 按样本量加权
        "median": float(np.median(v)),
        "min": float(v.min()),
        "max": float(v.max()),
        "n_groups": int(len(v)),
        "n_negative": int((v < 0).sum()),
        "frac_negative": float((v < 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    out_rows = []
    reports = []

    # ------------------------------------------------------------------
    # 1) 留一拓扑：per-group CSV
    # ------------------------------------------------------------------
    for fn, label, pooled_fn in [
        ("loto_per_topology_final.csv", "LOTO / topology (100k)",
         "loto_results_final.csv"),
        ("loto_per_metal_final.csv", "LOTO / metal (100k)",
         "loto_results_final.csv"),
    ]:
        p = os.path.join(args.outdir, fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        agg = four_ways(d)
        pooled = None
        pp = os.path.join(args.outdir, pooled_fn)
        if os.path.exists(pp):
            lr = pd.read_csv(pp)
            key = ("留一拓扑" if "topology" in fn else "留一金属中心")
            m = lr[lr["strategy"].astype(str).str.contains(key, na=False)]
            if len(m):
                pooled = float(m["R2"].iloc[0])

        # 剔除最大组后再算一次，检验"被大组主导"的程度
        if "topology" in fn:
            big = d.loc[d["n_test"].idxmax()]
            rest = d[d["group"] != big["group"]]
            agg_rest = four_ways(rest)
        else:
            big, agg_rest = None, None

        out_rows.append({
            "result": label,
            "pooled_as_reported": pooled,
            **agg,
        })
        if agg_rest:
            out_rows.append({
                "result": label + f"  [剔除最大组 {big['group']}]",
                "pooled_as_reported": None,
                **agg_rest,
            })

        reports.append(
            f"\n【{label}】\n"
            f"  稿件里报的 pooled R² = {pooled:.4f}\n"
            f"  macro(每组等权)      = {agg['macro']:+.4f}\n"
            f"  weighted(按 n 加权)  = {agg['weighted']:+.4f}\n"
            f"  median               = {agg['median']:+.4f}\n"
            f"  范围 {agg['min']:+.3f} ~ {agg['max']:+.3f}   "
            f"负值组 {agg['n_negative']}/{agg['n_groups']} "
            f"({agg['frac_negative'] * 100:.0f}%)\n"
            + (f"  剔除 {big['group']}(n={int(big['n_test'])}) 后："
               f"weighted {agg_rest['weighted']:+.4f}, "
               f"macro {agg_rest['macro']:+.4f}, "
               f"median {agg_rest['median']:+.4f}\n" if agg_rest else "")
        )

    # ------------------------------------------------------------------
    # 2) 拓扑内 vs 跨拓扑（3.10）：只有 6 个拓扑
    # ------------------------------------------------------------------
    p = os.path.join(args.outdir, "within_vs_cross_final.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        rep = "\n【3.10 拓扑内 vs 跨拓扑（25k，8 个结构描述符）】\n"
        for col, name in (("R2_within", "within"), ("R2_cross", "cross")):
            a = four_ways(d.rename(columns={col: "R2"}))
            rep += (f"  {name:7s} macro {a['macro']:+.4f}   "
                    f"weighted {a['weighted']:+.4f}   "
                    f"median {a['median']:+.4f}   "
                    f"范围 {a['min']:+.3f} ~ {a['max']:+.3f}   "
                    f"负值 {a['n_negative']}/{a['n_groups']}\n")
            out_rows.append({"result": f"within_vs_cross / {name} (25k)",
                             "pooled_as_reported": None, **a})
        reports.append(rep)

    # The withdrawn CGCNN diagnostic is intentionally not merged into this audit.
    # It used a different row schema and is documented separately in SI Table S2;
    # keeping it out prevents a mixed-schema metrics_audit.csv from being consumed
    # as if every row represented a grouped-split aggregation.

    df = pd.DataFrame(out_rows)
    op = os.path.join(args.outdir, "metrics_audit.csv")
    df.to_csv(op, index=False, encoding="utf-8-sig")

    print("=" * 78)
    print("指标口径审计")
    print("=" * 78)
    for r in reports:
        print(r)
    print("\n" + "=" * 78)
    print("规则（写进 Methods，以后一律遵守）")
    print("=" * 78)
    print("  1. 凡是分组划分，必须同时报告 macro / weighted / median / pooled 四者。")
    print("  2. 正文里用作主结论的，必须是 macro 或 median（每组同等权重的'典型表现'），")
    print("     不能是 pooled（会被最大组主导）。")
    print("  3. 若 pooled 与 macro 符号相反，必须在正文明确说明，并解释原因。")
    print("  4. 比较 R² 时，两侧测试集的目标变量方差必须可比；方差不同则改用")
    print("     RMSE 或标准化后的指标。")
    print(f"\n已保存: {op}")


if __name__ == "__main__":
    main()
