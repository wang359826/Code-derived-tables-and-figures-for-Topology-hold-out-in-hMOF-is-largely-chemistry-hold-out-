#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
elite_vs_full.py —— §3.12 归因数字的溯源脚本（回应 R3-M4）。

该节声称以下数字来自存档，但它们从未被脚本产出：
  - 全库 ≥ 2.0 mmol/g 的行数（9,588）
  - elite 文件行数（8,206）
  - 逐拓扑 qualifying / retained / lost 计数
  - 保留行与被排除行的平均吸附量
  - 四个缺失拓扑在全库的最大吸附量
  - 448 个粗粒度组成中在全库多拓扑的个数

本脚本从 data/all_MOFs_screening_data.csv 与 data/top_MOFs_screening_data.csv
出发，直接计算上述量，产出 elite_vs_full.csv。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-csv", default=os.path.join(HERE, "..", "data",
                                                       "all_MOFs_screening_data.csv"))
    ap.add_argument("--elite-csv", default=os.path.join(HERE, "..", "data",
                                                        "top_MOFs_screening_data.csv"))
    ap.add_argument("--target", default="CO2_uptake_P0.15bar_T298K [mmol/g]")
    ap.add_argument("--threshold", type=float, default=2.0,
                    help="精英子集的吸附量阈值（mmol/g）")
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 加载
    full = pd.read_csv(args.full_csv)
    elite = pd.read_csv(args.elite_csv)

    # 全库 ≥ threshold 的行集合
    full_qual = full[full[args.target] >= args.threshold].copy()

    # ---- 行匹配：必须按真实主键连接，禁止退化为行索引比较 ----
    # 2026-10-14 修：原版检查 "filename" 列，但两个 hMOF CSV 都没有该列（主键是
    # MOFname），于是静默走进 else 分支用 RangeIndex 比较，把「保留/排除」变成了
    # 「全库文件顺序的前 8,206 行 / 其余 1,382 行」，产出与主稿 §3.12 相反的逐拓扑
    # 计数与均值（pcu lost 684 对 265、tbo lost 0 对 1、lost_mean 2.046 对 2.685）。
    # 现改为按主键连接，并在任一假设不成立时直接失败
    # （REPORTING_CHECKLIST D4：空帧必须响亮地失败，不能静默跳过）。
    KEY_CANDIDATES = ("MOFname", "filename", "name")
    key = next((k for k in KEY_CANDIDATES if k in full.columns and k in elite.columns), None)
    if key is None:
        raise SystemExit(
            "elite_vs_full: 找不到两表共有的主键列（试过 %s）。"
            "不要退化为行索引比较——那会产出与正文相反的数字。" % (KEY_CANDIDATES,))
    if full[key].duplicated().any():
        raise SystemExit("elite_vs_full: 全库主键 %r 有重复，无法唯一匹配。" % key)
    print(f"  行匹配主键 = {key!r}（全库 {len(full):,} 行，elite {len(elite):,} 行）")

    elite_keys = set(elite[key].dropna().astype(str))
    qual_key = full_qual[key].astype(str)
    retained = full_qual[qual_key.isin(elite_keys)]
    lost = full_qual[~qual_key.isin(elite_keys)]

    if len(retained) == 0 or len(lost) == 0:
        raise SystemExit("elite_vs_full: 匹配后 retained 或 lost 为空——主键口径错了。")
    if len(retained) + len(lost) != len(full_qual):
        raise SystemExit("elite_vs_full: retained + lost != qualifying，匹配不完整。")
    if len(retained) != len(elite):
        print(f"  ⚠️ retained({len(retained):,}) != elite 行数({len(elite):,})："
              f"elite 文件里有 {len(elite) - len(retained):,} 行不满足阈值或不在全库中。")

    retained_keys = set(retained[key].astype(str))

    # 逐拓扑统计（retained / lost 用同一个按键匹配出来的集合，不再按行索引）
    topo_stats = []
    for topo, g in full_qual.groupby("topology"):
        gk = g[key].astype(str)
        n_ret = int(gk.isin(retained_keys).sum())
        topo_stats.append({
            "topology": topo,
            "qualifying": len(g),
            "retained": n_ret,
            "lost": len(g) - n_ret,
        })

    # §3.12 的阈值论证讨论的是「精英子集丢掉的那四个拓扑」的最大吸附量，
    # 原版误列成精英子集**保留**的四个（acs/bcu/fof/pts），因此那四行数值不能
    # 支撑正文关于 nbo / etb / rht / tbo 的论证。这里改为正文实际使用的四个。
    missing_topos = ["nbo", "etb", "rht", "tbo"]
    max_uptake = {}
    for t in missing_topos:
        sub = full[full["topology"] == t]
        max_uptake[t] = float(sub[args.target].max()) if len(sub) > 0 else np.nan

    # 粗粒度组成的多拓扑计数：§3.12 要的是「精英子集保留的 448 个组成里在全库多拓扑的
    # 个数」（131），不是全库的总数（410）。两个都存档，键名分开，避免混用。
    from confound_diagnosis import KEYS_COARSE
    comp_key = full[KEYS_COARSE].fillna("NA").astype(str).agg("|".join, axis=1)
    topo_per_comp = full.groupby(comp_key)["topology"].nunique()
    n_multi_full_db = int((topo_per_comp > 1).sum())
    elite_tmp = pd.read_csv(args.elite_csv)
    elite_comp_key = (elite_tmp[KEYS_COARSE].fillna("NA").astype(str)
                      .agg("|".join, axis=1))
    n_multi_elite = int(sum(1 for c in set(elite_comp_key)
                            if topo_per_comp.get(c, 1) > 1))

    # 汇总
    rows = [{
        "metric": "full_qualifying",
        "value": len(full_qual),
        "note": "全库 ≥ 2.0 mmol/g 的行数",
    }, {
        "metric": "elite_rows",
        "value": len(elite),
        "note": "elite 文件行数",
    }, {
        "metric": "retained_rows",
        "value": len(retained),
        "note": "同时出现在全库 qualifying 与 elite 中的行数",
    }, {
        "metric": "lost_rows",
        "value": len(lost),
        "note": "全库 qualifying 但不在 elite 中的行数",
    }, {
        "metric": "retained_mean_uptake",
        "value": float(retained[args.target].mean()) if len(retained) > 0 else np.nan,
        "note": "保留行的平均吸附量",
    }, {
        "metric": "lost_mean_uptake",
        "value": float(lost[args.target].mean()) if len(lost) > 0 else np.nan,
        "note": "排除行的平均吸附量",
    }, {
        "metric": "full_qualifying_mean_uptake",
        "value": float(full_qual[args.target].mean()),
        "note": "全库 qualifying 的平均吸附量",
    }, {
        "metric": "n_compositions_multi_topology_elite",
        "value": n_multi_elite,
        "note": "精英子集保留的粗粒度组成中，在全库横跨 >1 个拓扑的个数（§3.12 的 131）",
    }, {
        "metric": "n_compositions_multi_topology_full_db",
        "value": n_multi_full_db,
        "note": "全库粗粒度组成中横跨 >1 个拓扑的个数（§3.2 的 410）",
    }]
    for t in missing_topos:
        rows.append({
            "metric": f"max_uptake_{t}",
            "value": max_uptake[t],
            "note": f"全库中拓扑 {t} 的最大吸附量",
        })

    out = pd.DataFrame(rows)
    p = os.path.join(args.outdir, "elite_vs_full.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"已保存: {p}")
    print(out.to_string(index=False))

    # 逐拓扑明细
    topo_df = pd.DataFrame(topo_stats)
    p2 = os.path.join(args.outdir, "elite_vs_full_by_topology.csv")
    topo_df.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p2}")
    print(topo_df.to_string(index=False))


if __name__ == "__main__":
    import argparse  # noqa: E402
    main()
