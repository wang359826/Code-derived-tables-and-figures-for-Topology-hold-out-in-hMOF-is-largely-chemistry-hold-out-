#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
见化学率 S_g 与"化学未见面"分解（回应审计清单 P0-5 / P1-1 / P1-2 / P1-4）
==========================================================================

审计清单指出：稿件标题说 "largely chemistry hold-out"，但正文只报了
**组成层** 74.7%（每个组成算一次），从未报读者真正要的**样本层**数字——
"一次拓扑留出所移走的样本里，有多大比例其化学在训练集中没有对应"。

本脚本用纯计数（不训练任何模型）给出三个层次，并把它逐拓扑分解：

  层次 1（组成层）  74.7%  —— 单拓扑组成 / 全部组成
  层次 2（样本层）  见 chemistry_seen_by_topology.csv 的合并行
  层次 3（分解层）  逐留出拓扑：见化学率 S_g = "该拓扑的样本中，其化学
                    在**其它**拓扑中也出现过"的比例。留出该拓扑后，
                    1 − S_g 就是"化学未见面"的比例。

S_g 就是审计清单 §4.2 要求的那张分解表，同时也是 P1-2（§4.2 机制
主张的直接测量）所需要的连续变量：S_g 越大，测试样本的构筑单元在
训练集中出现得越多，identity 特征就越有价值。

另给出 P1-4 所需的"可承载对照的最低共享组成门槛"统计量：全库 55 个
拓扑对里，共享组成数的分布（只有两个对非零）。

用法
----
  python chemistry_seen.py                       # 全库
  python chemistry_seen.py --csv <path>          # 指定数据
  python chemistry_seen.py --max-rows 50000      # 采样对照（可选）
"""

from __future__ import annotations

import argparse
import itertools
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

H_COARSE = ["metal_linker", "organic_linker1", "organic_linker2"]
H_FINE = H_COARSE + ["functional_groups"]


def _chemistry_keys(df: pd.DataFrame, keys, fillna: bool = True) -> pd.Series:
    """与 full_db_confound.py 保持完全一致的组成键构造方式。"""
    sub = df[keys]
    if fillna:
        sub = sub.fillna("NA")
    return sub.astype(str).agg("|".join, axis=1)


def decomposition(df: pd.DataFrame, keys, label: str) -> tuple[pd.DataFrame, dict]:
    """逐拓扑分解"化学在训练集中是否有对应"。

    对一个留出拓扑 t：
      n_t           留出该拓扑时被移走的样本数
      seen_t        其中化学在**其它拓扑**中也出现的样本数
      S_g = seen_t / n_t        —— 见化学率（g = t）
      1 − S_g                   —— 化学未见面比例

    合并口径（**两种，勿混**）：
      · **行口径 / 结构口径**：Σ_c n_c·1[k_c>1] / N
        —— "多少比例的**结构**其组成横跨 ≥2 个拓扑"，本表各行 **Seen 列之和**
        恰等于它。它与"组成单拓扑的结构占比"互补。
      · **成员关系口径**：Σ_c n_c·(k_c−1) / Σ_c n_c·k_c
        —— "在组成×拓扑成员关系里，有对应关系的占比"；对 Σ_c n_c·k_c 这个
        分母而言，它是 55.8%，其补是 44.2%。
    两者分母不同（324,426 vs 449,838），**不可互换**。
    """
    ck = _chemistry_keys(df, keys)
    tp = df["topology"].astype(str).reset_index(drop=True)
    d = pd.DataFrame({"ck": ck.values, "tp": tp.values})

    n_c = d.groupby("ck")["tp"].size()               # 每个组成的样本数
    n_top = d.groupby("ck")["tp"].nunique()          # 每个组成横跨几个拓扑
    ktot = d["ck"].map(n_top)                        # 每行：其组成横跨的拓扑数

    # 行口径：结构级"有对应"
    seen_rows_total = int((ktot > 1).sum())

    # 逐拓扑：只数"属于该拓扑"的那些行里、组成横跨 >1 的
    rows = []
    for t, g in d.groupby("tp"):
        n_t = len(g)
        cnt = g.groupby("ck").size()                 # 该拓扑内每个组成的样本数
        seen_t = int(sum(v for c, v in cnt.items() if n_top[c] > 1))
        rows.append({
            "held_out_topology": t,
            "n_samples": n_t,
            "seen": seen_t,
            "S_g_seen_fraction": (seen_t / n_t) if n_t else np.nan,
            "unseen_fraction": (1 - seen_t / n_t) if n_t else np.nan,
        })
    per = pd.DataFrame(rows).sort_values("n_samples", ascending=False)
    per.insert(0, "granularity", label)

    N = len(d)
    m_single = int(n_c[n_top == 1].sum())            # 单拓扑组成的样本（结构级未见）
    m_multi = int(n_c[n_top > 1].sum())              # 多拓扑组成的样本（结构级有对应）
    pair_unseen = int((n_c[n_top > 1] * (n_top[n_top > 1] - 1)).sum())
    total_pair = int((n_c[n_top > 1] * n_top[n_top > 1]).sum())
    total_memb = m_single + total_pair

    combined = {
        "granularity": label,
        "n_samples": N,
        "n_compositions": int(d["ck"].nunique()),
        "n_topologies": int(d["tp"].nunique()),
        "single_topology_compositions": int((n_top == 1).sum()),
        "multi_topology_compositions": int((n_top > 1).sum()),
        # 层次 1：组成层（论文头条口径）
        "frac_single_topology_composition_weighted": float((n_top == 1).mean()),
        # 层次 2a：结构层（"落在单拓扑组成里的结构占比" = 结构级未见面）
        "structures_single_topology_composition": m_single,
        "structures_multi_topology_composition": m_multi,
        "frac_single_topology_sample_weighted": float(m_single / N),
        "unseen_fraction_per_structure": float(m_single / N),
        "seen_fraction_per_structure": float(m_multi / N),
        # 层次 2b：成员关系层（Σ n_c·k_c = 449,838 为分母）
        "membership_single": m_single,
        "membership_total": total_memb,
        "membership_seen_fraction": float(m_single / total_memb),
        "membership_unseen_fraction": float(total_pair / total_memb),
        # 行口径（本表 Seen 列之和）必须等于 structures_multi_topology_composition
        "row_based_seen_total_check": seen_rows_total,
        # 干净化学留出的拓扑：S_g = 0 的拓扑个数与其样本占比
        "n_topologies_pure_chemistry_holdout": int((per["seen"] == 0).sum())
              if len(per) else 0,
        "sample_share_pure_chemistry_holdout": float(
            per.loc[per["seen"] == 0, "n_samples"].sum() / N) if len(per) else np.nan,
    }
    return per, combined


def pair_overlap_stats(df: pd.DataFrame, keys, label: str) -> pd.DataFrame:
    """P1-4：拓扑对共享组成的分布（可承载对照的门槛）。"""
    ck = _chemistry_keys(df, keys)
    tp = df["topology"].astype(str)
    d = pd.DataFrame({"ck": ck.values, "tp": tp.values})
    sets = {t: set(g["ck"]) for t, g in d.groupby("tp")}
    rows = []
    for a, b in itertools.combinations(sorted(sets), 2):
        inter = len(sets[a] & sets[b])
        uni = len(sets[a] | sets[b])
        rows.append({"granularity": label, "topo_a": a, "topo_b": b,
                     "shared_chem": inter,
                     "jaccard": inter / uni if uni else 0.0})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        HERE, "..", "data", "all_MOFs_screening_data.csv"))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--topology-col", default="topology",
                    help="ARC-MOF 使用 'Crystalnet' 作为拓扑列名")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    usecols = list(dict.fromkeys(H_FINE + [args.topology_col]))
    df = pd.read_csv(args.csv, usecols=usecols, encoding="utf-8")
    df = df.dropna(subset=["topology"]).reset_index(drop=True)
    if args.max_rows and args.max_rows < len(df):
        df = df.sample(args.max_rows, random_state=42).reset_index(drop=True)
        print(f"采样 {args.max_rows:,} 行")
    print(f"样本: {len(df):,} 行")

    per_all, comb_all, jac_all = [], [], []
    for keys, label in ((H_COARSE, "coarse(metal+2 linkers)"),
                        (H_FINE, "fine(+functional groups)")):
        per, comb = decomposition(df, keys, label)
        per_all.append(per)
        comb_all.append(comb)
        jac_all.append(pair_overlap_stats(df, keys, label))
        print(f"\n=== {label} ===")
        print(f"  组成层 单拓扑组成占比 = "
              f"{comb['frac_single_topology_composition_weighted']:.4%} "
              f"({comb['single_topology_compositions']:,}/"
              f"{comb['n_compositions']:,})")
        print(f"  结构层 单拓扑组成的结构占比 = "
              f"{comb['frac_single_topology_sample_weighted']:.4%}"
              f"  → **结构级未见** = {comb['unseen_fraction_per_structure']:.4%}"
              f"  (未见 {comb['structures_single_topology_composition']:,}"
              f" / 有对应 {comb['structures_multi_topology_composition']:,})")
        print(f"  成员关系层 有对应/未见 = "
              f"{comb['membership_seen_fraction']:.4%} / "
              f"{comb['membership_unseen_fraction']:.4%}  "
              f"({comb['membership_single']:,}/{comb['membership_total']:,})")
        print(f"  行口径对账：Seen 列之和 = {comb['row_based_seen_total_check']:,}"
              f"  （应等于结构级有对应 "
              f"{comb['structures_multi_topology_composition']:,}）"
              f"  {'OK' if comb['row_based_seen_total_check'] == comb['structures_multi_topology_composition'] else '❌ 不一致'}")
        print(f"  纯化学留出拓扑数 = "
              f"{comb['n_topologies_pure_chemistry_holdout']}/"
              f"{comb['n_topologies']}  （样本占比 "
              f"{comb['sample_share_pure_chemistry_holdout']:.2%}）")
        print(per.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    comb = pd.DataFrame(comb_all)
    per = pd.concat(per_all, ignore_index=True)
    jac = pd.concat(jac_all, ignore_index=True)

    tag = f"_{args.tag}" if args.tag else ""
    p1 = os.path.join(args.outdir, f"chemistry_seen{tag}.csv")
    p2 = os.path.join(args.outdir, f"chemistry_seen_by_topology{tag}.csv")
    p3 = os.path.join(args.outdir, f"topology_pair_overlap{tag}.csv")
    comb.to_csv(p1, index=False, encoding="utf-8-sig")
    per.to_csv(p2, index=False, encoding="utf-8-sig")
    jac.to_csv(p3, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p1}\n已保存: {p2}\n已保存: {p3}")


if __name__ == "__main__":
    main()
