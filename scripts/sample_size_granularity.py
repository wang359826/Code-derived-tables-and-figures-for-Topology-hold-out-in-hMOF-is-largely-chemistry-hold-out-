#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本量依赖 + 同 n 同粒度的跨库对照（回应 R8）
==============================================

R8 的两条实质问题：
------------------
① hMOF 的"98.0% 化学组成只出现在单一拓扑"是 **50,000 样本**的产物，
   而全库有 324,426 个结构。样本越小，同一化学组成被拆到多个拓扑的机会越少，
   排他率天然越高（25,000 → 98.9%，50,000 → 98.0%）。
   → 必须给出**样本量曲线**，并以全库数字为准（或明确标注 n）。
② §3.14 结论二的跨库比较用的是"随机压缩到同一粒度"——而这正是本稿 §4.3 第 14 条
   **自己宣布禁用**的方法（随机压缩只能把化学定义变粗，不能变细）。自相矛盾。
   → 本脚本改用**同一样本量 n + 两种真实粒度定义**做对照，不做任何随机重组。

做法
----
对 hMOF 与 ARC-MOF，分别在 n ∈ {25k, 50k, 100k, 200k, 全库} 上，
用各库自己的两种真实粒度定义计算：
    单拓扑化学占比、H(T|C)/H(T)、每个化学的平均样本数 r = n / M

产出
----
  outputs/sample_size_granularity.csv
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
ARCDIR = os.path.join(HERE, "..", "data", "arcmof")

H_COARSE = ["metal_linker", "organic_linker1", "organic_linker2"]
H_FINE = H_COARSE + ["functional_groups"]


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def confound(ck, tp):
    """向量化的单拓扑占比 + 条件熵（不用 groupby.apply，18 万组会挂）。"""
    n = len(ck)
    d = pd.DataFrame({"ck": np.asarray(ck, dtype=object),
                      "tp": np.asarray(tp, dtype=object)})
    pair = d.groupby(["ck", "tp"]).size().reset_index(name="n_ct")
    ntop = pair.groupby("ck")["tp"].nunique()
    single = float((ntop == 1).mean())
    p_t = d["tp"].value_counts(normalize=True).to_numpy()
    HT = entropy(p_t)
    n_c = pair.groupby("ck")["n_ct"].transform("sum")
    hc = float((pair["n_ct"] / n * (-np.log2(pair["n_ct"] / n_c))).sum())
    return {
        "n_samples": n,
        "n_chem": int(d["ck"].nunique()),
        "frac_single_topology": single,
        "H_T": HT,
        "H_T_given_C": hc,
        "cond_frac": float(hc / HT) if HT > 0 else np.nan,
        "r": n / max(1, int(d["ck"].nunique())),
        "n_topologies": int(d["tp"].nunique()),
    }


def chem_key(df, keys):
    return df[keys].fillna("NA").astype(str).agg("|".join, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmof", default=os.path.join(
        HERE, "..", "data", "all_MOFs_screening_data.csv"))
    ap.add_argument("--arc-topo", default=os.path.join(
        ARCDIR, "all_topology_lists.csv"))
    ap.add_argument("--mc", default=os.path.join(ARCDIR, "mc-clusters.csv"))
    ap.add_argument("--flig", default=os.path.join(ARCDIR, "flig-clusters.csv"))
    ap.add_argument("--func", default=os.path.join(ARCDIR, "func-clusters.csv"))
    ap.add_argument("--sizes", default="25000,50000,100000,200000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    rows = []

    # ---------------- hMOF ----------------
    print("读取 hMOF …")
    hm = pd.read_csv(args.hmof, usecols=list(dict.fromkeys(
        H_FINE + ["topology"])), encoding="utf-8")
    hm = hm.dropna(subset=["topology"])
    print(f"  hMOF 全库 {len(hm):,} 行")
    tot = len(hm)
    for n in sizes + [tot]:
        if n >= tot:
            sub = hm
            tag = f"{tot}（全库）"
        else:
            sub = hm.sample(n, random_state=args.seed)
            tag = f"{n}"
        for gname, keys in (("coarse(金属+两连接体)", H_COARSE),
                            ("fine(+官能团)", H_FINE)):
            st = confound(chem_key(sub, keys), sub["topology"].astype(str))
            rows.append({"database": "hMOF", "n_requested": tag,
                         "granularity": gname, **st})
            print(f"  hMOF n={tag:>12s} {gname:22s} "
                  f"单拓扑={st['frac_single_topology']:.4f} "
                  f"r={st['r']:.3f}")

    # ---------------- ARC-MOF ----------------
    print("\n读取 ARC-MOF …")
    ar = pd.read_csv(args.arc_topo, encoding="utf-8", low_memory=False)
    ar = ar.rename(columns={"Name": "struct", "filename": "topo_from_name",
                            "Crystalnet": "topo",
                            "likely topology": "topo_likely"})
    for path, col in ((args.mc, "mc"), (args.flig, "flig"), (args.func, "func")):
        c = pd.read_csv(path, encoding="utf-8").rename(
            columns={"filename": "struct", "cluster_id": col})
        ar = ar.merge(c[["struct", col]], on="struct", how="left")
    ar = ar.dropna(subset=["topo", "mc", "flig", "func"])
    print(f"  ARC-MOF 可用 {len(ar):,} 行")
    tot_a = len(ar)
    for n in sizes + [tot_a]:
        if n >= tot_a:
            sub = ar
            tag = f"{tot_a}（全库）"
        else:
            sub = ar.sample(n, random_state=args.seed)
            tag = f"{n}"
        coarse = (sub["mc"].astype(str) + "|" + sub["flig"].astype(str))
        fine = coarse + "|" + sub["func"].astype(str)
        for gname, key in (("coarse(mc+flig)", coarse),
                           ("fine(+func)", fine)):
            st = confound(key, sub["topo"].astype(str))
            rows.append({"database": "ARC-MOF", "n_requested": tag,
                         "granularity": gname, **st})
            print(f"  ARC  n={tag:>12s} {gname:22s} "
                  f"单拓扑={st['frac_single_topology']:.4f} "
                  f"r={st['r']:.3f}")

    d = pd.DataFrame(rows)
    p = os.path.join(args.outdir, "sample_size_granularity.csv")
    d.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p}\n")
    print(d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("怎么读")
    print("=" * 78)
    print("  · 同一库内看 n 从小到大：单拓扑占比若系统性下降，说明该指标")
    print("    **依赖样本量**，稿件必须写明 n（R8-①）。")
    print("  · 跨库比较时只在同一 n、同一粒度档位（coarse 对 coarse，")
    print("    fine 对 fine）上看——**不做任何随机压缩对齐**（R8-②）。")


if __name__ == "__main__":
    main()
