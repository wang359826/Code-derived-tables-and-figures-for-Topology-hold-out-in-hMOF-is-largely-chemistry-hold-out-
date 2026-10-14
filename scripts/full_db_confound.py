#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全库混淆统计（不采样）+ 拓扑对化学集合重叠（回应 R8 的连带问题）
================================================================

R8 查出：hMOF 的"98.0% 化学组成只出现在单一拓扑"是 50,000 样本的产物，
全库 324,426 个结构上其实是 93.2%。连带问题是：
"**除 pcu↔nbo 外所有拓扑对的化学集合重叠为 0**"这句话会不会也是小样本产物？
本脚本在**全库**上重算 Jaccard 矩阵，给出确定答案。

用法
----
  python full_db_confound.py                # 全库
  python full_db_confound.py --max-rows 50000   # 也可指定采样量做对照
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


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def confound(ck, tp):
    n = len(ck)
    d = pd.DataFrame({"ck": np.asarray(ck, dtype=object),
                      "tp": np.asarray(tp, dtype=object)})
    pair = d.groupby(["ck", "tp"]).size().reset_index(name="n_ct")
    ntop = pair.groupby("ck")["tp"].nunique()
    single = float((ntop == 1).mean())
    # 2026-10-11 新增（回应复核清单 B4）：两种加权口径都要给出。
    # 单拓扑占比默认是**组成等权**（每个组成算一次）；样本加权则问"多少比例的
    # 结构落在单拓扑组成里"。两者可以差 13 个百分点（粗粒度 74.7% vs 61.3%），
    # 因此头条数字必须写明用的是哪一种。
    n_by_c = pair.groupby("ck")["n_ct"].sum()
    single_w = float(n_by_c[ntop == 1].sum() / n)
    p_t = d["tp"].value_counts(normalize=True).to_numpy()
    HT = entropy(p_t)
    n_c = pair.groupby("ck")["n_ct"].transform("sum")
    # 样本加权的 H(T|C)（= Σ_c (n_c/N)·H(T|C=c)，此前唯一被报告的口径）
    hc = float((pair["n_ct"] / n * (-np.log2(pair["n_ct"] / n_c))).sum())
    # 组成等权的 H(T|C)：先在每个组成内部算熵，再对组成取平均
    def _h(s):
        p = s / s.sum()
        return float(-(p * np.log2(p)).sum())
    hc_comp = float(pair.groupby("ck")["n_ct"].apply(lambda s: _h(s.to_numpy())).mean())
    return {"n_samples": n, "n_chem": int(d["ck"].nunique()),
            "frac_single_topology": single, "H_T": HT, "H_T_given_C": hc,
            "cond_frac": float(hc / HT) if HT > 0 else np.nan,
            "r": n / max(1, int(d["ck"].nunique())),
            "n_topologies": int(d["tp"].nunique()),
            # 新增的两种加权口径
            "frac_single_topology_sample_weighted": single_w,
            "H_T_given_C_comp_weighted": hc_comp,
            "cond_frac_comp_weighted": float(hc_comp / HT) if HT > 0 else np.nan}


def jaccard_matrix(ck, tp, top_n=15):
    """返回 top_n 个最大拓扑之间的化学集合 Jaccard。"""
    d = pd.DataFrame({"ck": np.asarray(ck, dtype=object),
                      "tp": np.asarray(tp, dtype=object)})
    big = d["tp"].value_counts().head(top_n).index.tolist()
    sets = {t: set(d.loc[d["tp"] == t, "ck"]) for t in big}
    rows = []
    for a, b in itertools.combinations(big, 2):
        inter = len(sets[a] & sets[b])
        uni = len(sets[a] | sets[b])
        rows.append({"topo_a": a, "topo_b": b, "shared_chem": inter,
                     "jaccard": inter / uni if uni else 0.0})
    return pd.DataFrame(rows).sort_values("shared_chem", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        HERE, "..", "data", "all_MOFs_screening_data.csv"))
    ap.add_argument("--max-rows", type=int, default=None,
                    help="None = 全库")
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv, usecols=list(dict.fromkeys(
        H_FINE + ["topology"])), encoding="utf-8")
    df = df.dropna(subset=["topology"])
    if args.max_rows and args.max_rows < len(df):
        df = df.sample(args.max_rows, random_state=42)
        print(f"采样 {args.max_rows:,} 行")
    print(f"样本: {len(df):,} 行")

    tp = df["topology"].astype(str)
    out_rows, jac_all = [], []
    for keys, label in ((H_COARSE, "coarse(金属+两连接体)"),
                        (H_FINE, "fine(+官能团)")):
        ck = df[keys].fillna("NA").astype(str).agg("|".join, axis=1)
        st = confound(ck, tp)
        out_rows.append({"granularity": label, **st})
        print(f"\n{label}: 单拓扑占比 = {st['frac_single_topology']:.4f}  "
              f"H(T|C)/H(T) = {st['cond_frac']:.4f}  "
              f"M = {st['n_chem']:,}  r = {st['r']:.3f}")
        print(f"    样本加权单拓扑占比 = "
              f"{st['frac_single_topology_sample_weighted']:.4f}   "
              f"组成等权 H(T|C)/H(T) = {st['cond_frac_comp_weighted']:.4f}")
        j = jaccard_matrix(ck, tp)
        j["granularity"] = label
        jac_all.append(j)
        nz = j[j["shared_chem"] > 0]
        print(f"  非零重叠的拓扑对: {len(nz)} / {len(j)}")
        if len(nz):
            print(nz.head(12).to_string(index=False,
                                        float_format=lambda x: f"{x:.4f}"))

    d = pd.DataFrame(out_rows)
    p1 = os.path.join(args.outdir, f"confound_{args.tag}.csv")
    d.to_csv(p1, index=False, encoding="utf-8-sig")
    dj = pd.concat(jac_all)
    p2 = os.path.join(args.outdir, f"jaccard_{args.tag}.csv")
    dj.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p1}\n已保存: {p2}")


if __name__ == "__main__":
    main()
