#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精英子集检验：在"筛选真正会选中的那一小撮材料"里，混淆还在吗？
================================================================

审稿人/读者最自然的一个反驳是：

    "你的结论建立在**全库**上。可我做筛选只关心**顶部**那一小撮材料——
     万一顶部材料恰好跨越了多个拓扑、多个连接体，混淆就不成立了。"

本脚本直接回答这个问题：把 confound_diagnosis 的混淆诊断
原封不动跑一遍 `top_MOFs_screening_data.csv`（8,206 行，hMOF 的头部材料子集），
再与全库随机 25,000 行的结果对照。

指标（与 §3.2 完全一致，可直接并列）
------------------------------------
- single_topology_chemistry_frac : 只出现在单一拓扑的化学组成占比
- H(topology), H(topology|chemistry), 条件熵占比
- 拓扑分布集中度（最大拓扑占比、Top-3 占比、归一化熵）

为什么这值得做
--------------
如果精英子集的**集中度比全库还高**（很可能，因为高性能材料往往成簇），
那么本文的结论不但没被削弱，反而在最需要它的地方更强：
**恰恰是筛选真正会挑出来的那批材料，化学与拓扑绑得最死。**

运行
----
    python elite_subset_confound.py

输出
----
    outputs/elite_subset_confound.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confound_diagnosis import (  # noqa: E402
    KEYS_COARSE, KEYS_FINE, confound_report, entropy,
)
import mof_leakage_benchmark as mb  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass


def concentration(df, topo_col="topology"):
    """拓扑分布集中度。"""
    vc = df[topo_col].astype(str).value_counts()
    p = (vc / vc.sum()).to_numpy()
    h = entropy(p)
    hmax = np.log2(len(p)) if len(p) > 1 else 1.0
    return {
        "n_samples": int(len(df)),
        "n_topologies": int(len(vc)),
        "max_topo_share": float(p[0]),
        "top3_topo_share": float(p[:3].sum()),
        "norm_entropy": float(h / hmax) if hmax > 0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-seeds", type=int, default=0,
                    help=">1 时，额外把 \"Random 25,000 of full db\" 参照行在这么多个种子上"
                         "重算一遍，写出 elite_reference_seeds.csv（均值±sd 与范围）。"
                         "2026-10-14 新增：Table 17 的参照行此前是单次抽样（seed 42），"
                         "而 elite 本身是固定文件、核心统计量是确定性计数，"
                         "所以这一节唯一需要多种子的就是参照行。")
    ap.add_argument("--seed-list", default="42,143,244,345,446",
                    help="参照行使用的种子列表")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "..", "outputs")
    os.makedirs(outdir, exist_ok=True)

    full = os.path.join(here, "..", "data", "all_MOFs_screening_data.csv")
    elite = os.path.join(here, "..", "data", "top_MOFs_screening_data.csv")

    rows = []

    # 全库随机 25,000 行（与其余章节同一套抽样，保证可比）
    df_full = mb.load_data(full, 25000)
    for keys, gran in ((KEYS_COARSE, "coarse"), (KEYS_FINE, "fine")):
        r, _jac = confound_report(df_full, keys, label=f"full_25k_{gran}")
        r.update(concentration(df_full))
        r["population"] = "full 25,000 (random)"
        r["granularity"] = gran
        rows.append(r)

    # ---- 参照行的多种子不确定性（2026-10-14 新增）----
    if args.reference_seeds > 1:
        seed_list = [int(s) for s in args.seed_list.split(",")][:args.reference_seeds]
        print(f"\n参照行多种子重算：{seed_list}")
        srows = []
        for sd in seed_list:
            mb.RANDOM_STATE = sd          # 抽样层的种子
            d = mb.load_data(full, 25000)
            for keys, gran in ((KEYS_COARSE, "coarse"), (KEYS_FINE, "fine")):
                r, _ = confound_report(d, keys, label=f"full_25k_{gran}")
                srows.append({
                    "seed": sd,
                    "granularity": gran,
                    "n_chem": r["n_chem"],
                    "frac_single_topology": r["frac_single_topology"],
                    "H_T_given_C": r["H_T_given_C"],
                    "cond_frac": r["cond_frac"],
                })
            print(f"  seed {sd}: " + "  ".join(
                f"{x['granularity']} n_chem={x['n_chem']} "
                f"single={100*x['frac_single_topology']:.1f}% "
                f"ratio={100*x['cond_frac']:.2f}%"
                for x in srows[-2:]))
        sdf = pd.DataFrame(srows)
        summary = (sdf.groupby("granularity")
                   .agg(n_chem_mean=("n_chem", "mean"), n_chem_min=("n_chem", "min"),
                        n_chem_max=("n_chem", "max"),
                        frac_single_mean=("frac_single_topology", "mean"),
                        frac_single_sd=("frac_single_topology", "std"),
                        frac_single_min=("frac_single_topology", "min"),
                        frac_single_max=("frac_single_topology", "max"),
                        cond_frac_mean=("cond_frac", "mean"),
                        cond_frac_sd=("cond_frac", "std"),
                        cond_frac_min=("cond_frac", "min"),
                        cond_frac_max=("cond_frac", "max"),
                        n_seeds=("seed", "count"))
                   .reset_index())
        p_ref = os.path.join(outdir, "elite_reference_seeds.csv")
        sdf.to_csv(p_ref, index=False, encoding="utf-8-sig")
        p_sum = os.path.join(outdir, "elite_reference_seeds_summary.csv")
        summary.to_csv(p_sum, index=False, encoding="utf-8-sig")
        print(f"\n已保存: {p_ref}\n已保存: {p_sum}")
        print(summary.to_string(index=False))
        mb.RANDOM_STATE = 42          # 还原，避免影响后续

    # 精英子集
    df_elite = pd.read_csv(elite)
    n_missing_fg = int(df_elite["functional_groups"].isna().sum())
    print(f"\n精英子集: {len(df_elite)} 行 x {df_elite.shape[1]} 列"
          f"（functional_groups 缺失 {n_missing_fg} 行）")
    # 2026-10-14 修：Table 17 的 fine 行明确声明「剔除非标注行」（6,214 组成 over
    # 8,145 行）。原版对 elite 的 fine 直接用 confound_report 的 fillna("NA") 口径，
    # 把 61 行缺失值当成一个组成水平，给出 6,275 over 8,206，与正文不一致。
    # 现按 dropna 口径出主行，并另出一行 fillna 口径以便对照——两种口径都存档，
    # 不再有一个被引文件与正文数字不符的情况。
    for keys, gran, df_use, note in (
            (KEYS_COARSE, "coarse", df_elite, "all rows (no missing coarse keys)"),
            (KEYS_FINE, "fine", df_elite.dropna(subset=["functional_groups"]),
             f"drop {n_missing_fg} rows with missing functional_groups"),
            (KEYS_FINE, "fine_fillna", df_elite,
             'unannotated functional_groups treated as one value ("NA")')):
        r, jac = confound_report(df_use, keys, label=f"elite_{gran}")
        if gran == "fine":
            jac.to_csv(os.path.join(outdir, "elite_subset_jaccard.csv"),
                       index=False, encoding="utf-8-sig")
        r.update(concentration(df_use))
        r["population"] = f"elite {len(df_use):,} (top MOFs)"
        r["granularity"] = "fine" if gran.startswith("fine") else "coarse"
        r["missing_value_rule"] = note
        rows.append(r)

    out = pd.DataFrame(rows)
    p = os.path.join(outdir, "elite_subset_confound.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 78)
    print(out.to_string(index=False))
    print(f"\n已保存: {p}")


if __name__ == "__main__":
    main()
