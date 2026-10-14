#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
见化学率 S_g：把 §4.2 的机制主张从"三点代理"升级为直接测量
（回应审计清单 P1-2）

§4.2 现有的推理是**间接**的：三个分组变量的"粗细"不同 → 构筑单元身份
信息能否带出训练集不同 → 身份特征价值不同（−0.011 / +0.001 / +0.032）。
本脚本给出**直接测量**。身份特征由 426 维 one-hot 中的类别列构成，其类别
取值来自五个变量：

    topology, metal_linker, organic_linker1, organic_linker2, functional_groups

对每个分组变量 g，用该变量本身诱导的划分（留一分组 / GroupKFold），逐折
计算每一个类别特征在该折测试样本上的**可用率**

    a_f = 测试样本中「该折训练集里出现过取值为 f 的样本」的比例

再给出两个综合量：

    S_g^min  = min_f a_f          —— "木桶"口径：最稀缺的那一族身份还不能用
    S_g^prod = Π_f a_f            —— 联合口径：一个测试样本的身份特征整体可用的概率
    S_g^mean = mean_f a_f

S_g 越小 → 身份特征越无用。机制预测：S_g 应当与 §4.2 的身份效应
−0.011（topology）/ +0.001（metal）/ +0.032（linker pair）同序。

**不训练任何模型，纯计数。**

用法
----
  python chemistry_seen_grouping.py --csv data/all_MOFs_screening_data.csv
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
    "volume [A^3]",
    "weight [u]",
    "surface_area [m^2/g]",
    "void_fraction",
    "void_volume [cm^3/g]",
    "largest_free_sphere_diameter [A]",
    "largest_included_sphere_along_free_sphere_path_diameter [A]",
    "largest_included_sphere_diameter [A]",
]


def load_draw(csv_path: str, max_rows: int, seed: int) -> pd.DataFrame:
    """与 multitarget_confound.py / mof_leakage_benchmark.py 完全一致的取数。"""
    df = pd.read_csv(csv_path)
    if max_rows and max_rows < len(df):
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    d = df[["topology", "metal_linker", "organic_linker1",
            "organic_linker2", "functional_groups"]
           + NUMERIC_FEATURES + [TARGET]]
    d = d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return d


def feature_arrays(d: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "topology": d["topology"].astype(str).to_numpy(),
        "metal_linker": d["metal_linker"].astype(str).to_numpy(),
        "organic_linker1": d["organic_linker1"].astype(str).to_numpy(),
        "organic_linker2": d["organic_linker2"].astype(str).to_numpy(),
        "linker_pair": (d["organic_linker1"].astype(str) + "_"
                        + d["organic_linker2"].astype(str)).to_numpy(),
        "functional_groups": d["functional_groups"].astype(str).to_numpy(),
    }


def availability_for_split(feats: dict[str, np.ndarray],
                           folds: list[tuple[np.ndarray, np.ndarray]]
                           ) -> dict[str, float]:
    """按折样本数加权的逐特征可用率。"""
    n = len(next(iter(feats.values())))
    num = {k: 0.0 for k in feats}
    for tr, te in folds:
        for k, v in feats.items():
            train = np.unique(v[tr])
            hit = np.isin(v[te], train).sum()
            num[k] += (hit / len(te)) * len(te)
    return {k: num[k] / n for k in feats}


def leave_one_group_folds(cat: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(cat))
    folds = []
    for g in pd.unique(cat):
        te = idx[cat == g]
        tr = idx[cat != g]
        if len(te):
            folds.append((tr, te))
    return folds


def groupkfold_folds(cat: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros((len(cat), 1)), groups=cat))


def kfold_folds(n: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=n_splits, shuffle=True,
                      random_state=seed).split(np.zeros((n, 1))))


def summarise(avail: dict[str, float], owner: str | None) -> dict[str, float]:
    """把逐特征可用率压成综合量。

    owner = 该划分所留出的那个变量名（其自身的 one-hot 列在该折训练集中
    **按构造恒为不可用**，因此 "保留下来的身份信息" 用其余列的平均值刻画）。
    """
    own_map = {"topology": "topology",
               "metal centre (metal_linker)": "metal_linker",
               "linker pair (organic_linker1+2)": "linker_pair"}
    own = own_map.get(owner or "", None)
    rest = [v for k, v in avail.items()
            if k not in {own, "organic_linker1", "organic_linker2"}]
    # organic_linker1/2 只是构造 linker_pair 的原料，不单独作为 one-hot 列；
    # 但在 linker_pair 划分下它们仍可用，故只在综合量里剔除自身列。
    return {"avail_of_grouping_variable_itself":
            float(avail[own]) if own and own in avail else float("nan"),
            "avail_surviving_identity_mean": float(np.mean(rest)) if rest else np.nan,
            "avail_surviving_identity_min": float(np.min(rest)) if rest else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        HERE, "..", "data", "all_MOFs_screening_data.csv"))
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = load_draw(args.csv, args.max_rows, args.seed)
    n = len(d)
    print(f"建模样本: {n:,} 行（seed={args.seed}；正文写作 25,000，"
          f"dropna 后实际 {n:,}）")

    feats = feature_arrays(d)

    splits = {
        "topology": leave_one_group_folds(feats["topology"]),
        "metal centre (metal_linker)": leave_one_group_folds(feats["metal_linker"]),
        "linker pair (organic_linker1+2)": leave_one_group_folds(feats["linker_pair"]),
        "random split (reference)": kfold_folds(n, 10, args.seed),
    }
    n_groups = {
        "topology": int(pd.unique(feats["topology"]).size),
        "metal centre (metal_linker)": int(pd.unique(feats["metal_linker"]).size),
        "linker pair (organic_linker1+2)": int(pd.unique(feats["linker_pair"]).size),
        "random split (reference)": 10,
    }
    # 机制 §4.2 实测身份效应（十目标配对差中位数），用于并排对照
    effect = {"topology": -0.011, "metal centre (metal_linker)": +0.001,
              "linker pair (organic_linker1+2)": +0.032,
              "random split (reference)": +0.031}

    rows = []
    for name, folds in splits.items():
        avail = availability_for_split(feats, folds)
        summ = summarise(avail, name)
        row = {"grouping_variable": name, "n_groups": n_groups[name],
               "identity_effect_median": effect[name], **summ}
        row.update({f"avail_{k}": v for k, v in avail.items()})
        rows.append(row)
        print(f"\n  {name}  (组数 {n_groups[name]})")
        for k, v in avail.items():
            print(f"      {k:18s} 可用率 = {v:.4f}")
        print(f"      → 自身列 {summ['avail_of_grouping_variable_itself']:.3f}  "
              f"| 其余身份列 均值 {summ['avail_surviving_identity_mean']:.3f}  "
              f"最小 {summ['avail_surviving_identity_min']:.3f}  "
              f"| 身份效应 {effect[name]:+.3f}")

    out = pd.DataFrame(rows)
    p = os.path.join(args.outdir, "chemistry_seen_grouping.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p}")

    # 机制一致性检验：保留下来的身份信息多寡是否与身份效应同序
    from scipy.stats import spearmanr
    for m in ("avail_surviving_identity_mean", "avail_surviving_identity_min"):
        rho, pv = spearmanr(out[m].to_numpy(float),
                            out["identity_effect_median"].to_numpy(float))
        print(f"  Spearman({m} vs identity effect) = {rho:+.3f} (p = {pv:.3f}, n = 4)")


if __name__ == "__main__":
    main()
