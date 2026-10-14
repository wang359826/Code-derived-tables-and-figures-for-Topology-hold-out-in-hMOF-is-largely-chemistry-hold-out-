# -*- coding: utf-8 -*-
"""
粒度对照检验：ARC-MOF 的化学排他率更高，是不是因为"化学组成"划得更细？

问题
----
exclusivity_across_databases.py 得到：
    hMOF(Boyd-Woo) fine  加权排他率 0.878   （1.55 个样本 / 化学组成）
    ARC-MOF        fine  加权排他率 0.995   （1.25 个样本 / 化学组成）
两者粒度**接近**，但并非严格相同；审稿人一定会问"是不是你把 ARC-MOF 的
化学划得更细，所以排他率天然更高"。

检验设计
--------
对两个数据库的**细粒度化学 ID 序列**做同一件事：
把 M 个唯一化学组成用随机映射压缩到 m 组（m 从小到大），
每压缩一次就重算一次"样本量加权的化学排他率"。
- 横轴 m 表示化学组成的"粗度"（m 越小越粗）
- 若 ARC-MOF 的整条曲线都在 hMOF 之上，则结论与粒度无关，是数据库性质；
- 若两条曲线在相同 m 下重合，则说明差异只是粒度造成的假象。

随机映射对每个 m 用若干种子重复，取均值与极值，避免单次映射偶然性。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
DATA = os.path.join(HERE, "..", "data")


def weighted_exclusivity(chem_ids, topo, min_size=1000):
    """给定化学 ID 序列与拓扑序列，返回样本量加权的排他率。"""
    d = pd.DataFrame({"c": chem_ids, "t": topo}).dropna()
    counts = d["t"].value_counts()
    keep = counts[counts >= min_size].index
    d = d[d["t"].isin(keep)]
    if d.empty:
        return np.nan, 0, 0
    per = d.groupby("t")["c"].apply(set)
    allsets = list(per)
    num = den = 0.0
    for t, Ct in per.items():
        others = set().union(*[s for u, s in per.items() if u != t])
        ex = (len(Ct) - len(Ct & others)) / len(Ct)
        n = counts[t]
        num += ex * n
        den += n
    return num / den, len(per), d["c"].nunique()


def compress(chem_ids, m, seed):
    """把唯一化学 ID 随机压缩到 m 组。

    ⚠️ 当 m >= 唯一 ID 数时必须直接返回原 ID：若照样随机映射，M 个 ID 扔进
    M 个桶会有约 37% 的碰撞（生日悖论），m=M 就不再等于"不压缩"，
    会把满粒度的排他率凭空压低——这是本脚本第一版踩过的坑。
    """
    codes, uniq = pd.factorize(chem_ids)
    if m >= len(uniq):
        return pd.Series(codes, index=chem_ids.index)
    rng = np.random.default_rng(seed)
    lut = rng.integers(0, m, size=len(uniq))
    return pd.Series(lut[codes], index=chem_ids.index)


def main():
    # ---------- hMOF ----------
    h = pd.read_csv(os.path.join(DATA, "all_MOFs_screening_data.csv"),
                    usecols=["metal_linker", "organic_linker1",
                             "organic_linker2", "functional_groups", "topology"],
                    low_memory=False)
    hfin = (h[["metal_linker", "organic_linker1", "organic_linker2",
               "functional_groups"]].fillna("NA").astype(str)
            .agg("|".join, axis=1))
    h_topo = h["topology"].astype(str)
    h_M = hfin.nunique()
    print(f"hMOF: n={len(h)}, 细粒度唯一化学组成 M={h_M}")

    # ---------- ARC-MOF ----------
    top = pd.read_csv(os.path.join(DATA, "arcmof", "all_topology_lists.csv"),
                      encoding="utf-8-sig", dtype=str)
    top.columns = [c.strip() for c in top.columns]
    top = top.rename(columns={"Name": "file", "filename": "topology_in_filename"})
    for name, key in [("mc-clusters.csv", "mc"), ("flig-clusters.csv", "flig"),
                      ("func-clusters.csv", "func")]:
        f = pd.read_csv(os.path.join(DATA, "arcmof", name),
                        encoding="utf-8-sig", dtype={"filename": str})
        f.columns = [c.strip() for c in f.columns]
        f = f.rename(columns={"cluster_id": key, "filename": "file"})
        top = top.merge(f[["file", key]].drop_duplicates("file"), on="file", how="left")
    top = top.dropna(subset=["mc", "flig", "func"])
    afin = (top[["mc", "flig", "func"]].astype("int64").astype(str)
            .agg("|".join, axis=1))
    a_topo = top["Crystalnet"].astype(str)
    a_M = afin.nunique()
    print(f"ARC-MOF: n={len(top)}, 细粒度唯一化学组成 M={a_M}")

    # ---------- 扫描 ----------
    # ⚠️ 关键：不能按"绝对组数 m"比较两个数据库，因为两者样本量不同
    #    （hMOF 324,426 vs ARC-MOF 214,543），相同 m 下 avg_n/chem 并不相同。
    #    必须在**相同的"每个化学组成的平均样本数" r = n/m** 下比较，
    #    即令 m_db = n_db / r。r 才是"化学粒度"的可比度量。
    rows = []
    # 网格下界取两库中**较粗**的那个满粒度 r = n/M，保证两库在该点都能落在
    # 真实压缩状态（否则细的那个库会被 M 截断，r 与网格点不符，比较失效）。
    r_floor = max(len(h) / h_M, len(top) / a_M)
    ratio_grid = sorted({round(r_floor, 3), 2.0, 3.0, 5.0, 10.0, 30.0,
                         100.0, 300.0, 1000.0, 3000.0})
    print(f"\n满粒度 r: hMOF {len(h)/h_M:.3f}, ARC-MOF {len(top)/a_M:.3f} "
          f"→ 网格下界 {r_floor:.3f}\n")
    for r in ratio_grid:
        for db, chem, tp, M in [("hMOF(Boyd-Woo)", hfin, h_topo, h_M),
                                ("ARC-MOF", afin, a_topo, a_M)]:
            mm = int(min(round(len(chem) / r), M))
            vals = []
            for s in range(4):
                cc = compress(chem, mm, seed=s)
                w, ntopo, nchem = weighted_exclusivity(cc, tp)
                if not np.isnan(w):
                    vals.append(w)
            if not vals:
                continue
            rows.append({
                "database": db, "r_target": r,
                "r_samples_per_chem": len(chem) / mm,
                "m_groups": mm, "M_full": M,
                "is_full_granularity": bool(mm >= M),
                "exclusivity_mean": float(np.mean(vals)),
                "exclusivity_min": float(np.min(vals)),
                "exclusivity_max": float(np.max(vals)),
                "n_topologies_used": ntopo,
            })
            print(f"  r={len(chem)/mm:8.2f}  m={mm:7d}/{M:7d}  {db:16s}  "
                  f"excl = {np.mean(vals):.4f}  [{np.min(vals):.4f}, {np.max(vals):.4f}]")

    d = pd.DataFrame(rows)
    p = os.path.join(OUT, "granularity_control.csv")
    d.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n已保存:", p)

    # ---------- 结论判读 ----------
    print("\n" + "=" * 70)
    print("判读：把两库对齐到相同的「每化学平均样本数」后比较")
    print("=" * 70)
    piv = d.pivot_table(index="r_target", columns="database",
                        values="exclusivity_mean")
    piv["ARC_minus_hMOF"] = piv["ARC-MOF"] - piv["hMOF(Boyd-Woo)"]
    piv = piv.sort_index(ascending=False)   # r 越大 = 化学粒度越粗
    print(piv.round(4).to_string())
    # 只在两边都有值的粒度点上判定
    valid = piv.dropna(subset=["ARC-MOF", "hMOF(Boyd-Woo)"])
    n_above = int((valid["ARC_minus_hMOF"] > 0).sum())
    print(f"\n  ARC-MOF 高于 hMOF 的粒度点：{n_above}/{len(valid)}")
    if n_above == len(valid):
        print("  → 在所有化学粒度下 ARC-MOF 的排他率都更高。"
              "差异不是粒度造成的，而是数据库构造方式的差异。")
    elif n_above == 0:
        print("  → 在所有化学粒度下 hMOF 的排他率都不低于 ARC-MOF。"
              "两库的差异**不能**用'ARC-MOF 更混淆'来解释——"
              "原始差异主要是粒度造成的。")
    else:
        print("  → 存在粒度点使结论反转，必须谨慎表述，只能报'量级相当'。")


if __name__ == "__main__":
    main()
