# -*- coding: utf-8 -*-
r"""
每个拓扑的"化学排他率"：留出该拓扑时，它的化学组成有多少是完全没见过的？

定义
----
对于拓扑 t，设 C_t = 该拓扑下所有化学组成的集合，C_other = 其余拓扑的化学组成并集。
    exclusivity(t) = |C_t \ C_other| / |C_t|
exclusivity = 1 → 留出 t 时训练集里一个同化学结构都没有（纯外推）
exclusivity = 0 → t 的化学全都在别的拓扑里出现过（可以插值）

这是"留出拓扑 ≡ 留出化学"这句话最直接的度量，比 Jaccard 更好读：
Jaccard 会被小集合放大（|A∩B|/|A∪B|，两个都很小时分母小），
exclusivity 是单向的、按拓扑样本加权，不会被小拓扑的偶然重叠主导。

对 hMOF 与 ARC-MOF 各算一遍，得到跨数据库对照。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
DATA = os.path.join(HERE, "..", "data")


def exclusivity_table(df, chem_keys, topo_col, min_size=30):
    """返回逐拓扑的 exclusivity 表（只保留样本量 >= min_size 的拓扑）。"""
    d = df.dropna(subset=chem_keys + [topo_col]).copy()
    chem = d[chem_keys].astype(str).agg("|".join, axis=1)
    d = d.assign(_c=chem)
    counts = d[topo_col].value_counts()
    rows = []
    for t, n in counts.items():
        if n < min_size:
            continue
        Ct = set(d.loc[d[topo_col] == t, "_c"])
        Co = set(d.loc[d[topo_col] != t, "_c"])
        inter = len(Ct & Co)
        rows.append({
            "topology": t, "n": int(n), "n_chem": len(Ct),
            "n_chem_shared_elsewhere": inter,
            "exclusivity": (len(Ct) - inter) / len(Ct) if Ct else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main():
    out = []

    # ---------------- hMOF（Boyd–Woo 数据库，本项目主数据库） ----------------
    print("=" * 70)
    print("hMOF (Boyd-Woo 2019, 本文主数据库)")
    print("=" * 70)
    h = pd.read_csv(os.path.join(DATA, "all_MOFs_screening_data.csv"),
                    usecols=["MOFname", "topology", "metal_linker",
                             "organic_linker1", "organic_linker2",
                             "functional_groups"],
                    low_memory=False)
    for gran, keys in [("coarse(metal+2linker)",
                        ["metal_linker", "organic_linker1", "organic_linker2"]),
                       ("fine(+functional_groups)",
                        ["metal_linker", "organic_linker1", "organic_linker2",
                         "functional_groups"])]:
        t = exclusivity_table(h, keys, "topology", min_size=30)
        t["database"] = "hMOF(Boyd-Woo)"
        t["granularity"] = gran
        out.append(t)
        w = (t["exclusivity"] * t["n"]).sum() / t["n"].sum()
        print(f"\n  [{gran}] 拓扑数 = {len(t)}（样本 >= 30）")
        print(f"    样本量加权 exclusivity = {w:.4f}   "
              f"中位 = {t['exclusivity'].median():.4f}   "
              f"最小 = {t['exclusivity'].min():.4f}")
        print(t[["topology", "n", "n_chem", "n_chem_shared_elsewhere",
                 "exclusivity"]].head(12).to_string(index=False))

    # ---------------- ARC-MOF ----------------
    print("\n" + "=" * 70)
    print("ARC-MOF (Burner et al. 2023, 混合库)")
    print("=" * 70)
    top = pd.read_csv(os.path.join(DATA, "arcmof", "all_topology_lists.csv"),
                      encoding="utf-8-sig", dtype=str)
    top.columns = [c.strip() for c in top.columns]
    top = top.rename(columns={"Name": "file", "filename": "topology_in_filename"})
    for name, key in [("mc-clusters.csv", "mc"),
                      ("flig-clusters.csv", "flig"),
                      ("func-clusters.csv", "func")]:
        f = pd.read_csv(os.path.join(DATA, "arcmof", name),
                        encoding="utf-8-sig", dtype={"filename": str})
        f.columns = [c.strip() for c in f.columns]
        f = f.rename(columns={"cluster_id": key, "filename": "file"})
        top = top.merge(f[["file", key]].drop_duplicates("file"),
                        on="file", how="left")
    top["file"] = top["file"].astype(str)

    # 用 Crystalnet（独立预测的拓扑，与文件名无循环依赖）作为主拓扑列
    for topo_col in ["Crystalnet", "topology_in_filename"]:
        for gran, keys in [("coarse(mc+flig)", ["mc", "flig"]),
                           ("fine(mc+flig+func)", ["mc", "flig", "func"])]:
            t = exclusivity_table(top, keys, topo_col, min_size=30)
            if t.empty:
                continue
            t["database"] = "ARC-MOF"
            t["granularity"] = gran
            t["topo_col"] = topo_col
            out.append(t)
            w = (t["exclusivity"] * t["n"]).sum() / t["n"].sum()
            print(f"\n  [{topo_col} | {gran}] 拓扑数 = {len(t)}（样本 >= 30）")
            print(f"    样本量加权 exclusivity = {w:.4f}   "
                  f"中位 = {t['exclusivity'].median():.4f}   "
                  f"最小 = {t['exclusivity'].min():.4f}")
            print(t[["topology", "n", "n_chem",
                     "n_chem_shared_elsewhere", "exclusivity"]].head(10)
                  .to_string(index=False))

    allt = pd.concat(out, ignore_index=True)
    p = os.path.join(OUT, "exclusivity_across_databases.csv")
    allt.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n已保存:", p)

    # 汇总一行一库
    print("\n" + "=" * 70)
    print("汇总：样本量加权的化学排他率")
    print("=" * 70)
    for (db, gran), g in allt.groupby(["database", "granularity"]):
        w = (g["exclusivity"] * g["n"]).sum() / g["n"].sum()
        print(f"  {db:16s} {gran:26s} 拓扑 {len(g):3d} 个  "
              f"n={int(g['n'].sum()):7d}  weighted excl = {w:.4f}  "
              f"median = {g['exclusivity'].median():.4f}")


if __name__ == "__main__":
    main()
