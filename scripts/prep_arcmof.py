#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prep_arcmof.py —— 把 ARC-MOF 的四个原始元数据文件合并成化学分解脚本可直接消费的
单一 CSV，列名与 chemistry_seen.py / full_db_confound.py 保持一致。

ARC-MOF 原始结构
----------------
  all_topology_lists.csv : Name, filename, Crystalnet, likely topology
  mc-clusters.csv        : filename, cluster_id, Database      (金属中心簇)
  flig-clusters.csv      : filename, cluster_id, Database      (有机连接体簇)
  func-clusters.csv      : filename, cluster_id, Database      (官能团簇)

输出
----
  data/arcmof/arcmof_confound.csv
      filename, topology, metal_linker, organic_linker1, organic_linker2,
      functional_groups, source

  metal_linker          <- mc-clusters.cluster_id
  organic_linker1       <- flig-clusters.cluster_id
  organic_linker2       <- flig-clusters.cluster_id（ARC-MOF 无第二个连接体列，
                           按 coarse 口径与 chemistry_seen 的键构造保持一致；
                           同一组成内的样本其值相同，不影响分组）
  functional_groups     <- func-clusters.cluster_id
  source                <- DB 前缀（filename 的前 3 个字符，如 DB0/DB12）
  topology              <- Crystalnet 列

用法
----
  python prep_arcmof.py --data-dir data/arcmof --out data/arcmof/arcmof_confound.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "data", "arcmof"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--topology-col", default="Crystalnet")
    args = ap.parse_args()

    dd = args.data_dir
    out = args.out or os.path.join(dd, "arcmof_confound.csv")

    topo = pd.read_csv(os.path.join(dd, "all_topology_lists.csv"))
    mc = pd.read_csv(os.path.join(dd, "mc-clusters.csv"))
    fl = pd.read_csv(os.path.join(dd, "flig-clusters.csv"))
    fn = pd.read_csv(os.path.join(dd, "func-clusters.csv"))

    print(f"all_topology_lists : {len(topo):,} 行")
    print(f"mc-clusters        : {len(mc):,} 行")
    print(f"flig-clusters      : {len(fl):,} 行")
    print(f"func-clusters      : {len(fn):,} 行")

    base = topo[["Name", args.topology_col]].rename(
        columns={args.topology_col: "topology", "Name": "filename"})

    mc = mc[["filename", "cluster_id"]].rename(columns={"cluster_id": "metal_linker"})
    fl = fl[["filename", "cluster_id"]].rename(columns={"cluster_id": "organic_linker1"})
    fn = fn[["filename", "cluster_id"]].rename(columns={"cluster_id": "functional_groups"})

    df = base.merge(mc, on="filename", how="left")
    df = df.merge(fl, on="filename", how="left")
    df = df.merge(fn, on="filename", how="left")

    # ARC-MOF 只有一个连接体簇列；organic_linker2 用同一值填充以复用 coarse 键构造
    df["organic_linker2"] = df["organic_linker1"]

    # 来源子库：filename 以 DB<n>- 开头
    df["source"] = df["filename"].astype(str).str.extract(r"^(DB\d+)")[0]
    df["source"] = df["source"].fillna("OTHER")

    cols = ["filename", "topology", "metal_linker", "organic_linker1",
            "organic_linker2", "functional_groups", "source"]
    df = df[cols]

    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {out}")
    print(f"行数 {len(df):,}；来源子库 {df['source'].nunique()} 个；"
          f"拓扑 {df['topology'].nunique()} 个")
    miss = df[["metal_linker", "organic_linker1", "functional_groups"]].isna().sum()
    print("\n缺失计数（化学列）：")
    print(miss.to_string())


if __name__ == "__main__":
    main()
