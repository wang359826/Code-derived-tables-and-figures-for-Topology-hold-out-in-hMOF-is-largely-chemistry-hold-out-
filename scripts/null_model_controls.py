#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
null_model_controls.py —— SI §S3 的两项零模型对照（规格已定、数据就位后即可运行）。

对照 ①：精英子集拓扑置换零模型（主稿 §3.12）
    只保留精英子集中实际出现的 7 种拓扑，随机置换拓扑标签但保持每个拓扑的
    样本数（边缘分布）不变，B 次置换后统计"单拓扑化学组成占比"的零分布，
    与观测值 100.0% 对比。若零分布本身接近 100%，则 §3.12 的"更极端"只是
    "子集更小 + 拓扑池更窄"的算术后果。

对照 ②：按拓扑基数条件化的来源比较（主稿 §3.15 Result 3）
    对每个来源子库，在与该来源拓扑数相同的条件下重算排他率（保持观测到的
    拓扑数边缘分布），检验跨来源排序是否只是拓扑丰富度 H(T) 的机械投影。
    §3.15 已给出提示性相关 Spearman ρ = −0.86 (p = 0.006)。

用法（数据就位后）：
    python null_model_controls.py --data-dir ../data
    输入文件（MD5 见 DATA.md）：
      data/top_MOFs_screening_data.csv        精英子集 8,206 行
      data/arcmof/all_topology_lists.csv      264,225 行
      data/arcmof/mc-clusters.csv / flig-clusters.csv / func-clusters.csv
    输出：outputs/null_elite_permutation.csv、outputs/null_topology_cardinality.csv

✅ 运行状态（2026-10-11）：两项对照**已实际运行**（b = 10,000，粗/细两个粒度），
输出见 outputs/null_elite_permutation.csv、outputs/null_topology_cardinality.csv
（细粒度版本在 outputs/_fine/）。主稿相关段落已据此更新。

⚠️ 实现须知（复现者与审稿人请注意）：
1. 对照 ① 的**置换单位是样本（行）**：逐行重排拓扑标签、保持拓扑样本数不变。
   若改用「组成」为置换单位，单拓扑占比在置换下恒等于观测值（平凡），两种单位
   给出相反结论——主稿已写明所用单位。
2. 三个簇文件的实际列名是 filename / cluster_id / Database（2026-10-11 按实数据
   核对；原稿假设的 cluster_mc/flig/func 不存在），合并键为 filename ↔
   all_topology_lists.csv 的 Name 列。合并后 mc/flig 各缺 7,874 行、func 缺 49,682 行，
   单拓扑占比只在三个簇都齐的样本上计算（与主稿表口径一致）。
3. 对照 ② 的基准是「按该来源拓扑数、从全库抽取等规模样本」10 次抽样；来源筛选
   为 n ≥ 200（与主稿 §3.15 一致）。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# 与 DATA.md / 主稿 §2.2 一致的口径
COARSE_COLS = ["metal_linker", "organic_linker1", "organic_linker2"]
FINE_COLS = COARSE_COLS + ["functional_groups"]
ARC_COARSE = ["mc", "flig"]
ARC_FINE = ["mc", "flig", "func"]

B_DEFAULT = 10_000
B_CARD_DEFAULT = 1_000   # 对照 ② 的来源内置换次数（DB0 规模下的运行时间上限）
SEED = 42


def single_topology_fraction(df, comp_cols, topo_col="topology"):
    """按组成等权的单拓扑占比（与 full_db_confound.py 同口径）。

    只使用组成列完整的样本（与主稿各表的 dropna 口径一致；ARC-MOF 的三个簇
    文件并非覆盖全部行，若把缺失当成一个组成值会虚增 M）。
    """
    d = df.dropna(subset=list(comp_cols))
    comp = d[comp_cols].astype(str).agg("|".join, axis=1)
    nunique = d.groupby(comp)[topo_col].nunique()
    return (nunique == 1).mean(), len(nunique)


def control_1_elite_permutation(df_elite, comp_cols, b=B_DEFAULT, seed=SEED,
                                level="row"):
    """对照 ①：保持每拓扑样本数不变，置换拓扑标签的零分布。

    参数
    ----
    level : "row"（默认）或 "composition"
      - "row"       ：逐行重排拓扑标签，保持拓扑的样本数（边缘分布）不变。
      - "composition"：在组成级别重排：每个组成保持其样本数不变，但将其拓扑标签
                       随机设为该组成在观测数据中实际出现过的拓扑之一（按样本数
                       加权抽样）。这会同时打破边缘分布，因此 p 值不可直接比较，
                       只用于量化组成级聚类对单拓扑占比的贡献。
    """
    d0 = df_elite.dropna(subset=list(comp_cols))
    comp = d0[comp_cols].astype(str).agg("|".join, axis=1).to_numpy()
    topo = d0["topology"].to_numpy()
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({"comp": comp, "topology": topo})
    obs_nun = base.groupby("comp", sort=False)["topology"].nunique()
    obs, m_obs = (obs_nun == 1).mean(), len(obs_nun)

    null = np.empty(b)
    for i in range(b):
        if level == "row":
            base["topology"] = rng.permutation(topo)
        elif level == "composition":
            # 每个组成从其观测拓扑中按样本数加权随机抽取一个
            comp_topo = base.groupby("comp")["topology"].value_counts()
            picked = {}
            for c in base["comp"].unique():
                idx = comp_topo[c].index
                prob = comp_topo[c].values / comp_topo[c].values.sum()
                picked[c] = rng.choice(idx, p=prob)
            base["topology"] = base["comp"].map(picked)
        else:
            raise ValueError(f"Unknown level: {level}")
        nun = base.groupby("comp", sort=False)["topology"].nunique()
        null[i] = (nun == 1).mean()
    return {"observed": obs, "M": m_obs, "level": level,
            "null_mean": null.mean(), "null_sd": null.std(),
            "null_q025": np.quantile(null, 0.025),
            "null_q975": np.quantile(null, 0.975),
            "p_one_sided": (null >= obs).mean()}


def control_2_topology_cardinality(df_full, comp_cols, topo_col="topology",
                                   b=B_CARD_DEFAULT, seed=SEED, min_n=200):
    """对照 ②：来源内拓扑标签置换零模型（按拓扑基数条件化）。

    2026-10-11 重写。原实现是「随机挑 n_topo_src 种拓扑后取其全部行」作为基准，
    有两个问题：(i) 挑出的池常远小于该来源的样本量（DB1：池 353 行 vs 观测
    16,565 行），基准与观测不可比；(ii) 某些来源（DB6，仅 1 种拓扑）池为空，
    触发 pandas 的 agg→DataFrame 退化而崩溃。原版本的输出**不可引用**。

    现实现按 SI §S3 的规格：对每个来源子库，保持其**化学组成结构**（每个组成
    的样本数）与**拓扑样本数边缘分布**不变，逐样本置换拓扑标签 b 次，得到
    「该来源的规模与拓扑丰富度下、化学-拓扑随机关联」的零分布：
      - observed 显著低于零分布 → 该来源的化学-拓扑耦合确实更松（不是拓扑
        基数 H(T) 的机械投影）；
      - observed ≈ 零分布 → 该来源的排他率可由其拓扑丰富度解释。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for src, g in df_full.groupby("source"):
        if len(g) < min_n:  # 与主稿口径一致：n >= 200
            continue
        d = g.dropna(subset=list(comp_cols))
        if len(d) == 0:
            continue
        comp = d[comp_cols].astype(str).agg("|".join, axis=1)
        if isinstance(comp, pd.DataFrame):  # 空表/重复列时 agg 会退化为 DataFrame
            comp = comp.iloc[:, 0]
        topo = d[topo_col].to_numpy()
        base = pd.DataFrame({"comp": comp.to_numpy(), "topology": topo})
        obs_nun = base.groupby("comp", sort=False)["topology"].nunique()
        obs = float((obs_nun == 1).mean())
        null = np.empty(b)
        for i in range(b):
            base["topology"] = rng.permutation(topo)
            nun = base.groupby("comp", sort=False)["topology"].nunique()
            null[i] = (nun == 1).mean()
        rows.append({
            "source": src, "n": len(d), "M": len(obs_nun),
            "n_topologies": int(pd.Series(topo).nunique()),
            "observed": obs,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "p_lower": float((null <= obs).mean()),
            "b": b,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         os.pardir, "data"),
                    help="数据目录；默认取本脚本所在目录的 ../data（HERE 相对），"
                         "因此从任何工作目录运行都成立（2026-10-14 修：原默认值 "
                         "'../data' 是 CWD 相对的，只有 cd scripts 后才正确）")
    ap.add_argument("--outdir",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         os.pardir, "outputs"),
                    help="输出目录；同上，默认 HERE 相对")
    ap.add_argument("--granularity", choices=["coarse", "fine"], default="coarse")
    ap.add_argument("-b", type=int, default=B_DEFAULT,
                    help="对照 ① 的置换次数")
    ap.add_argument("--b-card", type=int, default=B_CARD_DEFAULT,
                    help="对照 ② 的来源内置换次数")
    ap.add_argument("--require-cols", default="",
                    help="逗号分隔列名；给定则 ARC-MOF 对照只在这些列完整的行上计算，"
                         "使 coarse/fine 共享同一支撑集（与主稿 §3.15 来源表口径一致："
                         "该表的 coarse 列也是在 mc/flig/func 三列齐全的行上算的）。"
                         "例如 --require-cols mc,flig,func")
    args = ap.parse_args()

    elite_path = os.path.join(args.data_dir, "top_MOFs_screening_data.csv")
    if not os.path.exists(elite_path):
        sys.exit(
            f"找不到精英子集数据 {elite_path}。请按 DATA.md 第 1 节下载 "
            "Materials Cloud 记录 nsp7v-2dk91 并校验 MD5 后重跑。")

    comp_cols = COARSE_COLS if args.granularity == "coarse" else FINE_COLS
    df_elite = pd.read_csv(elite_path, low_memory=False)

    os.makedirs(args.outdir, exist_ok=True)
    res1 = control_1_elite_permutation(df_elite, comp_cols, b=args.b)
    print("对照 ① 精英子集置换零模型（{} 粒度）:".format(args.granularity))
    for k, v in res1.items():
        print(f"  {k}: {v}")
    pd.DataFrame([res1]).to_csv(
        os.path.join(args.outdir, "null_elite_permutation.csv"), index=False)

    # 对照 ② 需要 ARC-MOF 元数据（四个文件，MD5 见 DATA.md）
    arcmof_top = os.path.join(args.data_dir, "arcmof", "all_topology_lists.csv")
    arcmof_mc = os.path.join(args.data_dir, "arcmof", "mc-clusters.csv")
    arcmof_flig = os.path.join(args.data_dir, "arcmof", "flig-clusters.csv")
    arcmof_func = os.path.join(args.data_dir, "arcmof", "func-clusters.csv")
    if not all(os.path.exists(p) for p in (arcmof_top, arcmof_mc, arcmof_flig, arcmof_func)):
        print("对照 ② 跳过：ARC-MOF 元数据未就位（按 DATA.md 第 2 节下载）。")
        return

    top = pd.read_csv(arcmof_top, low_memory=False)
    # all_topology_lists.csv 前两列名字互换（DATA.md 陷阱 1）：
    # 检测哪一列带 .cif 后缀并重命名
    cols = list(top.columns[:2])
    if top[cols[1]].astype(str).str.endswith(".cif").mean() > 0.5:
        top = top.rename(columns={cols[0]: "filename", cols[1]: "name"})
    else:
        top = top.rename(columns={cols[0]: "name", cols[1]: "filename"})
    top["source"] = top["name"].astype(str).str.split("-").str[0]
    # Crystalnet 列名在 ARC-MOF 元数据中可能有变体，这里做显式断言：
    topo_col = "Crystalnet" if "Crystalnet" in top.columns else "crystalnet"
    assert topo_col in top.columns, "请核对 all_topology_lists.csv 的拓扑列名"

    # 三个簇文件的实际列名是 filename / cluster_id / Database（2026-10-11 按实数据核对，
    # 原稿假设的 cluster_mc / cluster_flig / cluster_func 并不存在，会导致合并崩溃）。
    # 合并键是该文件的 filename（CIF 文件名），对应 all_topology_lists.csv 的 Name 列——
    # 该列在前两列互换陷阱修正后已被重命名为 name。簇列逐个改名以免同名冲突。
    def _load_cluster(path, out_col):
        d = pd.read_csv(path, low_memory=False)
        missing = [c for c in ("filename", "cluster_id") if c not in d.columns]
        assert not missing, (
            f"{os.path.basename(path)} 缺少列 {missing}；实际列为 {list(d.columns)}")
        d = d.rename(columns={"filename": "name", "cluster_id": out_col})[["name", out_col]]
        return d.drop_duplicates("name")

    df = (top.merge(_load_cluster(arcmof_mc, "mc"), on="name", how="left")
             .merge(_load_cluster(arcmof_flig, "flig"), on="name", how="left")
             .merge(_load_cluster(arcmof_func, "func"), on="name", how="left"))
    print(f"ARC-MOF 合并完成：{len(df)} 行，"
          f"mc/flig/func 缺失 {df[['mc','flig','func']].isna().sum().to_dict()}")
    if args.require_cols:
        need = [c.strip() for c in args.require_cols.split(",") if c.strip()]
        df = df.dropna(subset=need)
        print(f"已按 --require-cols {need} 限制支撑集：剩余 {len(df)} 行")

    arc_comp = ARC_COARSE if args.granularity == "coarse" else ARC_FINE
    res2 = control_2_topology_cardinality(
        df, arc_comp, topo_col=topo_col, b=args.b_card)
    print("对照 ② 来源内拓扑标签置换零模型（按拓扑基数条件化）：")
    print(res2.to_string(index=False))
    res2.to_csv(os.path.join(args.outdir, "null_topology_cardinality.csv"), index=False)


if __name__ == "__main__":
    main()
