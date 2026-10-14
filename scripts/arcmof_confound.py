# -*- coding: utf-8 -*-
"""
把 hMOF 上的"化学–拓扑混淆"诊断复制到 ARC-MOF（第二个、且是**混合**数据库）。

动机
----
§3.2/§3.12 只在 hMOF 一个数据库上证明了"留出拓扑 ≡ 留出化学"。
最自然的反驳是："这是 hMOF 这种纯枚举库的构造缺陷，别推广。"
ARC-MOF 正好是最强的检验对象：它**混合**了实验库（CoRE MOF）与多个计算枚举库
（hMOF / BW-DB / ToBaCCo 等），共约 28 万个 MOF。
- 若混淆在 ARC-MOF 上同样严重 → 结论从"hMOF 一例"升级为"枚举数据库通病"；
- 若明显更轻 → 恰好量化出"混淆程度由数据库构造方式决定"，同样是结论。

数据
----
- all_topology_lists.csv : filename, filename拓扑, Crystalnet 拓扑, likely topology
- mc-clusters.csv        : 金属簇(metal chemistry) 聚类 id
- flig-clusters.csv      : 连接体(functionalized linker) 聚类 id
- func-clusters.csv      : 官能团(functional group) 聚类 id
  （cluster_id < 0 表示"不属于任何簇"，即被认为是独特结构 → 单独统计，不参与
    化学键构造，否则会把大量唯一值当成化学组成、人为夸大单拓扑占比）

口径纪律
--------
本脚本只做**描述性**统计（熵、集合重叠、单拓扑占比），不做任何 R² 比较，
因此不涉及"换支持集就换分母"的陷阱。所有比例都同时报"含唯一结构"与
"仅已归簇结构"两版，避免读者误读。
"""
import os
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "arcmof")
OUT = os.path.join(HERE, "..", "outputs")


def entropy(p):
    """香农熵（bit）。p 为各类的计数向量。"""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def confound_stats(df, chem_keys, topo_col, label):
    """核心诊断：给定化学键与拓扑列，算 4 个标量。"""
    d = df.dropna(subset=chem_keys + [topo_col]).copy()
    if len(d) == 0:
        return None
    chem = d[chem_keys].astype(str).agg("|".join, axis=1)
    topo = d[topo_col].astype(str)

    n_chem = chem.nunique()
    n_topo = topo.nunique()
    joint = pd.DataFrame({"c": chem, "t": topo})
    # 每个化学组成落在几种拓扑上
    per_chem = joint.groupby("c")["t"].nunique()
    frac_single = float((per_chem == 1).mean())

    H_T = entropy(topo.value_counts())
    # H(T|C) = Σ_c p(c) H(T|C=c)
    H_T_given_C = 0.0
    for c, sub in joint.groupby("c"):
        w = len(sub) / len(joint)
        H_T_given_C += w * entropy(sub["t"].value_counts())
    cond_frac = H_T_given_C / H_T if H_T > 0 else np.nan

    # 最大组主导度（提示 pooled 型聚合会被单组主导）
    share = topo.value_counts(normalize=True)
    return {
        "label": label,
        "topo_col": topo_col,
        "n_samples": len(d),
        "n_chem": n_chem,
        "n_topologies": n_topo,
        "frac_single_topology": frac_single,
        "H_T": H_T,
        "H_T_given_C": H_T_given_C,
        "cond_frac": cond_frac,
        "max_topo_share": float(share.max()),
        "top3_topo_share": float(share.head(3).sum()),
        "norm_entropy": H_T / np.log2(n_topo) if n_topo > 1 else np.nan,
        "top_chem_keys": "+".join(chem_keys),
    }


def jaccard_pairs(df, chem_keys, topo_col, min_size=20):
    """拓扑两两之间"化学集合"的 Jaccard 重叠。min_size 过滤过小的拓扑。"""
    d = df.dropna(subset=chem_keys + [topo_col]).copy()
    chem = d[chem_keys].astype(str).agg("|".join, axis=1)
    d = d.assign(_c=chem)
    counts = d[topo_col].value_counts()
    keep = counts[counts >= min_size].index.tolist()
    sets = {t: set(d.loc[d[topo_col] == t, "_c"]) for t in keep}
    rows = []
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            A, B = sets[a], sets[b]
            inter, union = len(A & B), len(A | B)
            rows.append({"topo_a": a, "topo_b": b,
                         "n_a": len(A), "n_b": len(B),
                         "intersection": inter,
                         "jaccard": inter / union if union else 0.0})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-topo-size", type=int, default=20,
                    help="Jaccard 时忽略小于该规模的拓扑")
    ap.add_argument("--top-n-topo", type=int, default=40,
                    help="只保留样本量前 N 的拓扑（控制热力图规模）")
    ap.add_argument("--out-tag", type=str, default="arcmof")
    args = ap.parse_args()

    print("载入 ARC-MOF 元数据 …")
    top = pd.read_csv(os.path.join(DATA, "all_topology_lists.csv"),
                      encoding="utf-8-sig", dtype=str)
    top.columns = [c.strip() for c in top.columns]
    # ⚠️ 该文件前两列的名字与内容对调了：
    #     'Name'    列 = CIF 文件名（如 DB0-m12_o10_bcu.cif）
    #     'filename'列 = 文件名里报告的拓扑（如 bcu）
    #   实测确认：'Name' 是唯一带 .cif 后缀的列。故重命名以避免误用。
    if "Name" in top.columns and top["Name"].astype(str).str.endswith(".cif").mean() > 0.5:
        top = top.rename(columns={"Name": "file"})
        top = top.rename(columns={"filename": "topology_in_filename"})
    print("  拓扑表:", top.shape, list(top.columns))

    frames = {}
    for name, key in [("mc-clusters.csv", "mc"),
                      ("flig-clusters.csv", "flig"),
                      ("func-clusters.csv", "func")]:
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            print(f"  [跳过] 缺少 {name}")
            continue
        f = pd.read_csv(p, encoding="utf-8-sig", dtype={"filename": str})
        f.columns = [c.strip() for c in f.columns]
        f = f.rename(columns={"cluster_id": key, "filename": "file"})
        f["file"] = f["file"].astype(str)
        f = f[["file", key]].drop_duplicates("file")
        frames[key] = f
        pos = (f[key] > 0).sum()
        print(f"  {name:20s} n={len(f):7d}  已归簇(>0)={pos:7d} "
              f"({pos/len(f)*100:5.1f}%)  唯一(<0)={len(f)-pos}")

    df = top.copy()
    df["file"] = df["file"].astype(str)
    for k, f in frames.items():
        df = df.merge(f, on="file", how="left")
    print("  合并后:", df.shape)
    for k in ["mc", "flig", "func"]:
        if k in df.columns:
            print(f"    {k} 匹配率: {df[k].notna().mean()*100:.1f}%")

    # ---- 拓扑列选择
    topo_candidates = [c for c in df.columns
                       if c.lower() in ("crystalnet", "likely topology",
                                        "topology_in_filename", "topology")]
    print("  可用拓扑列:", topo_candidates)

    # 文件名前缀当来源标记（DB0- / DB1- / DB12- …）
    import re
    def src(fn):
        m = re.match(r"^(DB\d+)", str(fn))
        return m.group(1) if m else "other"
    df["source"] = df["file"].map(src)
    print("  来源前缀 top12:",
          {k: int(v) for k, v in df["source"].value_counts().head(12).items()})

    rows = []
    reports = []
    for topo_col in topo_candidates:
        if topo_col == "file":
            continue
        for gran, keys in [("coarse(mc+flig)", ["mc", "flig"]),
                           ("fine(mc+flig+func)", ["mc", "flig", "func"])]:
            if not all(k in df.columns for k in keys):
                continue
            # 版本 A：所有结构（含唯一结构）
            r = confound_stats(df, keys, topo_col, f"{args.out_tag} ALL {gran}")
            if r:
                r["subset"] = "all"
                r["granularity"] = gran
                reports.append(r)
            # 版本 B：仅已归簇结构（三个 key 都 > 0）
            sub = df.copy()
            for k in keys:
                sub = sub[sub[k] > 0]
            r = confound_stats(sub, keys, topo_col, f"{args.out_tag} CLUSTERED {gran}")
            if r:
                r["subset"] = "clustered_only"
                r["granularity"] = gran
                reports.append(r)

    rep = pd.DataFrame(reports)
    os.makedirs(OUT, exist_ok=True)
    p_rep = os.path.join(OUT, f"confound_{args.out_tag}.csv")
    if rep.empty:
        print("\n[错误] 没有任何混淆统计产出——检查列名是否变化、"
              "或化学键列是否全为 NaN。不静默通过。")
        raise SystemExit(1)
    rep.to_csv(p_rep, index=False, encoding="utf-8-sig")
    print("\n=== 混淆统计 ===")
    cols = ["label", "subset", "granularity", "n_samples", "n_chem",
            "n_topologies", "frac_single_topology", "cond_frac", "max_topo_share"]
    print(rep[cols].to_string(index=False))
    print("saved:", p_rep)

    # ---- Jaccard（用最大的拓扑集合，clustered_only + coarse）
    best_topo = None
    for c in topo_candidates:
        if c != "filename":
            best_topo = c
            break
    if best_topo and all(k in df.columns for k in ["mc", "flig"]):
        sub = df[(df["mc"] > 0) & (df["flig"] > 0)].copy()
        counts = sub[best_topo].value_counts()
        keep = counts.head(args.top_n_topo).index
        sub = sub[sub[best_topo].isin(keep)]
        j = jaccard_pairs(sub, ["mc", "flig"], best_topo,
                          min_size=args.min_topo_size)
        if len(j):
            p_j = os.path.join(OUT, f"jaccard_{args.out_tag}.csv")
            j.to_csv(p_j, index=False, encoding="utf-8-sig")
            nz = j[j["jaccard"] > 0]
            print(f"\n=== Jaccard（拓扑列={best_topo}，保留 {len(keep)} 个拓扑）===")
            print(f"  拓扑对数 = {len(j)}，有重叠的对数 = {len(nz)}"
                  f"（{len(nz)/len(j)*100:.1f}%）")
            if len(nz):
                print(nz.sort_values("jaccard", ascending=False)
                      .head(15).to_string(index=False))
            print("saved:", p_j)

        # 按来源分层：同一诊断在各来源子集上重跑
        print("\n=== 按来源分层（coarse 粒度，clustered_only）===")
        rows2 = []
        for s, g in sub.groupby("source"):
            if len(g) < 200:
                continue
            r = confound_stats(g, ["mc", "flig"], best_topo, s)
            if r:
                r["source"] = s
                rows2.append(r)
        if rows2:
            sr = pd.DataFrame(rows2).sort_values("n_samples", ascending=False)
            p_s = os.path.join(OUT, f"confound_by_source_{args.out_tag}.csv")
            sr.to_csv(p_s, index=False, encoding="utf-8-sig")
            print(sr[["source", "n_samples", "n_topologies", "n_chem",
                      "frac_single_topology", "cond_frac"]].to_string(index=False))
            print("saved:", p_s)

    print("\n完成。")


if __name__ == "__main__":
    main()
