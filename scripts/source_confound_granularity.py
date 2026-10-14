# -*- coding: utf-8 -*-
"""
来源分层的混淆诊断（v2）：控制化学粒度后，ARC-MOF 各来源库的混淆程度是否仍有差异？

⚠️ v1 / v2 被否决的原因（不要重蹈）
-----------------------------------
**v1**：用「样本量加权排他率」且 `min_topo_size = max(20, n_source * 0.02)`
——按来源样本量取比例，导致 DB0(169k) 只剩 **5 个拓扑**、DB6 只剩 **1 个**
（排他率恒等于 1.0，纯属退化），而小来源保留 11 个。拓扑基数不可比，
排他率完全失真。

**v2**：改用 frac_single_topology（正确），但**粒度对齐用随机压缩，方法本身失效**：
随机压缩只能"把细的化学定义变粗"，不能反向；于是对齐点 r* 必须取
max(各来源 r)，各来源到 r* 的距离就不同——DB12 的 r 本就是 1.77，
**完全没被压缩**（0.9846 → 0.9846），而 DB0 被压了 1.48 倍
（0.9994 → 0.5219）。**扭曲程度不一致，对齐后的差异主要由"离 r* 多远"
驱动，而不是构造方式**。v2 报的"对齐后极差 406%"就是这样来的，已作废。

v3 的做法（本版）
-----------------
1. **主指标** frac_single_topology —— 与论文 Table 7 / Table 12 一致，
   不依赖拓扑过滤。
2. **化学粒度用真实的多级定义，不做随机压缩**：
   - coarse = mc + flig（金属中心 + 有机连接体聚类）
   - fine   = mc + flig + func（再加官能团）
   两级都是**有化学意义的真实定义**，不是随机重组。若结论在两级上一致，
   就不是粒度造成的。
3. **不确定性用 bootstrap（有放回重采样样本）**，而不是随机压缩化学 ID。
   bootstrap 不改变化学定义，只反映抽样变异，可合法地喂给 Kruskal-Wallis。
4. 随机压缩的结果**保留但降级为辅助参考**，并在输出里显式标注其局限。

设计
----
- 来源 = 文件名前缀 DB0/DB1/…（保留 n ≥ --min-source-n）
- 拓扑 = **Crystalnet**（全库统一列；绝不按来源切换拓扑列——那会让
  DB0 38 个 vs DB13 8 个，不可比）

用法
----
python source_confound_granularity.py --min-source-n 200 --boot 10
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kruskal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
DATA = os.path.join(HERE, "..", "data")


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def confound_metrics(chem, topo):
    """返回不受拓扑过滤影响的混淆指标。

    frac_single_topology : 化学加权的「只出现在单一拓扑的化学组成」占比
                           —— 论文主指标，跨库/跨来源可比
    H_T, H_T_given_C     : 拓扑熵与条件熵（bit）
    cond_frac            : H(T|C) / H(T)，越小表示化学几乎完全决定拓扑
    n_topologies         : 该子集的拓扑基数
    max_topo_share       : 最大拓扑的样本占比（拓扑多样性反向指标）
    """
    d = pd.DataFrame({"c": np.asarray(chem), "t": np.asarray(topo)}).dropna()
    if d.empty:
        return dict(frac_single_topology=np.nan, H_T=np.nan,
                    H_T_given_C=np.nan, cond_frac=np.nan,
                    n_topologies=0, max_topo_share=np.nan, n_chem=0)
    n_topo_per_chem = d.groupby("c")["t"].nunique()
    frac = float((n_topo_per_chem == 1).mean())
    share = d["t"].value_counts(normalize=True)
    H_T = entropy(share.to_numpy())
    H_T_given_C = 0.0
    for _, sub in d.groupby("c"):
        w = len(sub) / len(d)
        H_T_given_C += w * entropy(sub["t"].value_counts(normalize=True).to_numpy())
    return dict(frac_single_topology=frac, H_T=H_T, H_T_given_C=H_T_given_C,
                cond_frac=(H_T_given_C / H_T) if H_T > 0 else np.nan,
                n_topologies=int(d["t"].nunique()),
                max_topo_share=float(share.iloc[0]),
                n_chem=int(d["c"].nunique()))


def compress(chem_ids, m, seed):
    """把唯一化学 ID 随机压缩到 m 组。m >= 唯一数时直接返回原编码。"""
    codes, uniq = pd.factorize(chem_ids)
    if m >= len(uniq):
        return pd.Series(codes, index=chem_ids.index)
    rng = np.random.default_rng(seed)
    lut = rng.integers(0, m, size=len(uniq))
    return pd.Series(lut[codes], index=chem_ids.index)


def load():
    print("载入 ARC-MOF …")
    top = pd.read_csv(os.path.join(DATA, "arcmof", "all_topology_lists.csv"),
                      encoding="utf-8-sig", dtype=str)
    top.columns = [c.strip() for c in top.columns]
    # ⚠️ 该文件前两列名字与内容对调（实测：只有 'Name' 带 .cif 后缀）
    if "Name" in top.columns and top["Name"].astype(str).str.endswith(".cif").mean() > 0.5:
        top = top.rename(columns={"Name": "file", "filename": "topology_in_filename"})
    else:
        top = top.rename(columns={"filename": "topology_in_filename"})

    for name, key in [("mc-clusters.csv", "mc"), ("flig-clusters.csv", "flig"),
                      ("func-clusters.csv", "func")]:
        f = pd.read_csv(os.path.join(DATA, "arcmof", name),
                        encoding="utf-8-sig", dtype={"filename": str})
        f.columns = [c.strip() for c in f.columns]
        f = f.rename(columns={"cluster_id": key, "filename": "file"})
        f["file"] = f["file"].astype(str)
        f = f[["file", key]].drop_duplicates("file")
        top = top.merge(f, on="file", how="left")
        if len(top[key].dropna()) == 0:
            raise SystemExit(f"✗ {name} 合并后全为空——文件列名可能又变了")

    top["source"] = top["file"].astype(str).str.extract(r"^(DB\d+)")[0].fillna("other")
    # 统一拓扑列 = Crystalnet（全库一致；绝不按来源切换拓扑列）
    top["topo"] = top["Crystalnet"].astype(str).str.strip()
    top = top[~top["topo"].isin(["", "nan", "None", "NaN"])]
    top = top.dropna(subset=["mc", "flig", "func"])
    top["chem"] = (top[["mc", "flig", "func"]].astype("int64").astype(str)
                   .agg("|".join, axis=1))
    if top.empty:
        raise SystemExit("✗ 有效样本为空")
    print(f"  有效样本（拓扑+化学均非空）: {len(top):,}")
    print(f"  Crystalnet 拓扑总数: {top['topo'].nunique()}")
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-source-n", type=int, default=200)
    ap.add_argument("--boot", type=int, default=10,
                    help="每个来源的 bootstrap 次数（用于 Kruskal-Wallis）")
    ap.add_argument("--keep-compress", action="store_true",
                    help="同时跑随机压缩对齐（已知不稳健，仅作辅助参考）")
    args = ap.parse_args()

    top = load()
    counts = top["source"].value_counts()
    srcs = [s for s, c in counts.items() if c >= args.min_source_n and s != "other"]
    print(f"\n保留来源（n ≥ {args.min_source_n}）: {srcs}\n")

    # ---------- 真实的两种化学粒度 ----------
    rows = []
    boot_vals = {}
    rng = np.random.default_rng(42)
    for s in srcs:
        sub = top[top["source"] == s]
        rec = {"source": s, "n_samples": len(sub)}
        for gname, cols in [("coarse", ["mc", "flig"]),
                            ("fine", ["mc", "flig", "func"])]:
            chem = (sub[cols].astype("int64").astype(str).agg("|".join, axis=1))
            m = confound_metrics(chem, sub["topo"])
            rec[f"{gname}_n_chem"] = m["n_chem"]
            rec[f"{gname}_frac_single"] = m["frac_single_topology"]
            rec[f"{gname}_cond_frac"] = m["cond_frac"]
            rec[f"{gname}_r"] = len(sub) / max(m["n_chem"], 1)
            if gname == "fine":
                rec["n_topologies"] = m["n_topologies"]
                rec["H_T"] = m["H_T"]
                rec["max_topo_share"] = m["max_topo_share"]
        # bootstrap（只改样本，不改化学定义）
        bv = []
        chem_f = (sub[["mc", "flig", "func"]].astype("int64").astype(str)
                  .agg("|".join, axis=1)).to_numpy()
        topo_f = sub["topo"].to_numpy()
        n = len(sub)
        for _ in range(args.boot):
            idx = rng.integers(0, n, size=n)
            bv.append(confound_metrics(chem_f[idx], topo_f[idx])["frac_single_topology"])
        boot_vals[s] = bv
        rec["boot_mean"] = float(np.mean(bv))
        rec["boot_sd"] = float(np.std(bv, ddof=1)) if len(bv) > 1 else 0.0
        rows.append(rec)
        print(f"  {s:5s} n={len(sub):7d}  topo={rec['n_topologies']:4d} | "
              f"coarse r={rec['coarse_r']:5.2f} 单拓扑={rec['coarse_frac_single']:.4f} | "
              f"fine r={rec['fine_r']:5.2f} 单拓扑={rec['fine_frac_single']:.4f} "
              f"(boot {rec['boot_mean']:.4f}±{rec['boot_sd']:.4f})")
    out = pd.DataFrame(rows)

    # ---------- 随机压缩（辅助，已知不稳健） ----------
    if args.keep_compress:
        r_star = float(out["fine_r"].max())
        print(f"\n[辅助] 随机压缩对齐 r* = {r_star:.3f}"
              f"（⚠️ 各来源被压缩的比例不同，结果不可作为主结论）")
        comp = {}
        for s in srcs:
            sub = top[top["source"] == s]
            chem = (sub[["mc", "flig", "func"]].astype("int64").astype(str)
                    .agg("|".join, axis=1))
            n_chem = int(out.loc[out["source"] == s, "fine_n_chem"].iloc[0])
            mm = int(min(round(len(sub) / r_star), n_chem))
            vals = [confound_metrics(compress(chem, mm, sd), sub["topo"])
                    ["frac_single_topology"] for sd in range(4)]
            comp[s] = float(np.mean(vals))
        out["compress_aligned"] = out["source"].map(comp)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "source_confound_granularity.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print("\n已保存:", p)

    # ---------- 判读 1：两级真实粒度 ----------
    print("\n" + "=" * 76)
    print("判读 1：两种**真实**化学粒度下，来源之间的混淆程度")
    print("=" * 76)
    print(out[["source", "n_samples", "n_topologies",
               "coarse_r", "coarse_frac_single",
               "fine_r", "fine_frac_single", "boot_sd"]]
          .round(4).to_string(index=False))
    for g in ["coarse", "fine"]:
        v = out[f"{g}_frac_single"]
        print(f"\n  {g:6s}: 范围 {v.min():.4f} ~ {v.max():.4f}  极差 {v.max()-v.min():.4f}")
    # 两级是否一致（同一个来源在两级下的排名是否相同）
    sr, sp = spearmanr(out["coarse_frac_single"], out["fine_frac_single"])
    print(f"\n  两级之间 Spearman ρ = {sr:+.3f} (p = {sp:.4f})")
    print("  → 若显著为正：混淆程度是来源的**稳定属性**，不随化学粒度定义而变。")

    groups = [v for v in boot_vals.values() if len(v) >= 2]
    if len(groups) >= 3:
        H, pv = kruskal(*groups)
        print(f"\n  Kruskal-Wallis（来源间，bootstrap 作重复）: H = {H:.2f}, p = {pv:.3g}")

    # ---------- 判读 2：能不能被粒度解释 ----------
    print("\n" + "=" * 76)
    print("判读 2：来源间的差异能否被「化学粒度」解释？（关键对照）")
    print("=" * 76)
    for g in ["coarse", "fine"]:
        x = out[f"{g}_r"].to_numpy(dtype=float)
        y = out[f"{g}_frac_single"].to_numpy(dtype=float)
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() >= 4 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
            pr, pp = pearsonr(x[m], y[m])
            sr2, sp2 = spearmanr(x[m], y[m])
            print(f"  {g:6s}: r(样本/化学) vs 单拓扑占比  "
                  f"Pearson {pr:+.3f} (p={pp:.4f}) | Spearman {sr2:+.3f} (p={sp2:.4f})")
    print("  → 若均不显著：来源间混淆差异**不是**粒度造成的，"
          "而是各子库构造方式的属性。")

    # ---------- 判读 3：与拓扑多样性 ----------
    print("\n" + "=" * 76)
    print("判读 3：混淆程度 vs 拓扑多样性")
    print("=" * 76)
    for xcol, xlab in [("n_topologies", "拓扑基数"),
                       ("max_topo_share", "最大拓扑占比"),
                       ("H_T", "拓扑熵 H(T)")]:
        x = out[xcol].to_numpy(dtype=float)
        y = out["fine_frac_single"].to_numpy(dtype=float)
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() >= 4 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
            pr, pp = pearsonr(x[m], y[m])
            sr3, sp3 = spearmanr(x[m], y[m])
            print(f"  fine 单拓扑占比 vs {xlab:8s}: "
                  f"Pearson {pr:+.3f} (p={pp:.4f}) | Spearman {sr3:+.3f} (p={sp3:.4f})")
    print("  ⚠️ 该相关含机械成分（拓扑基数与单拓扑占比都由同一份数据算出），")
    print("     只能作为方向性提示，不能当因果证据写入正文。")


if __name__ == "__main__":
    main()
