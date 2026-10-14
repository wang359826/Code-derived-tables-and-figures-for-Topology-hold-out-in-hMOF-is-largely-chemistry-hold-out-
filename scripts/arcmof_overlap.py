#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ARC-MOF 与 hMOF 的**结构级重叠检验**（回应评审意见 R9）
=========================================================

背景（必须先承认的事实）
------------------------
Zenodo 记录 10.5281/zenodo.13891643 的官方描述写着 ARC-MOF
"spanning all publicly available MOF databases"，作者名单里包含
**Boyd, Peter G.** 与 **Woo, Tom K.**（hMOF 的作者），v6 更新说明更直接写着
"Removed of all m29 structures from the Boyd-Woo database"。

也就是说：**ARC-MOF 并不独立于 hMOF**，我们原先"文件名交集 = 0 → 独立数据库"
的论证**没有证明力**（只是重命名而已）。

本脚本把这件事从"猜"变成"量"：
------------------------------------
hMOF      : str_m5_o16_o16_sra_sym.77   → 归一化 → m5_o16_o16_sra
ARC-MOF   : DB0-m12_o10_bcu.cif         → 归一化 → m12_o10_bcu

两者使用**同一套命名约定**（金属 mX + 有机连接体 oX + 拓扑），
因此可以直接求交集，得到：
  1. 有多少 ARC-MOF 条目能在 hMOF 里逐名对上（即确定来自同一来源库）
  2. 这些条目集中在哪些 DB<n> 子库里 → 反推哪个子库是 hMOF 来源
  3. 剔除这些条目后，§3.14 的混淆结论是否仍然成立（留一来源 / 剔除同源）

产出
----
  outputs/arcmof_overlap.csv             总体 + 按来源子库的重叠统计
  outputs/confound_arcmof_excl_derived.csv  剔除已识别同源条目后的混淆统计
                                            （**coarse 与 fine 两套粒度都给**，
                                              带 `granularity` 列；正文 Table 20）

用法
----
  python arcmof_overlap.py
  python arcmof_overlap.py --exclude-source DB0    # 整库剔除某个子库再算
"""

from __future__ import annotations

import argparse
import os
import re
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

STRIP_HMOF = re.compile(r"^str_|_sym\.\d+$")
STRIP_ARC = re.compile(r"^DB\d+-|\.cif$")


def norm_hmof(s: str) -> str:
    return STRIP_HMOF.sub("", str(s)).strip()


def norm_arc(s: str) -> str:
    return STRIP_ARC.sub("", str(s)).strip()


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def confound_stats(names, topo, chem, label, n_topos_cap=None):
    """单拓扑化学占比 + 条件熵；返回 dict（与 arcmof_confound.py 口径一致）。"""
    n = len(names)
    if n == 0:
        return {"subset": label, "n_samples": 0, "n_chem": 0,
                "frac_single_topology": np.nan, "H_T": np.nan,
                "H_T_given_C": np.nan, "cond_frac": np.nan,
                "n_topologies": 0}
    # ⚠️ 坑：pd.Series(另一个 Series, index=别的索引) 会**重新索引**成全 NaN。
    #    必须先转成 ndarray 再套索引，否则 n_chem=0、统计量全 NaN 却照样跑完。
    ck = pd.Series(np.asarray(chem, dtype=object),
                   index=np.asarray(names, dtype=object)).astype(str)
    tp = pd.Series(np.asarray(topo, dtype=object),
                   index=np.asarray(names, dtype=object)).astype(str)
    assert ck.notna().all() and tp.notna().all(), "化学/拓扑列出现空值"

    d = pd.DataFrame({"ck": ck.to_numpy(), "tp": tp.to_numpy()})
    n_chem = d["ck"].nunique()
    # ⚠️ 不要用 groupby(...).apply()：18 万个化学组会把 pandas 拖死（实测挂起）。
    #    全部走向量化：先按 (化学, 拓扑) 聚合，再算熵。
    pair = d.groupby(["ck", "tp"]).size().reset_index(name="n_ct")
    ntop = pair.groupby("ck")["tp"].nunique()
    single = float((ntop == 1).mean())
    p_t = d["tp"].value_counts(normalize=True).to_numpy()
    HT = entropy(p_t)
    # H(T|C) = Σ_c Σ_t p(c,t) · (−log2 p(t|c))
    n_c = pair.groupby("ck")["n_ct"].transform("sum")
    ptc = pair["n_ct"] / n_c
    hc = float((pair["n_ct"] / n * (-np.log2(ptc))).sum())
    return {"subset": label, "n_samples": n, "n_chem": int(n_chem),
            "frac_single_topology": single, "H_T": HT,
            "H_T_given_C": float(hc),
            "cond_frac": float(hc / HT) if HT > 0 else np.nan,
            "n_topologies": int(tp.nunique())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hmof-csv",
                    default=os.path.join(HERE, "..", "data",
                                         "all_MOFs_screening_data.csv"))
    ap.add_argument("--arc-topo",
                    default=os.path.join(ARCDIR, "all_topology_lists.csv"))
    ap.add_argument("--mc", default=os.path.join(ARCDIR, "mc-clusters.csv"))
    ap.add_argument("--flig", default=os.path.join(ARCDIR,
                                                   "flig-clusters.csv"))
    ap.add_argument("--func", default=os.path.join(ARCDIR,
                                                   "func-clusters.csv"))
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("ARC-MOF ↔ hMOF 结构级重叠检验（R9）")
    print("=" * 78)

    # ---------- 1. 名字归一化 ----------
    hm = pd.read_csv(args.hmof_csv, usecols=["MOFname"], encoding="utf-8")
    hm_names = set(norm_hmof(v) for v in hm["MOFname"])
    print(f"hMOF 结构名（归一化后）: {len(hm_names):,} 个唯一名 / "
          f"{len(hm):,} 行")

    ar = pd.read_csv(args.arc_topo, encoding="utf-8")
    # ⚠️ 列名与内容对调：Name 列才是文件名，filename 列是"文件名里报告的拓扑"
    ar = ar.rename(columns={"Name": "struct", "filename": "topo_from_name",
                            "Crystalnet": "topo_crystalnet",
                            "likely topology": "topo_likely"})
    if "struct" not in ar.columns:
        raise SystemExit(f"列名不符，实际为 {ar.columns.tolist()}")
    ar["db"] = ar["struct"].astype(str).str.extract(r"^(DB\d+)-")[0]
    ar["norm"] = ar["struct"].map(norm_arc)
    print(f"ARC-MOF 条目: {len(ar):,}  来源子库: "
          f"{ar['db'].nunique()} 个")

    ar["in_hmof"] = ar["norm"].isin(hm_names)
    n_match = int(ar["in_hmof"].sum())
    print(f"\n★ 名字可对上的条目: {n_match:,} / {len(ar):,} "
          f"({n_match / len(ar) * 100:.1f}%)")

    # ---------- 2. 按来源子库分解 ----------
    per_db = (ar.groupby("db")
                .agg(n=("struct", "size"),
                     n_matched=("in_hmof", "sum"))
                .reset_index())
    per_db["pct_matched"] = per_db["n_matched"] / per_db["n"] * 100
    per_db["pct_of_arc"] = per_db["n"] / len(ar) * 100
    per_db = per_db.sort_values("n", ascending=False)
    print("\n按来源子库（DB<n>）分解：")
    print(per_db.to_string(index=False,
                           formatters={"pct_matched": "{:.1f}".format,
                                       "pct_of_arc": "{:.1f}".format}))

    p1 = os.path.join(args.outdir, "arcmof_overlap.csv")
    per_db.to_csv(p1, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p1}")

    # ---------- 3. 剔除同源条目后，混淆结论是否还成立 ----------
    print("\n" + "=" * 78)
    print("剔除已识别的同源条目后重算混淆（§3.14 结论是否还站得住）")
    print("=" * 78)

    def load_clusters(path, col):
        d = pd.read_csv(path, encoding="utf-8")
        d = d.rename(columns={"filename": "struct"})
        return d.set_index("struct")[col] if col in d.columns else None

    mc = load_clusters(args.mc, "cluster_id")
    fl = load_clusters(args.flig, "cluster_id")
    fu = load_clusters(args.func, "cluster_id")
    if mc is None or fl is None or fu is None:
        print("⚠️ 聚类文件列名不符，跳过第 3 步（只用拓扑做粗算）。")
        return

    # 两套「化学组成」定义：coarse = mc + flig；fine = coarse + func。
    # §3.15 的 Table 20 现在**两套粒度都给**（原来只有 fine，正文因此写着
    # "no coarse-granularity bound reported"）。
    granularities = {
        "fine": (mc.astype(str) + "|" + fl.astype(str) + "|" + fu.astype(str)),
        "coarse": (mc.astype(str) + "|" + fl.astype(str)),
    }

    rows = []
    for gran, chem in granularities.items():
        ar2 = ar.copy()
        ar2["chem"] = ar2["struct"].map(chem)
        ar2 = ar2.dropna(subset=["chem", "topo_crystalnet"])
        print(f"\n[{gran}] 可用于混淆统计的条目: {len(ar2):,}")

        tp = ar2["topo_crystalnet"].astype(str)
        ch = ar2["chem"].astype(str)
        idx = ar2["struct"].astype(str)

        block = [confound_stats(idx, tp, ch, "全部（未剔除）")]
        block.append(confound_stats(idx[~ar2["in_hmof"]], tp[~ar2["in_hmof"]],
                                    ch[~ar2["in_hmof"]],
                                    "剔除可对上名字的条目"))
        # 逐个剔除子库的敏感性
        for db in per_db["db"].dropna().tolist():
            m = ar2["db"] != db
            if m.sum() < 200:
                continue
            block.append(confound_stats(idx[m], tp[m], ch[m], f"剔除 {db}"))
        for r in block:
            r["granularity"] = gran
        rows.extend(block)

    d = pd.DataFrame(rows)
    print("\n" + d.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    p2 = os.path.join(args.outdir, "confound_arcmof_excl_derived.csv")
    d.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p2}")

    print("\n" + "=" * 78)
    print("怎么读（写进正文的版本）")
    print("=" * 78)
    print("  · 若重叠条目集中在某一个 DB<n>，说明 hMOF 来源的条目是**成块**进入")
    print("    ARC-MOF 的。此时'剔除该子库'这一行就是结论的稳健性上界：")
    print("    只要剔除后单拓扑占比仍然很高，§3.14 就不依赖 hMOF 自身的结构。")
    print("  · 若重叠条目分散在多个子库，则 ARC-MOF 的构造方式与 hMOF 纠缠更深，")
    print("    '跨数据库复现'必须降级为'同一来源谱系内的第二个库'。")


if __name__ == "__main__":
    main()
