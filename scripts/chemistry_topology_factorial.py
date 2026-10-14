#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
判别实验：跨拓扑外推失败，到底是「拓扑本身」还是「化学新颖性」造成的？
======================================================================

为什么要做这个实验（2026-10-09 新增）
------------------------------------
论文的标题/核心主张是 “geometry descriptors cannot extrapolate across topology”
（几何描述符无法跨拓扑外推）。但 hMOF 里「拓扑」和「化学组成」高度纠缠：
留出一种拓扑时，测试样本的连接体化学通常也是没见过的。
所以我们观察到的 R² 崩塌，可能根本不是「拓扑」造成的，而是「新化学」造成的。
两者给出的结论完全不同，必须拆开。

拆法（两个互补设计）
--------------------
【设计 1｜同一模型、同一份数据、只看测试子集】
    LOTO 留出拓扑 T，模型照常训练。把 T 的测试样本按
    「该样本的化学组成是否在训练集中出现过」分成两堆：
        - chem_seen   ：化学见过，只有拓扑是新的一→ 纯粹的拓扑迁移
        - chem_novel  ：化学也没见过                一→ 化学 + 拓扑双重迁移
    分别算 R²。**同一个模型、同一次训练**，只是测试子集不同，可比性最强。

【设计 2｜把整个数据池限制在「跨多拓扑的化学组成」上】
    只保留那些 chemistry key 同时出现在 ≥2 种拓扑里的样本（池 B）。
    在池 B 上做 LOTO：因为每个化学组成至少跨 2 种拓扑，
    而每次只留出 1 种拓扑，所以**测试集的化学组成在训练集里必然出现过**。
    于是池 B 上的 LOTO 是「化学 100% 见过、拓扑 100% 没见过」的干净迁移。
    把它和「全池 A 上的 LOTO（化学可能没见过）」对比：
        ΔR²(B) 明显小于 ΔR²(A)  →  罪魁祸首是化学新颖性，标题必须改
        ΔR²(B) ≈ ΔR²(A)         →  拓扑本身确实是壁垒，标题可以保留

指标口径（2026-10-09 修正，重要）
--------------------------------
本脚本一律同时报告四种聚合口径，禁止再拿 pooled 冒充「逐组代表」：
    pooled   : 把所有折的预测拼起来算一次 R²（会被大组主导）
    macro    : 逐组 R² 的简单平均（每组权重相同）
    weighted : 逐组 R² 按样本量加权平均
    median   : 逐组 R² 的中位数
四者差异本身就是结果的一部分，必须写进论文。

用法
----
  python chemistry_topology_factorial.py --max-rows 50000 --outdir outputs
  python chemistry_topology_factorial.py --chem-keys fine --max-rows 50000
  python chemistry_topology_factorial.py --target "CO2/N2_selectivity" --log-target

依赖：与 mof_leakage_benchmark.py 放在同一目录，复用它的特征工程。
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mof_leakage_benchmark import (  # noqa: E402
    TARGET, TARGET_SELECTIVITY, RANDOM_STATE,
    build_features, load_data, apply_log10,
)

CHEM_KEYS_COARSE = ["metal_linker", "organic_linker1", "organic_linker2"]
CHEM_KEYS_FINE = ["metal_linker", "organic_linker1",
                  "organic_linker2", "functional_groups"]


# --------------------------------------------------------------------------
# 聚合口径：四个一律全报
# --------------------------------------------------------------------------
def aggregate(per_group_records):
    """per_group_records: [(group_name, n, R2), ...] -> dict(四种口径)。"""
    d = pd.DataFrame(per_group_records, columns=["group", "n", "R2"])
    if len(d) == 0:
        return {}
    return {
        "macro": float(d["R2"].mean()),
        "weighted": float(np.average(d["R2"], weights=d["n"])),
        "median": float(d["R2"].median()),
        "min": float(d["R2"].min()),
        "max": float(d["R2"].max()),
        "n_groups": int(len(d)),
        "frac_negative": float((d["R2"] < 0).mean()),
        "detail": d,
    }


def _fit_predict(Xtr, ytr, Xte, n_estimators, seed):
    m = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                              n_jobs=-1)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def run_kfold(X, y, n_splits, seed, n_estimators):
    """随机划分（KFold），返回 pooled + 逐折。"""
    yt, yp, per = [], [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(kf.split(X)):
        p = _fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[te], n_estimators, seed)
        yt.extend(y.iloc[te].tolist())
        yp.extend(p.tolist())
        per.append((f"fold{i}", len(te), r2_score(y.iloc[te].tolist(), p.tolist())))
    return {"pooled": r2_score(yt, yp),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp))}, per


def run_group_kfold(X, y, groups, n_splits, seed, n_estimators):
    """按组划分（GroupKFold），返回 pooled + 逐折。"""
    n_splits = max(2, min(n_splits, groups.nunique()))
    yt, yp, per = [], [], []
    gkf = GroupKFold(n_splits=n_splits)
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=groups.values)):
        p = _fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[te], n_estimators, seed)
        yt.extend(y.iloc[te].tolist())
        yp.extend(p.tolist())
        per.append((f"fold{i}", len(te), r2_score(y.iloc[te].tolist(), p.tolist())))
    return {"pooled": r2_score(yt, yp),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp))}, per


def run_loto(X, y, groups, seed, n_estimators, chem_seen_mask=None):
    """
    Leave-One-Group-Out。

    chem_seen_mask: 可选的布尔 Series（与 X 同索引），标记「该样本的化学组成
    在本次训练集中出现过」。给了就把每个留出组的测试样本再拆成
    chem_seen / chem_novel 两堆，分别算 R²（设计 1）。
    """
    yt, yp, per = [], [], []
    seen_rows, novel_rows = [], []
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups=groups.values):
        if len(te) == 0:
            continue
        p = _fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[te], n_estimators, seed)
        yte = y.iloc[te].to_numpy()
        gname = str(groups.iloc[te[0]])
        per.append((gname, len(te), r2_score(yte.tolist(), p.tolist())))
        yt.extend(yte.tolist())
        yp.extend(p.tolist())

        if chem_seen_mask is not None:
            m = chem_seen_mask.iloc[te].to_numpy()
            for flag, bucket in ((True, seen_rows), (False, novel_rows)):
                if (m == flag).sum() >= 3:
                    yy, pp = yte[m == flag], p[m == flag]
                    bucket.append((gname, int((m == flag).sum()),
                                   r2_score(yy.tolist(), pp.tolist())))
    return ({"pooled": r2_score(yt, yp),
             "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
             "MAE": float(mean_absolute_error(yt, yp))},
            per, seen_rows, novel_rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--chem-keys", default="coarse",
                    choices=["coarse", "fine"],
                    help="coarse=金属+连接体1+连接体2；fine=再加上官能团")
    ap.add_argument("--n-estimators", type=int, default=120)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    keys = CHEM_KEYS_COARSE if args.chem_keys == "coarse" else CHEM_KEYS_FINE
    suffix = f"_fact{('-' + args.tag) if args.tag else ''}"

    raw_target = args.target
    if raw_target.lower() in ("selectivity", "sel", "co2/n2"):
        raw_target = TARGET_SELECTIVITY

    print("=" * 78)
    print("判别实验：跨拓扑失败 = 拓扑本身 还是 化学新颖性？")
    print("=" * 78)
    print(f"目标变量   : {raw_target}" + ("（log10）" if args.log_target else ""))
    print(f"化学组成键 : {keys}")
    print(f"样本上限   : {args.max_rows}")
    print(f"随机种子   : {args.seed}")

    df = load_data(args.csv, args.max_rows)
    if raw_target not in df.columns:
        print(f"错误：没有列 '{raw_target}'")
        sys.exit(1)

    # 化学组成键（在抽样之后算，保证和建模样本一致）
    chem_key = (df[keys].astype(str).agg("|".join, axis=1)
                if len(keys) > 1 else df[keys[0]].astype(str))

    X, y, groups, cats, df_clean = build_features(df, raw_target, encoding="onehot")
    if args.log_target:
        X, y, groups, n_bad, keep = apply_log10(X, y, groups)
        df_clean = df_clean.loc[keep].reset_index(drop=True)
        chem_key = chem_key.loc[keep].reset_index(drop=True)
        print(f"已取 log10（丢弃 {n_bad} 个 <=0 样本）")

    chem_key = chem_key.reset_index(drop=True)
    topo = groups["topology"].reset_index(drop=True)
    print(f"建模样本   : {len(X)} 行, {X.shape[1]} 特征")

    # ---- 池 B：化学组成跨 ≥2 种拓扑的样本 ----
    n_topo_per_chem = chem_key.groupby(chem_key).apply(
        lambda s: topo.loc[s.index].nunique())
    multi_chem = set(n_topo_per_chem[n_topo_per_chem > 1].index)
    mask_B = chem_key.isin(multi_chem).to_numpy()
    print(f"跨多拓扑的化学组成: {len(multi_chem)} 个，"
          f"覆盖样本 {mask_B.sum()} / {len(X)} "
          f"（{mask_B.mean() * 100:.1f}%）")

    t0 = time.time()
    rows = []

    def record(support, arm, agg, extra=None):
        r = {"support": support, "arm": arm}
        r.update({k: v for k, v in agg.items() if k != "detail"})
        if extra:
            r.update(extra)
        rows.append(r)
        print(f"  [{support:6s}] {arm:34s} "
              f"pooled={agg['pooled']:+.4f}  macro={agg['macro']:+.4f}  "
              f"median={agg['median']:+.4f}")

    # ==================================================================
    # 设计 2：支持集 A（全样本） vs 支持集 B（化学跨多拓扑）
    # ==================================================================
    for support, mask in (("A_full", np.ones(len(X), dtype=bool)),
                          ("B_chem_matched", mask_B)):
        Xi, yi = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
        ti = topo.loc[mask].reset_index(drop=True)
        ci = chem_key.loc[mask].reset_index(drop=True)
        print(f"\n---- 支持集 {support}：n={len(Xi)}, "
              f"拓扑 {ti.nunique()} 种 ----")

        if len(Xi) < 500:
            print("  样本太少，跳过")
            continue

        # 随机划分
        pooled, per = run_kfold(Xi, yi, 5, args.seed, args.n_estimators)
        pooled.update(aggregate(per))
        record(support, "random_split", pooled)

        # 按拓扑 LOTO（池 B 上等价于「化学见过 + 拓扑没见过」）
        if support == "A_full":
            # 顺便做设计 1：按 chem 是否在训练集中出现，拆分测试集
            tr_chem_seen = []
            logo = LeaveOneGroupOut()
            seen_mask = pd.Series(False, index=Xi.index)
            for tr, te in logo.split(Xi, yi, groups=ti.values):
                seen = set(ci.iloc[tr].unique())
                seen_mask.iloc[te] = ci.iloc[te].isin(seen).to_numpy()
            pooled_l, per_l, seen_rows, novel_rows = run_loto(
                Xi, yi, ti, args.seed, args.n_estimators,
                chem_seen_mask=seen_mask)
            tr_chem_seen.append(seen_mask.mean())
        else:
            pooled_l, per_l, seen_rows, novel_rows = run_loto(
                Xi, yi, ti, args.seed, args.n_estimators)
            seen_rows, novel_rows = [], []

        pooled_l.update(aggregate(per_l))
        record(support, "topology_LOTO", pooled_l)

        if seen_rows:
            a = aggregate(seen_rows)
            a["pooled"] = float(np.average(
                [r[2] for r in seen_rows],
                weights=[r[1] for r in seen_rows]))
            record(support, "topology_LOTO / chem_seen(设计1)", a)
        if novel_rows:
            a = aggregate(novel_rows)
            a["pooled"] = float(np.average(
                [r[2] for r in novel_rows],
                weights=[r[1] for r in novel_rows]))
            record(support, "topology_LOTO / chem_novel(设计1)", a)

        # 按化学组成留组（化学没见过，拓扑混合/见过）
        if ci.nunique() >= 5:
            pooled_c, per_c = run_group_kfold(Xi, yi, ci, 5, args.seed,
                                              args.n_estimators)
            pooled_c.update(aggregate(per_c))
            record(support, "chemistry_GroupKFold", pooled_c)

        print(f"  （本支持集耗时 {time.time() - t0:.0f} s）")

    out = pd.DataFrame(rows)
    out = out.drop(columns=["detail"], errors="ignore")
    p = os.path.join(args.outdir, f"chem_topo_factorial{suffix}.csv")
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p}")

    # ---- 核心判据 ----
    print("\n" + "=" * 78)
    print("判据：ΔR² = 随机划分 R² − 拓扑 LOTO R²")
    print("=" * 78)
    for support in ("A_full", "B_chem_matched"):
        sub = out[out.support == support]
        if sub.empty:
            continue
        rr = sub[sub.arm == "random_split"]["pooled"]
        ll = sub[sub.arm == "topology_LOTO"]["pooled"]
        if len(rr) and len(ll):
            print(f"  {support:16s} 随机 {rr.iloc[0]:.4f}  →  LOTO "
                  f"{ll.iloc[0]:.4f}   ΔR² = {rr.iloc[0] - ll.iloc[0]:.4f}")
    a = out[(out.support == "A_full") &
            (out.arm.str.startswith("topology_LOTO /"))]
    if not a.empty:
        print("\n  设计 1（同一模型、同一训练集，只切测试子集）：")
        for _, r in a.iterrows():
            print(f"    {r['arm']:38s} macro={r['macro']:+.4f} "
                  f"median={r['median']:+.4f} n_groups={int(r['n_groups'])}")
    print("\n  读法：")
    print("   · 若 B 的 ΔR² 远小于 A 的 ΔR² → 崩塌主要来自「化学没见过」，")
    print("     标题 'geometry cannot extrapolate across topology' 必须改。")
    print("   · 若两者接近 → 拓扑本身确实是壁垒，标题可保留。")
    print("   · 设计 1 中 chem_seen 那一行的 macro 是「纯拓扑迁移」的直接估计。")
    print(f"\n总耗时 {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
