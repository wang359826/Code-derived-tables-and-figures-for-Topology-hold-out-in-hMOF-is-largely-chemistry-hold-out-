#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多目标验证：拓扑留出的惩罚大小，能否被"连接体化学缺失程度"预测？
================================================================

这是把本文结论从"描述一个混淆"升级为"一个可预测的定量关系"的关键实验。

逻辑链
------
1. 已证明（confound_diagnosis.py）：hMOF 里拓扑 ≈ 化学组成的代理
   （98% 化学组成只出现在单一拓扑，H(拓扑|化学)/H(拓扑) = 2.3%）。
2. 若 (1) 成立，则**推论**应该是：
   留出拓扑的惩罚大小 ΔR²，应当由"目标变量有多依赖连接体化学"决定。
   - 某个目标若主要由孔几何决定（几何描述符已能刻画）→ 化学没那么重要 → ΔR² 小；
   - 某个目标若强烈依赖框架-客体相互作用（连接体化学）→ ΔR² 大。
3. 于是对 hMOF 里全部可用目标做一遍，检验 **ΔR² 与"连接体残差信号"的相关性**。

这一步同时把先行工作的两个说法统一起来：
    Asgari 等（拓扑留出最难）  ×  Alimardani 等（描述符充分性随材料区间而变）
    → 二者是同一机制的两个表现，且可由一个定量关系预测。

指标定义
--------
- **R²_random**      : 随机 5 折 CV，pooled（测试支持集 = 全部样本）
- **R²_topology**    : 按拓扑 GroupKFold(5)，pooled
   ⚠ 两者每个样本都恰好落在一个测试折里，**支持集相同**，所以 R² 可直接相减
     （这与"逐组 R² 再聚合"是两回事，不要混用。）
- **ΔR²**            : R²_random − R²_topology
- **linker_residual_R²** : 先用**仅 8 个结构派生描述符**的 RF 做 5 折 CV 得到残差，
                        再用连接器对 one-hot 的 RF 去预测该残差（5 折 CV）的 R²。
                        = "几何描述符抓不住的连接体化学信号强度"
- **topology_residual_R²** : 同上但用 topology one-hot，作为对照

用法
----
  python multitarget_confound.py --max-rows 25000 --outdir outputs
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mof_leakage_benchmark import (  # noqa: E402
    NUMERIC_FEATURES, TARGET_SELECTIVITY, load_data, build_features,
    apply_log10,
)

TARGETS = [
    ("CO2_uptake_P0.15bar_T298K [mmol/g]", False, "CO2 uptake 0.15 bar (post-combustion)"),
    ("CO2_uptake_P0.10bar_T363K [mmol/g]", False, "CO2 uptake 0.10 bar / 363 K (TSA regen)"),
    ("CO2_uptake_P0.70bar_T413K [mmol/g]", False, "CO2 uptake 0.70 bar / 413 K"),
    ("excess_CO2_uptake_P0.15bar_T298K [mmol/g]", False, "excess CO2 uptake 0.15 bar"),
    ("CO2_binary_uptake_P0.15bar_T298K [mmol/g]", False, "CO2 uptake, CO2/N2 binary"),
    ("N2_binary_uptake_P0.85bar_T298K [mmol/g]", False, "N2 uptake, CO2/N2 binary"),
    ("working_capacity_vacuum_swing [mmol/g]", False, "working capacity (VSA)"),
    ("working_capacity_temperature_swing [mmol/g]", False, "working capacity (TSA)"),
    ("heat_adsorption_CO2_P0.15bar_T298K [kcal/mol]", False, "heat of adsorption CO2"),
    (TARGET_SELECTIVITY, True, "log10(CO2/N2 selectivity)"),
]


def cv_predict(X, y, n_est, seed, groups=None, n_splits=5):
    """返回与 y 等长的交叉验证预测（每个样本恰好出现在一个测试折）。"""
    if groups is None:
        sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        it = sp.split(X)
    else:
        k = max(2, min(n_splits, groups.nunique()))
        it = GroupKFold(n_splits=k).split(X, y, groups=groups.values)
    pred = np.empty(len(y), dtype=float)
    for tr, te in it:
        m = RandomForestRegressor(n_estimators=n_est, random_state=seed, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--resid-estimators", type=int, default=60,
                    help="残差模型的树数（只用于 linker/topology 残差回归，可少一些）")
    ap.add_argument("--top-linkers", type=int, default=200,
                    help="连接体对 one-hot 保留的最频繁类别数，其余归为 __OTHER__")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-col", default="topology",
                    choices=["topology", "metal", "linker_pair"],
                    help="分组变量的敏感性检验（R7）：除 topology 外还可用 metal / "
                         "linker_pair 重跑同一 2×2，检验 §4.2 的机制猜想是否只对"
                         "拓扑分组成立。除 topology 外会写入 "
                         "multitarget_confound_<group>.csv，不覆盖主表。")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 84)
    print("多目标验证：ΔR²(拓扑留出) 是否由「连接体化学缺失程度」预测？")
    print("=" * 84)

    df = load_data(args.csv, args.max_rows)
    rows = []
    t_start = time.time()

    for col, use_log, nice in TARGETS:
        if col not in df.columns:
            print(f"  跳过（无此列）: {col}")
            continue
        d = df[["topology", "metal_linker", "organic_linker1",
                "organic_linker2", "functional_groups"] + NUMERIC_FEATURES + [col]]
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        if use_log:
            d = d[d[col] > 0].copy()
            y_raw = np.log10(d[col])
        else:
            y_raw = d[col]
        if len(d) < 2000:
            print(f"  跳过（样本不足）: {col}")
            continue

        X, y, groups, cats, _ = build_features(
            d.assign(**{"__y__": y_raw.values}), "__y__", encoding="onehot")
        topo = groups["topology"].reset_index(drop=True)
        lp = groups["linker_pair"].reset_index(drop=True)
        # R7：分组变量的敏感性检验。grp 是本次运行实际使用的分组变量；
        # topology 时与既有列完全一致（向后兼容），其他取值只**新增**列。
        if args.group_col == "topology":
            grp = topo
        elif args.group_col == "metal":
            grp = groups["metal"].reset_index(drop=True)
        else:
            grp = lp

        Xg = X[NUMERIC_FEATURES]                       # 仅结构派生描述符
        onehot_cols = [c for c in X.columns if c not in NUMERIC_FEATURES]
        Xt = X[[c for c in onehot_cols if str(c).startswith("topology_")]]
        # ⚠ 坑：build_features 的 onehot 只编码 CATEGORICAL_FEATURES
        # (functional_groups / metal_linker / topology)，**没有任何 linker_pair_ 列**。
        # 连接体 one-hot 必须由 groups["linker_pair"] 现场构造，否则这里会取到空表、
        # linker_residual_R² 全是 NaN，相关性检验被静默跳过。
        # 587 个连接体对全 one-hot 会让 RF 慢到不可用（实测 25k 行跑 10 个目标 > 40 分钟）。
        # 折中：只保留最频繁的 --top-linkers 个，其余归入 __OTHER__。
        # 这会略微低估连接体可解释的方差，但**对所有目标一视同仁**，
        # 而我们关心的是跨目标的**相对排序**，不是绝对水平。
        freq = lp.value_counts()
        keep = set(freq.nlargest(args.top_linkers).index)
        lp2 = lp.where(lp.isin(keep), other="__OTHER__")
        Xl = pd.get_dummies(lp2.astype(str), prefix="lp").astype(float)
        Xl = Xl.reset_index(drop=True)
        Xl.index = X.index

        t0 = time.time()
        # ⚠️ 2×2 因子设计（回应 R10）。
        #    旧版只算了三个格子，(Xg, 随机) 与 (X, 拓扑留出) 做对比，
        #    **同时换了划分与特征集**，"10/10 加身份有害"绝大部分其实是划分效应。
        #    缺失的第四格 (Xg, 拓扑留出) 现在补上，于是可以干净地分解：
        #        划分效应（同特征集）：random → topology
        #        身份效应（同划分）  ：   Xg   → X（在 topology 划分下）
        p_rand = cv_predict(X, y, args.n_estimators, args.seed)          # 随机 + 身份
        p_topo = cv_predict(X, y, args.n_estimators, args.seed,
                            groups=topo)                                 # 拓扑 + 身份
        p_rand_geo = cv_predict(Xg, y, args.n_estimators, args.seed)     # 随机 + 仅结构
        p_topo_geo = cv_predict(Xg, y, args.n_estimators, args.seed,
                                groups=topo)                             # 拓扑 + 仅结构
        r2_rand = r2_score(y, p_rand)
        r2_topo = r2_score(y, p_topo)
        r2_rand_geo = r2_score(y, p_rand_geo)
        r2_topo_geo = r2_score(y, p_topo_geo)

        # 几何模型的残差 → 用连接体 / 拓扑身份去解释
        p_geo = cv_predict(Xg, y, args.n_estimators, args.seed)
        # R7：用本次 --group-col 指定的分组变量再取同一 2×2 的两个格
        p_grp = cv_predict(X, y, args.n_estimators, args.seed, groups=grp)
        p_grp_geo = cv_predict(Xg, y, args.n_estimators, args.seed, groups=grp)
        r2_grp = r2_score(y, p_grp)
        r2_grp_geo = r2_score(y, p_grp_geo)
        resid = (y - p_geo).reset_index(drop=True)
        sd_resid = float(resid.std())
        r2_linker_resid = r2_score(
            resid, cv_predict(Xl, resid, args.resid_estimators, args.seed)) \
            if Xl.shape[1] > 1 else np.nan
        r2_topo_resid = r2_score(
            resid, cv_predict(Xt, resid, args.resid_estimators, args.seed)) \
            if Xt.shape[1] > 1 else np.nan

        rows.append({
            "target": col, "label": nice, "log": use_log, "n": len(X),
            "sd_y": float(y.std()),
            "R2_random_full": float(r2_rand),
            "R2_topology_full": float(r2_topo),
            "R2_random_geo": float(r2_rand_geo),
            "R2_topology_geo": float(r2_topo_geo),
            # ---- 分解后的两个效应（这才是可写进正文的量）----
            # 划分效应（特征集固定为全特征 / 固定为仅结构）：随机 − 拓扑
            "split_effect_full": float(r2_rand - r2_topo),
            "split_effect_geo": float(r2_rand_geo - r2_topo_geo),
            # 身份效应（划分固定）：全特征 − 仅结构
            "identity_effect_topo": float(r2_topo - r2_topo_geo),
            "identity_effect_random": float(r2_rand - r2_rand_geo),
            # 旧字段保留（= 划分效应，全特征），供旧图表/文档对照
            "R2_random": float(r2_rand),
            "R2_topology": float(r2_topo),
            "delta_R2": float(r2_rand - r2_topo),
            "R2_geometry_only": float(r2_rand_geo),
            "linker_residual_R2": float(r2_linker_resid),
            "topology_residual_R2": float(r2_topo_resid),
            "sd_residual": sd_resid,
            # R7：以 --group-col 指定的分组变量重做的同一 2×2（默认 topology 时与
            # R2_topology_full / R2_topology_geo / identity_effect_topo 相同）
            "group_col": args.group_col,
            "R2_grp_full": float(r2_grp),
            "R2_grp_geo": float(r2_grp_geo),
            "split_effect_grp": float(r2_rand - r2_grp),
            "identity_effect_grp": float(r2_grp - r2_grp_geo),
        })
        r = rows[-1]
        print(f"  {nice:44s} 划分效应={r['split_effect_full']:+.3f}  "
              f"身份效应(拓扑划分内)={r['identity_effect_topo']:+.3f}  "
              f"({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    fname = ("multitarget_confound.csv" if args.group_col == "topology"
             else f"multitarget_confound_{args.group_col}.csv")
    p = os.path.join(args.outdir, fname)
    out.to_csv(p, index=False, encoding="utf-8-sig")

    # ---------------- 2×2 分解汇总（R10 的核心输出）----------------
    print("\n" + "=" * 84)
    print("2×2 分解：划分效应 vs 身份效应（R10）")
    print("=" * 84)
    try:
        from scipy.stats import wilcoxon
        se = out["split_effect_full"].to_numpy()
        ie = out["identity_effect_topo"].to_numpy()
        se_g = out["split_effect_geo"].to_numpy()
        ie_r = out["identity_effect_random"].to_numpy()
        print(f"  划分效应（全特征，随机−拓扑）  中位 {np.median(se):+.3f}  "
              f"[{se.min():+.3f}, {se.max():+.3f}]")
        print(f"  划分效应（仅结构，随机−拓扑）  中位 {np.median(se_g):+.3f}  "
              f"[{se_g.min():+.3f}, {se_g.max():+.3f}]")
        print(f"  身份效应（拓扑划分内，全−仅结构）中位 {np.median(ie):+.3f}  "
              f"[{ie.min():+.3f}, {ie.max():+.3f}]")
        print(f"  身份效应（随机划分内，全−仅结构）中位 {np.median(ie_r):+.3f}  "
              f"[{ie_r.min():+.3f}, {ie_r.max():+.3f}]")
        n_less = int((ie < 0).sum())
        print(f"\n  ★ 身份效应为负（'加身份有害'）的目标数: {n_less}/{len(ie)}")
        if len(ie) >= 5:
            w, pw = wilcoxon(ie)
            print(f"    Wilcoxon（身份效应 ≠ 0）: W = {w:.1f}, p = {pw:.4f}")
        # R7：分组变量敏感性（--group-col）
        if args.group_col != "topology":
            ieg = out["identity_effect_grp"].to_numpy()
            print(f"\n  ★ --group-col {args.group_col} 下的同一 2×2：")
            print(f"    身份效应（{args.group_col} 划分内，全−仅结构）中位 "
                  f"{np.median(ieg):+.3f}  [{ieg.min():+.3f}, {ieg.max():+.3f}]  "
                  f"负值 {int((ieg < 0).sum())}/{len(ieg)}")
            if len(ieg) >= 5:
                w2, pw2 = wilcoxon(ieg)
                print(f"    Wilcoxon: W = {w2:.1f}, p = {pw2:.4f}")
            print(f"    对照：拓扑划分下的身份效应中位 {np.median(ie):+.3f} "
                  f"(p 见上)")
        # 旧结论"10/10 有害"实际量的是 split_effect + identity_effect 之和
        old = out["R2_topology"].to_numpy() - out["R2_random_geo"].to_numpy()
        print(f"\n  旧口径（R2_拓扑全特征 − R2_随机仅结构）中位 {np.median(old):+.3f}"
              f"  ← 这就是稿子曾报的'中位惩罚'，其中绝大部分是划分效应")
        print(f"  划分效应占旧口径的比例（中位）: "
              f"{np.median(se / np.where(old == 0, np.nan, old)):.2f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  汇总失败: {exc}")

    print("\n" + "=" * 84)
    print("相关性检验")
    print("=" * 84)
    if out["linker_residual_R2"].isna().any():
        print("\n  ⚠ WARNING: 有目标的 linker_residual_R² 是 NaN —— 说明连接体 one-hot")
        print("    没有构造成功（多半是取到空表），相关性检验必然失效，请检查 Xl 构造。")
    if len(out) >= 4:
        sub = out.dropna(subset=["linker_residual_R2", "delta_R2"])
        if len(sub) < 4:
            print("\n  ⚠ WARNING: 可用目标不足 4 个，无法做相关性检验。")
        else:
            from scipy.stats import pearsonr, spearmanr
            x = sub["linker_residual_R2"].to_numpy()
            yv = sub["delta_R2"].to_numpy()
            pr, pp = pearsonr(x, yv)
            sr, sp = spearmanr(x, yv)
            print(f"  n targets = {len(sub)}")
            print(f"  Pearson  r = {pr:+.3f}  (p = {pp:.4f})")
            print(f"  Spearman rho = {sr:+.3f}  (p = {sp:.4f})")
            print("\n  读法：若 r 显著为正，说明「几何描述符抓不住的连接体化学」越多，")
            print("        留出拓扑的惩罚就越大 —— 这正是拓扑≡化学代理这一混淆所预言的。")
            inter = sub[["target", "delta_R2", "linker_residual_R2",
                         "R2_random", "R2_topology"]]
            print("\n  " + inter.to_string(index=False))

    print(f"\n已保存: {p}   总耗时 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
