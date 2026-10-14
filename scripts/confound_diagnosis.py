#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混淆诊断 + 受控的等网状（isoreticular）对照实验
==============================================

背景：上一版判别实验（chemistry_topology_factorial.py）的设计 2 是坏的，原因已查明：
------------------------------------------------------------------------------
chemistry key = 金属 + 连接体1 + 连接体2 时，
    388 个"跨多拓扑"的化学组成里，**387 个只跨 (pcu, nbo) 这一对**，
    剩下的 1 个跨 (rht, tbo)。
也就是说所谓"化学匹配池"其实只有 **两个**拓扑，不是 11 个；
而且 **nbo 的目标方差只有 0.0067（sd 0.082），是 pcu（0.196 / sd 0.443）的 1/29**。
在这种支持集上比较 R²，等于比较分母相差 29 倍的两个量——没有意义。

所以本脚本换成两件能做对的事：
------------------------------------
【A】量化「化学组成 → 拓扑」的混淆程度
    - 有多少比例的化学组成只出现在单一拓扑
    - 条件熵 H(topology | chemistry) / H(topology)：知道化学后还剩多少拓扑不确定性
    - 各拓扑的chemistry集合两两 Jaccard 重叠
    → 若 H(T|C) ≈ 0，说明「留出拓扑」≈「留出化学」，
      那么领域里通行的 topology-holdout 根本无法区分这两个假说。
      **这个"做不到"本身就是可发表的结果。**

【B】唯一一处 hMOF 真正提供的受控对照：pcu ↔ nbo
    387 个化学组成同时存在于 pcu 与 nbo（真正的等网状多晶型对）。
    固定**同一个测试集**，只换训练集：
        (a) 用除 T 外所有拓扑训练          → 化学见过（经配对拓扑）
        (b) 用除 T 和配对拓扑外所有拓扑训练 → 化学没见过
    两者测试集完全相同 → 方差口径一致，可直接比较，且同时报 R² / MAE / RMSE。

用法
----
  python confound_diagnosis.py --max-rows 50000 --outdir outputs
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mof_leakage_benchmark as mlb  # noqa: E402
from mof_leakage_benchmark import (  # noqa: E402
    TARGET, build_features, load_data,
)

KEYS_COARSE = ["metal_linker", "organic_linker1", "organic_linker2"]
KEYS_FINE = ["metal_linker", "organic_linker1",
             "organic_linker2", "functional_groups"]


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def confound_report(df, keys, topo_col="topology", label=""):
    ck = (df[keys].fillna("NA").astype(str).agg("|".join, axis=1))
    tp = df[topo_col].fillna("NA").astype(str)

    n_chem = ck.nunique()
    ntop_per_chem = ck.groupby(ck).apply(lambda s: tp.loc[s.index].nunique())
    single = float((ntop_per_chem == 1).mean())

    # H(T) 与 H(T|C)
    pt = tp.value_counts(normalize=True).values
    HT = entropy(pt)
    hc = 0.0
    for _, g in tp.groupby(ck):
        w = len(g) / len(tp)
        hc += w * entropy(g.value_counts(normalize=True).values)
    cond = hc / HT if HT > 0 else np.nan

    # 2026-10-12 新增（回应 AUD-E1 的"两套加权必须都报"）：
    #   上面的 single / hc 口径混用——single 是**组成等权**（每个组成算一次），
    #   hc 是**结构加权**（每个结构算一次）。这里把两套口径都算全，避免读者误读。
    n_by_c = tp.groupby(ck).size()
    single_struct = float(n_by_c[ntop_per_chem == 1].sum() / len(tp))
    hc_comp = float(tp.groupby(ck).apply(
        lambda s: entropy(s.value_counts(normalize=True).values)).mean())
    cond_comp = hc_comp / HT if HT > 0 else np.nan

    # 拓扑对的 chemistry 集合 Jaccard
    sets = {t: set(ck[tp == t].unique()) for t in tp.unique()}
    rows = []
    for a, b in itertools.combinations(sorted(sets, key=lambda t: -len(sets[t]))[:8], 2):
        inter = len(sets[a] & sets[b])
        un = len(sets[a] | sets[b])
        rows.append({"topo_a": a, "topo_b": b, "chem_overlap": inter,
                     "jaccard": inter / un if un else 0.0,
                     "n_a": int((tp == a).sum()), "n_b": int((tp == b).sum())})
    jac = pd.DataFrame(rows).sort_values("chem_overlap", ascending=False)

    print(f"\n【化学-拓扑混淆 · {label}】key = {keys}")
    print(f"  化学组成总数 {n_chem}，其中只出现在单一拓扑的比例 = {single * 100:.1f}%")
    print(f"  H(topology) = {HT:.3f} bit")
    print(f"  H(topology | chemistry) = {hc:.3f} bit")
    print(f"  **条件熵占比 = {cond * 100:.1f}%**  "
          f"（越低说明化学几乎完全决定了拓扑）")
    print("  化学集合重叠最多的拓扑对：")
    if len(jac):
        print(jac.head(6).to_string(index=False))
    return {"label": label, "n_chem": n_chem,
            "frac_single_topology": single,
            "frac_single_topology_struct": single_struct,
            "H_T": HT, "H_T_given_C": hc, "cond_frac": cond,
            "H_T_given_C_comp": hc_comp, "cond_frac_comp": cond_comp}, jac


def controlled_isoreticular(df, X, y, topo, ck, target_topo, n_est, seed,
                            outdir, size_match=False, sm_seeds=5):
    """
    同一个测试集（target_topo 的全部样本），只换训练集：
      (a) 除 target_topo 外所有拓扑
      (b) 除 target_topo 与其"化学配对拓扑"外所有拓扑
    """
    out_rows = []
    paired = set()
    # 找出与 target_topo 共享化学组成最多的那个拓扑（即配对拓扑）
    sets = {t: set(ck[topo == t].unique()) for t in topo.unique()}
    if target_topo in sets:
        cand = [(t, len(sets[target_topo] & sets[t]))
                for t in sets if t != target_topo]
        cand.sort(key=lambda x: -x[1])
        best_t, best_n = cand[0]
        paired = {best_t}
        print(f"\n  {target_topo} 的化学配对拓扑 = {best_t}"
              f"（共享 {best_n} 个化学组成）")

    te_mask = (topo == target_topo).to_numpy()
    n_te = int(te_mask.sum())
    if n_te < 50:
        print(f"  {target_topo} 只有 {n_te} 个样本，跳过")
        return []

    def _fit_eval(tr_mask, regime):
        """训练并在同一测试集上评估，返回一行结果。"""
        if tr_mask.sum() < 200:
            return None
        m = RandomForestRegressor(n_estimators=n_est, random_state=seed,
                                  n_jobs=-1)
        m.fit(X.iloc[tr_mask], y.iloc[tr_mask])
        p = m.predict(X.iloc[te_mask])
        yt = y.iloc[te_mask].to_numpy()
        sd = float(yt.std())
        row = {
            "test_topology": target_topo,
            "regime": regime,
            "n_train": int(tr_mask.sum()),
            "n_test": n_te,
            "sd_y_test": sd,
            "R2": float(r2_score(yt, p.tolist())),
            "MAE": float(mean_absolute_error(yt, p)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, p))),
            "MAE_over_sd": float(mean_absolute_error(yt, p) / sd) if sd > 0 else np.nan,
        }
        print(f"    {regime:42s} n_train={row['n_train']:6d}  "
              f"R²={row['R2']:+.4f}  MAE={row['MAE']:.4f}  "
              f"RMSE={row['RMSE']:.4f}  MAE/sd={row['MAE_over_sd']:.4f}")
        return row

    mask_a = (~te_mask)                                     # (a) 除自己外全部
    mask_b = (~te_mask) & (~topo.isin(paired).to_numpy())   # (b) 再去掉配对拓扑

    for regime, msk in (("train_all_other_topologies", mask_a),
                        (f"train_excluding_pair{'|'.join(paired)}", mask_b)):
        row = _fit_eval(msk, regime)
        if row is not None:
            out_rows.append(row)

    # ---------------- (c) 样本量匹配对照（回应 R1）----------------
    # (a) 与 (b) 的训练集规模差 3.98 倍（42,329 vs 10,647），
    # 所以 (b) 变差可能只是"训练数据变少"，而不是"化学没见过"。
    # (c) 从 (a) 的池子里随机抽到与 (b) **同样多的**样本，多个种子取平均：
    #     若 MAE(c) ≈ MAE(a) 而 MAE(b) 明显更差 → 是化学效应；
    #     若 MAE(c) ≈ MAE(b)                   → 只是样本量效应。
    if size_match and paired:
        n_b = int(mask_b.sum())
        idx_a = np.where(mask_a)[0]
        if n_b >= 200 and n_b < len(idx_a):
            print(f"    [样本量匹配] 从 (a) 的 {len(idx_a)} 个样本中 "
                  f"随机抽 {n_b} 个 × {sm_seeds} 个种子")
            maes, rmses = [], []
            for i in range(sm_seeds):
                rng = np.random.default_rng(seed + 1000 + i)
                pick = rng.choice(idx_a, size=n_b, replace=False)
                m2 = np.zeros(len(topo), dtype=bool)
                m2[pick] = True
                row = _fit_eval(m2, f"size_matched_seed{i}")
                if row is not None:
                    out_rows.append(row)
                    maes.append(row["MAE"])
                    rmses.append(row["RMSE"])
            if maes:
                print(f"    [样本量匹配] MAE = {np.mean(maes):.4f} ± "
                      f"{np.std(maes, ddof=1) if len(maes) > 1 else 0:.4f}"
                      f"   RMSE = {np.mean(rmses):.4f}")
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--n-estimators", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    # ⚠️ 论文摘要里的 "MAE 0.1065 ± 0.0032 → 0.1811 ± 0.0136 (+70.0%)"
    #    来自 isoreticular_control_seeds.csv（5 个种子，连抽样一起换）。
    #    早期版本曾把这一支删掉，导致该 CSV **没有任何脚本能再生成**
    #    （只有 fig_confound_and_control.py 在读它）→ 2026-10-10 补回。
    ap.add_argument("--seeds", type=int, default=1,
                    help=">1 时额外跑多种子受控对照，写出 "
                         "isoreticular_control_seeds.csv")
    ap.add_argument("--seed-list", type=str,
                    default="42,143,244,345,446")
    ap.add_argument("--control-topos", type=str, default="pcu,nbo")
    # 样本量匹配对照（回应 R1：(a) 与 (b) 训练集规模差 3.98 倍）
    ap.add_argument("--size-match", action="store_true",
                    help="额外跑一个'从 (a) 随机抽到与 (b) 同规模'的对照臂")
    ap.add_argument("--size-match-seeds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("化学 × 拓扑 混淆诊断 + 受控等网状对照")
    print("=" * 78)

    df = load_data(args.csv, args.max_rows)
    X, y, groups, cats, dfc = build_features(df, TARGET, encoding="onehot")
    topo = groups["topology"].reset_index(drop=True)

    conf_rows, jac_all = [], []
    for keys, label in ((KEYS_COARSE, "coarse(金属+两连接体)"),
                        (KEYS_FINE, "fine(+官能团)")):
        c, j = confound_report(df, keys, "topology", label)
        conf_rows.append(c)
        j["key"] = label
        jac_all.append(j)

    pd.DataFrame(conf_rows).to_csv(
        os.path.join(args.outdir, "chem_topo_confound.csv"),
        index=False, encoding="utf-8-sig")
    pd.concat(jac_all).to_csv(
        os.path.join(args.outdir, "chem_topo_jaccard.csv"),
        index=False, encoding="utf-8-sig")

    # ---- 受控对照：只对最大的几个拓扑做 ----
    print("\n" + "=" * 78)
    print("受控对照（同一测试集，只换训练集）")
    print("=" * 78)
    ck = (df[KEYS_COARSE].fillna("NA").astype(str)
          .agg("|".join, axis=1).reset_index(drop=True))
    rows = []
    big = topo.value_counts().head(4).index.tolist()
    for t in big:
        print(f"\n  测试拓扑 = {t}（n={int((topo == t).sum())}）")
        rows += controlled_isoreticular(df, X, y, topo, ck, t,
                                        args.n_estimators, args.seed, args.outdir,
                                        size_match=args.size_match,
                                        sm_seeds=args.size_match_seeds)
    if rows:
        d = pd.DataFrame(rows)
        p = os.path.join(args.outdir, "isoreticular_control.csv")
        d.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n已保存: {p}")
        if args.size_match:
            sm = d[d["regime"].astype(str).str.startswith("size_matched")]
            ps = os.path.join(args.outdir,
                              "isoreticular_control_sizematched.csv")
            sm.to_csv(ps, index=False, encoding="utf-8-sig")
            print(f"已保存: {ps}（样本量匹配对照，回应 R1）")
    else:
        print("\n⚠️ 受控对照没有产出任何行——请检查数据/拓扑列，不要静默跳过。")

    # ---- 多种子受控对照（论文 §3.4 的 ± 数字来源）----
    if args.seeds > 1:
        seed_list = [int(s) for s in args.seed_list.split(",")][:args.seeds]
        topos = [t.strip() for t in args.control_topos.split(",") if t.strip()]
        print("\n" + "=" * 78)
        print(f"多种子受控对照（{len(seed_list)} 个种子：抽样 + 模型一并换）")
        print("=" * 78)
        srows = []
        for sd in seed_list:
            print(f"\n  --- seed = {sd} ---")
            # load_data() 用模块级 RANDOM_STATE 采样，改它即可换抽样
            mlb.RANDOM_STATE = sd
            dfs = load_data(args.csv, args.max_rows)
            Xs, ys, gs, _, _ = build_features(dfs, TARGET, encoding="onehot")
            ts = gs["topology"].reset_index(drop=True)
            cks = (dfs[KEYS_COARSE].fillna("NA").astype(str)
                   .agg("|".join, axis=1).reset_index(drop=True))
            for t in topos:
                for r in controlled_isoreticular(dfs, Xs, ys, ts, cks, t,
                                                 args.n_estimators, sd,
                                                 args.outdir):
                    r = dict(r)
                    r["seed"] = sd
                    r["test_topo"] = r.pop("test_topology")
                    r["regime"] = ("with_pair" if r["regime"] ==
                                   "train_all_other_topologies"
                                   else "without_pair")
                    r.pop("n_test", None)
                    r.pop("sd_y_test", None)
                    srows.append(r)
        mlb.RANDOM_STATE = args.seed
        if srows:
            ds = pd.DataFrame(srows)
            cols = ["seed", "test_topo", "regime", "n_train",
                    "R2", "MAE", "RMSE", "MAE_over_sd"]
            ds = ds[[c for c in cols if c in ds.columns]]
            ps = os.path.join(args.outdir, "isoreticular_control_seeds.csv")
            ds.to_csv(ps, index=False, encoding="utf-8-sig")
            print(f"\n已保存: {ps}")
            # 顺手把 ± 汇总打出来，方便直接抄进摘要
            for t in ds["test_topo"].unique():
                s = ds[ds.test_topo == t]
                a = s[s.regime == "with_pair"]["MAE"]
                b = s[s.regime == "without_pair"]["MAE"]
                ra = s[s.regime == "with_pair"]["RMSE"]
                rb = s[s.regime == "without_pair"]["RMSE"]
                print(f"  {t}: MAE {a.mean():.4f}±{a.std(ddof=1):.4f} → "
                      f"{b.mean():.4f}±{b.std(ddof=1):.4f} "
                      f"({(b.mean() / a.mean() - 1) * 100:+.1f}%) | "
                      f"RMSE {ra.mean():.4f}±{ra.std(ddof=1):.4f} → "
                      f"{rb.mean():.4f}±{rb.std(ddof=1):.4f} "
                      f"({(rb.mean() / ra.mean() - 1) * 100:+.1f}%)")
        else:
            print("\n⚠️ 多种子分支没有产出任何行——显式告警，不要静默通过。")

    print("\n" + "=" * 78)
    print("怎么读")
    print("=" * 78)
    print("  · 条件熵占比越低 → 化学几乎决定拓扑 → topology-holdout 与")
    print("    chemistry-holdout 在 hMOF 上是同一件事，无法区分。")
    print("    **这个'做不到'就是结论**：领域把 topology holdout 当金标准，")
    print("    但它测的其实是未见化学。")
    print("  · 受控对照里，(a) 与 (b) 若 MAE 接近 → 化学见过没有帮助，")
    print("    说明壁垒在拓扑本身；若 (a) 明显更好 → 化学是主因。")
    print("    注意：两侧测试集完全相同，所以 R²/MAE 都可直接比较。")


if __name__ == "__main__":
    main()
