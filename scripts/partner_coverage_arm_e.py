# -*- coding: utf-8 -*-
"""
Arm (e)：把「building-block 多样性数量」与「matched-partner 覆盖」分开
====================================================================

§4.5 里唯一被点名「在本库内无法分离」的东西
------------------------------------------
> What remains **open** is the decomposition of arm (c)'s advantage: pcu supplies
> both the library's dominant chemistry and nbo's only chemical partnership, so
> "building-block diversity" and "matched-partner coverage" cannot be varied
> independently in this database.

arm (d)（composition-matched）没能分开这两者，因为它**按组成剖面去匹配 (b)**，
而被匹配的正是决定覆盖的那个量：(d) 的 #comp = 496 = (b)，覆盖也和 (b) 一样为零，
所以 (d) 与 (b) 的 MAE 完全相同（0.1811）并不提供区分信息。

arm (e) 换了切法：**只剔除配对拓扑里那些组成出现在测试集中的行**。

    (a) 全部（含配对拓扑）                       n = 42,329   #comp 1,484  覆盖 满
    (b) 配对拓扑整条移除                         n = 10,647   #comp   496  覆盖 零
    (e) 只移除「共享组成」的配对拓扑行            n ≈ 41,9xx   #comp ~1,1xx  覆盖 零

关键性质：**(e) 与 (a) 天然近似规模匹配**（只去掉共享的那部分行），但
matched-partner 覆盖归零，同时 #comp 只掉共享的那一小部分、多样性基本保留。于是：

  - 若 (e) 的 MAE ≈ (a)（≈0.107）⇒ 多样性**数量**是驱动量，覆盖无关；
  - 若 (e) 的 MAE ≈ (b)（≈0.181）⇒ **覆盖**是驱动量，多样性的多少不重要。

这是 hMOF 这一个库内能做的、唯一有区分力的设计。

口径与 `diversity_control.py` 完全一致
--------------------------------------
同一个 N_ROWS = 50,000 抽样、同一组 5 个 seed（42/143/244/345/446，抽样·划分·模型
一起重抽）、同一套 8 个数值特征 + 3 个 one-hot 块、同一个 120 树 RandomForest、
同一套组成定义 `KEYS_COARSE = metal_linker | organic_linker1 | organic_linker2`。
因此 (a) 与 (b) 的数字应逐位复现 `outputs/composition_matched_control.csv`，
(e) 是新增的第三个臂。

产出
----
    outputs/arm_e_partner_coverage.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys_path = HERE
import sys  # noqa: E402
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from diversity_control import (  # noqa: E402
    TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES, KEYS_COARSE,
    N_ESTIMATORS, N_ROWS, SEED_LIST, CONTROL_TOPOS,
    load_and_build, diversity_stats,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

OUT = os.path.join(HERE, "..", "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        HERE, "..", "data", "all_MOFs_screening_data.csv"))
    ap.add_argument("--outdir", default=OUT)
    ap.add_argument("--topos", default=",".join(CONTROL_TOPOS))
    ap.add_argument("--arms", default="a,b,e,f",
                    help="要重算的臂，逗号分隔，取值 a/b/e/f（默认全部）。"
                         "只重算部分臂时输出到 arm_e_partner_coverage_<arms>.csv，"
                         "以免为改一列而重跑全部模型。")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="RandomForest 并行度。默认 1，原因有二：(i) 受限沙箱下 "
                         "joblib 的多进程池无法创建命名管道，会以 WinError 5 失败；"
                         "(ii) 在固定 random_state 下 n_jobs 不改变拟合结果。")
    args = ap.parse_args()
    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    os.makedirs(args.outdir, exist_ok=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    topos = [t.strip() for t in args.topos.split(",") if t.strip()]
    rows = []

    for seed in SEED_LIST:
        X, y, topo, ck, lp, metal = load_and_build(args.csv, seed)
        topo = topo.reset_index(drop=True)
        ck = ck.reset_index(drop=True)
        lp = lp.reset_index(drop=True)
        metal = metal.reset_index(drop=True)
        tv = topo.to_numpy()
        cv = ck.to_numpy()
        print(f"\n{'=' * 78}\nseed = {seed}   建模样本 {len(X):,}\n{'=' * 78}")

        for target_topo in topos:
            te = np.where(tv == target_topo)[0]
            if len(te) < 50:
                print(f"  {target_topo}: 测试样本仅 {len(te)}，跳过")
                continue

            # 配对拓扑 = 与之共享化学组成最多的拓扑（与 §3.4 同一定义）
            sets = {t: set(cv[tv == t]) for t in pd.unique(tv)}
            cand = sorted(((t, len(sets[target_topo] & sets[t]))
                           for t in sets if t != target_topo),
                          key=lambda x: -x[1])
            partner, n_shared = cand[0]

            te_mask = np.zeros(len(tv), dtype=bool)
            te_mask[te] = True
            test_comps = set(cv[te])

            idx_a = np.where(~te_mask)[0]
            idx_b = np.where((~te_mask) & (tv != partner))[0]
            # ---- arm (e) 的定义就在这一行 ----
            # 只剔除「属于配对拓扑」且「组成出现在测试集里」的行
            shared_and_partner = np.array(
                [(tv[i] == partner) and (cv[i] in test_comps)
                 for i in range(len(tv))])
            idx_e = np.where((~te_mask) & ~shared_and_partner)[0]

            n_removed_e = int(shared_and_partner.sum())
            print(f"\n  测试拓扑 {target_topo}（n_test={len(te):,}）"
                  f"  配对拓扑 = {partner}（共享 {n_shared} 个组成）"
                  f"  (e) 剔除 {n_removed_e:,} 行")

            def fit_eval(idx, regime):
                if len(idx) < 200:
                    return None
                m = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                           random_state=seed, n_jobs=args.n_jobs)
                m.fit(X.iloc[idx], y.iloc[idx])
                p = m.predict(X.iloc[te])
                yt = y.iloc[te].to_numpy()
                r = {"seed": seed, "test_topo": target_topo,
                     "partner": partner, "regime": regime,
                     "n_shared_compositions": int(n_shared),
                     "R2": float(r2_score(yt, p)),
                     "MAE": float(mean_absolute_error(yt, p)),
                     "RMSE": float(np.sqrt(mean_squared_error(yt, p)))}
                r.update(diversity_stats(idx, topo, ck, lp, metal))
                print(f"    {regime:22s} n_train={r['n_train']:6d}  "
                      f"MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                      f"#topo={r['n_topologies']:3d}  "
                      f"#lp={r['n_linker_pairs']:4d}  "
                      f"#comp={r['n_compositions']:6d}")
                return r

            for idx, regime, tag in ((idx_a, "a_with_partner", "a"),
                                     (idx_b, "b_partner_removed", "b"),
                                     (idx_e, "e_shared_only_removed", "e")):
                if tag not in arms:
                    continue
                r = fit_eval(idx, regime)
                if r is not None:
                    r["n_rows_removed_vs_a"] = int(len(idx_a) - len(idx))
                    rows.append(r)

            # ---- arm (f) size-matched control (R1-M4 resolution) ----
            # From arm (e)'s pool, draw the same number of rows as arm (b).
            # This isolates the size effect from the chemistry/coverage effect
            # in the (e)->(b) step.
            n_b = int((~te_mask & (tv != partner)).sum())
            idx_e = np.where((~te_mask) & ~shared_and_partner)[0]
            if "f" in arms and len(idx_e) >= n_b >= 200:
                rng = np.random.default_rng(seed + 2000)
                pick = rng.choice(idx_e, size=n_b, replace=False)
                # 2026-10-14 修：这里原来传的是布尔掩码 m2（长度 = len(tv) = 50,000），
                # 而 diversity_stats 用 len(idx) 记 n_train，于是 arm (f) 的 n_train
                # 被写成 50,000（= --max-rows 池大小），与正文/SI 的 10,656 冲突。
                # 传索引数组即可：X.iloc[pick] 与 topo.iloc[pick] 都成立，且
                # n_train = len(pick) = n_b（与同种子的 arm (b) 一致）。
                r = fit_eval(pick, "f_size_matched_from_e")
                if r is not None:
                    r["n_rows_removed_vs_a"] = int(len(idx_a) - len(pick))
                    rows.append(r)

    d = pd.DataFrame(rows)
    full_set = {"a", "b", "e", "f"}
    tag = "" if arms == full_set else "_" + "".join(sorted(arms))
    p = os.path.join(args.outdir, f"arm_e_partner_coverage{tag}.csv")
    d.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p}")

    print("\n" + "=" * 78)
    print("判读：5 seed 汇总（mean ± sd）")
    print("=" * 78)
    g = d.groupby(["test_topo", "regime"]).agg(
        n_train=("n_train", "mean"),
        n_comp=("n_compositions", "mean"),
        n_lp=("n_linker_pairs", "mean"),
        MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
    ).reset_index()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n若 (e) 贴近 (a) ⇒ 多样性数量驱动；贴近 (b) ⇒ matched-partner 覆盖驱动。")


if __name__ == "__main__":
    main()
