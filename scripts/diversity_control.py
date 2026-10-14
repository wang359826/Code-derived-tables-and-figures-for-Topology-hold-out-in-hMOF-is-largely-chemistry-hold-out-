#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多样性匹配对照臂：把 §3.4 受控对照的"训练化学多样性"这一替代解释闭合
（回应审计清单 P1-3）

现有设计
--------
§3.4 的 pcu ↔ nbo 对照改变的不止一件事：
    (a) with_pair    : 除 nbo 外的全部拓扑            n_train = 42,329
    (b) without_pair : 除 nbo 与 pcu 外的全部拓扑      n_train = 10,647
    (c) size_matched : 从 (a) 池中随机抽 10,647 个       n_train = 10,647
(c) 已经排除了"只是样本变少"，但**没有排除"训练化学多样性下降"**：
移除 pcu 同时移除了库中约 63% 的化学多样性。§4.4 如实承认了这一点。

本脚本补上第四个臂
------------------
    (d) composition_matched : 从 (a) 池中抽取与 (b) **同样多**的样本，但按拓扑
        分层，使各拓扑的样本占比与 (b) 一致。因为一个拓扑的化学组成几乎不跨
        拓扑，(b) 的拓扑剖面就唯一决定了它的构筑单元组成剖面，所以这个臂与 (b)
        在金属 / 连接体 / 官能团层面是匹配的。

关系与读法
----------
(c) 是**均匀**抽样，因此按比例带入 pcu（约占池的 60%），既匹配了样本量、也
顺带保留了 pcu 的化学覆盖 → 它在两个测试拓扑上都接近 (a)。
(d) 匹配了 (b) 的组成剖面 → 它应当接近 (b)，这正是抽样器正确性的检验。

因此 (a)–(d) 四臂一起读：
  · 若 MAE(d) ≈ MAE(b) 而 MAE(c) ≈ MAE(a)，则降解随"训练集里有没有与测试
    群体同源的构筑单元组成"而移动，而不是随"训练集看起来有多少种化学"移动；
  · 但同样的对比也说明，本文的数据**无法**把"构筑单元多样性"与"配对拓扑覆盖"
    分开——在 hMOF 里 pcu 既提供了 63% 的多样性、也是 nbo 唯一的化学配对拓扑，
    两者不可分别操纵。§4.4 如实保留这一条。

同时报告每个训练臂的四个多样性度量（#拓扑 / #金属种类 / #连接体对 / #组成），
使"组成匹配"可核验，而不是一句声明。

用法
----
  python diversity_control.py --csv data/all_MOFs_screening_data.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

TARGET = "CO2_uptake_P0.15bar_T298K [mmol/g]"
NUMERIC_FEATURES = [
    "volume [A^3]",
    "weight [u]",
    "surface_area [m^2/g]",
    "void_fraction",
    "void_volume [cm^3/g]",
    "largest_free_sphere_diameter [A]",
    "largest_included_sphere_along_free_sphere_path_diameter [A]",
    "largest_included_sphere_diameter [A]",
]
CATEGORICAL_FEATURES = ["functional_groups", "metal_linker", "topology"]
KEYS_COARSE = ["metal_linker", "organic_linker1", "organic_linker2"]
N_ESTIMATORS = 120
N_ROWS = 50_000
SEED_LIST = [42, 143, 244, 345, 446]
CONTROL_TOPOS = ["pcu", "nbo"]


def load_and_build(csv_path: str, seed: int):
    from sklearn.ensemble import RandomForestRegressor  # noqa: F401  (import check)
    df = pd.read_csv(csv_path)
    if N_ROWS < len(df):
        df = df.sample(N_ROWS, random_state=seed).reset_index(drop=True)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    X = df[cols].copy().astype(float)
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            X = pd.concat([X, pd.get_dummies(df[c].astype(str),
                                             prefix=c).astype(float)], axis=1)
    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    d = df.loc[mask].reset_index(drop=True)
    ck = d[KEYS_COARSE].fillna("NA").astype(str).agg("|".join, axis=1)
    lp = (d["organic_linker1"].astype(str) + "_"
          + d["organic_linker2"].astype(str))
    return X, y, d["topology"].astype(str), ck, lp, d["metal_linker"].astype(str)


def diversify_quota(pool_topo: pd.Series, target_topo: pd.Series,
                    n_target: int, rng: np.random.Generator,
                    pool_index: pd.Index):
    """按 target_topo 的拓扑剖面，从 pool_topo 中分层抽取 n_target 个索引。

    pool_index 是 pool_topo 对应的**原始行号**（pool_topo 已被 reset_index）。
    """
    pool_vals = pool_topo.to_numpy()
    by_group = {t: np.where(pool_vals == t)[0] for t in pd.unique(pool_vals)}
    want = target_topo.value_counts()
    frac = want / want.sum()
    picked = []
    shortfall = 0
    for t, f in frac.items():
        need = int(round(f * n_target))
        avail = by_group.get(t, np.array([], dtype=int))
        if len(avail) == 0:
            shortfall += need
            continue
        take = min(need, len(avail))
        shortfall += need - take
        picked.append(rng.choice(avail, size=take, replace=False))
    if shortfall > 0:
        chosen = np.concatenate(picked) if picked else np.array([], dtype=int)
        rest = np.setdiff1d(np.arange(len(pool_vals)), chosen)
        extra = rng.choice(rest, size=min(shortfall, len(rest)), replace=False)
        picked.append(extra)
    local = np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)
    return pool_index.to_numpy()[local]


def diversity_stats(idx, topo, ck, lp, metal) -> dict:
    return {"n_train": int(len(idx)),
            "n_topologies": int(topo.iloc[idx].nunique()),
            "n_metal_linkers": int(metal.iloc[idx].nunique()),
            "n_linker_pairs": int(lp.iloc[idx].nunique()),
            "n_compositions": int(ck.iloc[idx].nunique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        HERE, "..", "data", "all_MOFs_screening_data.csv"))
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    for seed in SEED_LIST:
        X, y, topo, ck, lp, metal = load_and_build(args.csv, seed)
        topo = topo.reset_index(drop=True)
        ck = ck.reset_index(drop=True)
        lp = lp.reset_index(drop=True)
        metal = metal.reset_index(drop=True)
        print(f"\n{'='*78}\nseed = {seed}   建模样本 {len(X):,}\n{'='*78}")

        for target_topo in CONTROL_TOPOS:
            te = np.where(topo.to_numpy() == target_topo)[0]
            if len(te) < 50:
                print(f"  {target_topo}: 测试样本仅 {len(te)}，跳过")
                continue
            # 配对拓扑 = 与之共享化学组成最多的拓扑
            sets = {t: set(ck[topo == t]) for t in topo.unique()}
            cand = sorted(((t, len(sets[target_topo] & sets[t]))
                           for t in sets if t != target_topo),
                          key=lambda x: -x[1])
            partner, n_shared = cand[0]
            print(f"\n  测试拓扑 {target_topo}（n_test={len(te):,}）"
                  f"  配对拓扑 = {partner}（共享 {n_shared} 个组成）")

            te_mask = np.zeros(len(topo), dtype=bool)
            te_mask[te] = True
            idx_a = np.where(~te_mask)[0]
            idx_b = np.where((~te_mask) &
                             (topo.to_numpy() != partner))[0]

            def fit_eval(idx, regime):
                if len(idx) < 200:
                    return None
                m = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                          random_state=seed, n_jobs=1)
                m.fit(X.iloc[idx], y.iloc[idx])
                p = m.predict(X.iloc[te])
                yt = y.iloc[te].to_numpy()
                sd = float(yt.std())
                r = {"seed": seed, "test_topo": target_topo, "regime": regime,
                     "R2": float(r2_score(yt, p)),
                     "MAE": float(mean_absolute_error(yt, p)),
                     "RMSE": float(np.sqrt(mean_squared_error(yt, p))),
                     "MAE_over_sd": float(mean_absolute_error(yt, p) / sd)}
                r.update(diversity_stats(idx, topo, ck, lp, metal))
                print(f"    {regime:22s} n_train={r['n_train']:6d}  "
                      f"MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                      f"#topo={r['n_topologies']:3d}  "
                      f"#lp={r['n_linker_pairs']:4d}  "
                      f"#comp={r['n_compositions']:6d}")
                return r

            rows.append(fit_eval(idx_a, "with_pair"))
            rows.append(fit_eval(idx_b, "without_pair"))

            n_b = len(idx_b)
            # (c) 样本量匹配：从 (a) 池中**均匀**抽 n_b 个（5 次）。
            #     注意：均匀抽样会按比例带入 pcu（约占池的 60%），因此这个臂
            #     既匹配了样本量，也顺带保留了 pcu 的化学覆盖。
            rng = np.random.default_rng(seed + 1000)
            maes, rmses = [], []
            for i in range(5):
                pick = rng.choice(idx_a, size=n_b, replace=False)
                r = fit_eval(pick, f"size_matched_seed{i}")
                if r:
                    rows.append(r)
                    maes.append(r["MAE"])
                    rmses.append(r["RMSE"])
            # (d) 分化匹配：从 (a) 池中按 (b) 的拓扑剖面分层抽 n_b 个（5 次）。
            #     因为一个拓扑的化学组成几乎不跨拓扑，(b) 的拓扑剖面就唯一决定
            #     了它的"构筑单元组成剖面"，所以这个臂与 (b) 在金属/连接体/官能团
            #     层面是匹配的，差别只在于具体抽到哪些行。
            pool_topo = topo.iloc[idx_a].reset_index(drop=True)
            tgt_topo = topo.iloc[idx_b].reset_index(drop=True)
            maes_d, rmses_d = [], []
            for i in range(5):
                rg = np.random.default_rng(seed + 2000 + i)
                pick = diversify_quota(pool_topo, tgt_topo, n_b, rg,
                                       pd.Index(idx_a))
                r = fit_eval(pick, f"composition_matched_seed{i}")
                if r:
                    rows.append(r)
                    maes_d.append(r["MAE"])
                    rmses_d.append(r["RMSE"])
            if maes_d:
                print(f"    → size-matched  MAE {np.mean(maes):.4f}±"
                      f"{np.std(maes, ddof=1):.4f}   "
                      f"composition-matched MAE {np.mean(maes_d):.4f}±"
                      f"{np.std(maes_d, ddof=1):.4f}")

    d = pd.DataFrame([r for r in rows if r])
    p = os.path.join(args.outdir, "composition_matched_control.csv")
    d.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p}")

    # ---- 汇总：把 (a)(b)(c)(d) 四个臂的 5 种子均值并排列出 ----
    print("\n" + "=" * 78)
    print("四臂汇总（跨 5 个种子）")
    print("=" * 78)
    d["arm"] = d["regime"].str.replace(r"_seed\d+$", "", regex=True)
    agg = (d.groupby(["test_topo", "arm"])
             .agg(n_seeds=("seed", "nunique"),
                  n_train=("n_train", "mean"),
                  MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
                  RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                  n_topologies=("n_topologies", "mean"),
                  n_linker_pairs=("n_linker_pairs", "mean"),
                  n_compositions=("n_compositions", "mean"))
             .reset_index())
    print(agg.round(4).to_string(index=False))
    pa = os.path.join(args.outdir, "composition_matched_summary.csv")
    agg.to_csv(pa, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {pa}")


if __name__ == "__main__":
    main()
