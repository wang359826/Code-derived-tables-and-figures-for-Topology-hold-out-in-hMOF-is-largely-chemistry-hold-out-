#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
§3.6「三个直观解释被否证」的分项占比：一次可核验的重算
（回应审计清单 P2-4：把 §3.6 从一句结论扩成"关键数 + 强度"）

正文 §3.6 现在写：最近邻距离解释 26%、特征族重叠解释 16%、SHAP 归因在两种
划分下几乎不变。前两个数字此前由不同的脚本在**不同抽样**上得到，本脚本把
三个量放在**同一次 25,000 抽样、同一个 5 折随机划分、同一个模型**下重算，
使它们可以并排引用（并与 §3.1 口径一致：n_estimators = 120）。

定义（全部以 k = 5 折随机划分为参照臂）：
  ΔR²  := R²(随机划分) − R²(留一拓扑)
  H1 近邻距离：按每折内测试样本到训练集的最近邻距离分箱，取距离 >= 留一拓扑
     中位数的子集，看它的 R² 追回了多少 ΔR²。
  H2 构筑单元家族：家族 = metal_linker + organic_linker1 + organic_linker2 +
     functional_groups（**不含 topology**，否则留一拓扑下该量恒为 0）。
     "家族已见"比"家族未见"多解释多少 R²。
  H3 身份特征的 SHAP 归因占比：随机划分 vs 留一拓扑。

用法
----
  python refuted_explanations.py --csv data/all_MOFs_screening_data.csv
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
    "volume [A^3]", "weight [u]", "surface_area [m^2/g]", "void_fraction",
    "void_volume [cm^3/g]", "largest_free_sphere_diameter [A]",
    "largest_included_sphere_along_free_sphere_path_diameter [A]",
    "largest_included_sphere_diameter [A]",
]
CATS = ["functional_groups", "metal_linker", "topology"]
N_EST = 120
SEED = 42


def build(csv_path: str, max_rows: int):
    df = pd.read_csv(csv_path)
    if max_rows and max_rows < len(df):
        df = df.sample(max_rows, random_state=SEED).reset_index(drop=True)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    X = df[NUMERIC_FEATURES].copy().astype(float)
    for c in CATS:
        X = pd.concat([X, pd.get_dummies(df[c].astype(str),
                                         prefix=c).astype(float)], axis=1)
    mask = y.notna() & X.notna().all(axis=1)
    return (X[mask].reset_index(drop=True), y[mask].reset_index(drop=True),
            df.loc[mask].reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(
        HERE, "..", "data", "all_MOFs_screening_data.csv"))
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--outdir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, LeaveOneGroupOut
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    X, y, d = build(args.csv, args.max_rows)
    print(f"建模样本 {len(X)}  特征 {X.shape[1]}")
    topo = d["topology"].astype(str).reset_index(drop=True)
    fam = (d["metal_linker"].astype(str) + "|" + d["organic_linker1"].astype(str)
           + "|" + d["organic_linker2"].astype(str) + "|"
           + d["functional_groups"].astype(str))
    Z = StandardScaler().fit_transform(X[NUMERIC_FEATURES].values)

    schemes = {
        "random": list(KFold(n_splits=5, shuffle=True,
                             random_state=SEED).split(X)),
        "leave_one_topology": list(LeaveOneGroupOut().split(
            X, y, groups=topo.values)),
    }
    samples = []
    for name, folds in schemes.items():
        for tr, te in folds:
            if len(te) == 0:
                continue
            m = RandomForestRegressor(n_estimators=N_EST, random_state=SEED,
                                      n_jobs=1)
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
            nn = NearestNeighbors(n_neighbors=1).fit(Z[tr])
            dist, _ = nn.kneighbors(Z[te])
            seen = fam.iloc[te].isin(set(fam.iloc[tr])).to_numpy()
            samples.append(pd.DataFrame({
                "scheme": name, "nn_dist": dist.ravel(),
                "family_seen": seen,
                "y_true": y.iloc[te].to_numpy(), "y_pred": pred}))
    S = pd.concat(samples, ignore_index=True)

    def r2(sub):
        return float(r2_score(sub["y_true"], sub["y_pred"])) if len(sub) >= 30 else np.nan

    r_rand = r2(S[S.scheme == "random"])
    r_topo = r2(S[S.scheme == "leave_one_topology"])
    dR2 = r_rand - r_topo
    print(f"\nR²(random) = {r_rand:.4f}   R²(leave-one-topology) = {r_topo:.4f}"
          f"   ΔR² = {dR2:.4f}")

    # ---- H1：近邻距离 ----
    med_t = float(S.loc[S.scheme == "leave_one_topology", "nn_dist"].median())
    r_far = r2(S[(S.scheme == "random") & (S.nn_dist >= med_t)])
    closed_nn = (r_rand - r_far) / dR2
    print(f"\nH1 nearest-neighbour distance")
    print(f"   median NN distance: random {S.loc[S.scheme=='random','nn_dist'].median():.4f}"
          f"  vs leave-one-topology {med_t:.4f}")
    print(f"   random-split R² restricted to distance >= topology median: {r_far:.4f}")
    print(f"   → distance explains {closed_nn*100:.0f}% of ΔR²")

    # ---- H2：构筑单元家族 ----
    print(f"\nH2 building-block family (metal+linkers+functional_groups, no topology)")
    out_rows = [{"hypothesis": "H1 nearest-neighbour distance",
                 "metric": "share of ΔR² explained", "value": closed_nn}]
    for scheme in ("random", "leave_one_topology"):
        s = S[S.scheme == scheme]
        print(f"   {scheme:20s} family seen: {s.family_seen.mean():.3f} of samples"
              f"   R²(seen) = {r2(s[s.family_seen]):.4f}"
              f"   R²(unseen) = {r2(s[~s.family_seen]):.4f}")
    s = S[S.scheme == "random"]
    closed_fam = (r2(s[s.family_seen]) - r2(s[~s.family_seen])) / dR2
    print(f"   → family-seen status explains {closed_fam*100:.0f}% of ΔR²")
    out_rows.append({"hypothesis": "H2 building-block family overlap",
                     "metric": "share of ΔR² explained", "value": closed_fam})

    # ---- H3：SHAP 归因（可选，依赖 shap）----
    try:
        import shap
        ident_share = {}
        for scheme in ("random", "leave_one_topology"):
            tr = schemes[scheme][0][0]
            m = RandomForestRegressor(n_estimators=N_EST, random_state=SEED,
                                      n_jobs=1).fit(X.iloc[tr], y.iloc[tr])
            expl = shap.TreeExplainer(m)
            n_sub = min(2000, len(tr))
            sv = expl.shap_values(X.iloc[tr[:n_sub]])
            imp = np.abs(sv).mean(axis=0)
            names = np.array(X.columns)
            # 2026-10-16 修：原掩码名为 ident，但它 True 时对应的是**结构描述符**
            # （8 个数值列），于是 imp[~ident] 求出的其实是 identity（类别）份额，
            # 却被打印成 "structure-derived share"，并把 1-shares 当作
            # "identity share" 写进 CSV —— 打印与 CSV 的标签都是反的。
            # 现改名 is_structure 并把标签对正：CSV 从此直接记录 identity 份额。
            # 注意：2026-10-14 归档的那份 CSV 产生于本次修复之前，它记录的 83.1/82.6
            # 是**结构描述符**份额，与本脚本今后写出的值互补，二者不可互换引用。
            is_structure = np.array([not n.startswith(tuple(c + "_" for c in CATS))
                                     for n in names])
            ident_share[scheme] = float(imp[~is_structure].sum() / imp.sum())
            print(f"\nH3 SHAP: identity share under {scheme}: "
                  f"{ident_share[scheme]*100:.1f}%  "
                  f"(structure-derived {100-ident_share[scheme]*100:.1f}%)")
        out_rows.append({"hypothesis": "H3 identity attribution (identity share, random)",
                         "metric": "identity share of |SHAP|",
                         "value": ident_share["random"]})
        out_rows.append({"hypothesis": "H3 identity attribution (identity share, hold-out)",
                         "metric": "identity share of |SHAP|",
                         "value": ident_share["leave_one_topology"]})
    except Exception as exc:  # noqa: BLE001
        print(f"\nH3 skipped (shap unavailable or failed): {exc}")

    summary = pd.DataFrame(out_rows)
    p = os.path.join(args.outdir, "refuted_explanations_summary.csv")
    summary.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p}")


if __name__ == "__main__":
    main()
