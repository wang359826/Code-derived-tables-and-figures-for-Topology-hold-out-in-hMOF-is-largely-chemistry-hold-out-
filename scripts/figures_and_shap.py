#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 5（预测 vs 实测双面板）+ Figure S1/S2（SHAP 归因分析）
============================================================

（2026-10-11 图序修订后：原 Figure 4/5/6 分别变为 Figure 5 / Figure S1 / Figure S2；
CSV 输出文件名沿用存档名不变，以便与 MD5SUMS 存档一致。）

本脚本回答两个审稿人必问的问题：

Q1. "R² 掉那么多，是不是少数离群点造成的？"
    → Figure 5：两张散点图并排。左 = 随机划分（泄漏），右 = 留一拓扑。
      如果是离群点驱动，点云会呈"少数点拉偏"的形态；如果是系统性
      外推失败，点云会整体变宽、且在高吸附量区出现明显塌缩。

Q2. "泄漏到底是怎么发生的？模型记住了什么？"
    → Figure S1：SHAP 归因按**特征族**聚合，对比两种划分下的归因结构。
      核心假设：随机划分下模型大量依赖构筑单元身份的 one-hot
      （topology / metal_linker / functional_groups）；
      留一拓扑下这些身份特征对新拓扑失效，模型被迫退回几何描述符。
      这个"归因转移"就是泄漏的机理证据——比只报一个 ΔR² 强得多。

    → Figure S2：几何描述符的 SHAP 依赖图，验证模型学到的物理是否合理
      （例如 void_fraction 越大、吸附量越低，符合低压下限域吸附增强的预期）。

运行
----
    # 先小样本跑通
    python figures_and_shap.py --max-rows 8000 --shap-samples 100 --tag smoke

    # 正式出图（约 15-40 分钟）
    python figures_and_shap.py --max-rows 25000 --shap-samples 200 --tag final

输出
----
  fig5_pred_vs_actual[_tag].png      双面板散点（Figure 5）
  figS1_shap_family[_tag].png        特征族归因对比（Figure S1）
  fig5_shap_family[_tag].csv         对应的数值表（Table S1；文件名沿用存档名）
  figS2_shap_dependence[_tag].png    几何描述符依赖图（Figure S2）
  fig4_predictions[_tag].csv         两种划分下的 (真实值, 预测值, 分组)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# 与 mof_leakage_benchmark.py 保持一致，保证数字能对上
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
N_ESTIMATORS = 120
RANDOM_STATE = 42

# 几何描述符的中文标签（Figure 6 用；图内文字用英文以免字体乱码）
GEO_LABELS = {
    "volume [A^3]": "volume",
    "weight [u]": "weight",
    "surface_area [m^2/g]": "surface area",
    "void_fraction": "void fraction",
    "void_volume [cm^3/g]": "void volume",
    "largest_free_sphere_diameter [A]": "largest free sphere d",
    "largest_included_sphere_along_free_sphere_path_diameter [A]":
        "largest incl. sphere (path) d",
    "largest_included_sphere_diameter [A]": "largest incl. sphere d",
}


def _make_model():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)


def load_data(csv_path, max_rows):
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 行")
    if max_rows and max_rows < len(df):
        # 必须随机采样：文件按拓扑排序，head() 会只拿到少数几种拓扑
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"随机采样 {max_rows} 行")
    return df


def build_features(df, target=TARGET):
    """构造特征矩阵（几何 + 类别 one-hot），与基准脚本完全一致。"""
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    y = pd.to_numeric(df[target], errors="coerce")
    X = df[cols].copy().astype(float)
    cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    for c in cats:
        X = pd.concat([X, pd.get_dummies(df[c].astype(str),
                                         prefix=c).astype(float)], axis=1)
    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
    dfc = df.loc[mask].reset_index(drop=True)
    groups = {
        "topology": dfc["topology"].astype(str),
        "metal": dfc["metal_linker"].astype(str),
    }
    return X, y, groups, cats, dfc


def family_of(col: str, cats) -> str:
    """把特征列归到"特征族"，用于聚合 SHAP 归因。"""
    for c in cats:
        if col.startswith(c + "_"):
            return c
    return "structure-derived descriptors"


# --------------------------------------------------------------------------
# Figure 4
# --------------------------------------------------------------------------
def collect_predictions(X, y, groups, n_splits=5):
    """两种划分下的袋外/折外预测。返回 DataFrame。"""
    from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

    rows = []

    # 左面板：5 折随机划分
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in kf.split(X):
        mdl = _make_model()
        mdl.fit(X.iloc[tr], y.iloc[tr])
        pred = mdl.predict(X.iloc[te])
        rows.append(pd.DataFrame({"y_true": y.iloc[te].values,
                                  "y_pred": pred,
                                  "split": "random"}))

    # 右面板：留一拓扑（每次留出一种拓扑，预测值汇总）
    logo = LeaveOneGroupOut()
    g = groups["topology"].values
    for tr, te in logo.split(X, y, groups=g):
        if len(te) == 0:
            continue
        mdl = _make_model()
        mdl.fit(X.iloc[tr], y.iloc[tr])
        pred = mdl.predict(X.iloc[te])
        rows.append(pd.DataFrame({"y_true": y.iloc[te].values,
                                  "y_pred": pred,
                                  "split": "unseen_topology"}))
    return pd.concat(rows, ignore_index=True)


def figure4(pred_df, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    panels = [("random", "Random split (leaky)", "#A32D2D"),
              ("unseen_topology", "Leave-one-topology-out", "#185FA5")]

    for ax, (key, title, color) in zip(axes, panels):
        d = pred_df[pred_df["split"] == key]
        r2 = r2_score(d["y_true"], d["y_pred"])
        rmse = float(np.sqrt(mean_squared_error(d["y_true"], d["y_pred"])))

        # 点太多，用散点 + 透明度；加 2D 直方图底更好看，但散点更诚实
        ax.scatter(d["y_true"], d["y_pred"], s=3, alpha=0.12,
                   color=color, edgecolors="none", rasterized=True)
        lo = float(min(d["y_true"].min(), d["y_pred"].min()))
        hi = float(max(d["y_true"].max(), d["y_pred"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True CO₂ uptake [mmol/g]")
        ax.set_title(f"{title}\nR² = {r2:.3f}   RMSE = {rmse:.3f}   n = {len(d)}",
                     fontsize=10.5)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.22, lw=0.5)

    axes[0].set_ylabel("Predicted CO₂ uptake [mmol/g]")
    fig.suptitle("Same model, same data — only the split changes", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"已保存: {outpath}")


# --------------------------------------------------------------------------
# Figure 5 + Table 4：SHAP 归因按特征族聚合
# --------------------------------------------------------------------------
def shap_family_attribution(X, y, groups, cats, n_splits=5,
                            shap_samples=200, scheme="random"):
    """在给定划分方案下，计算 SHAP 归因并按特征族聚合。

    scheme:
      random   —— 5 折随机划分，每折在测试集抽样算 SHAP
      topology —— 留一拓扑，每折在留出的拓扑上抽样算 SHAP
    """
    import shap
    from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

    fam = np.array([family_of(c, cats) for c in X.columns])
    families = ["structure-derived descriptors"] + list(cats)
    acc = {f: 0.0 for f in families}
    n_folds = 0

    if scheme == "random":
        splits = list(KFold(n_splits=n_splits, shuffle=True,
                            random_state=RANDOM_STATE).split(X))
    else:
        splits = list(LeaveOneGroupOut().split(X, y, groups=groups["topology"].values))

    for tr, te in splits:
        if len(te) == 0:
            continue
        mdl = _make_model()
        mdl.fit(X.iloc[tr], y.iloc[tr])

        # 每折只抽一部分测试样本算 SHAP，控制耗时
        idx = np.random.RandomState(RANDOM_STATE + n_folds).choice(
            te, size=min(shap_samples, len(te)), replace=False)
        Xs = X.iloc[idx]
        try:
            sv = shap.TreeExplainer(mdl).shap_values(Xs)
        except Exception as exc:  # noqa: BLE001
            print(f"    SHAP 失败（折 {n_folds}）: {exc}")
            n_folds += 1
            continue
        sv = np.asarray(sv)
        if sv.ndim == 3:                      # 某些版本的返回形状
            sv = sv[..., 0] if sv.shape[-1] == 1 else sv.mean(axis=-1)
        mean_abs = np.abs(sv).mean(axis=0)
        for f in families:
            acc[f] += float(mean_abs[fam == f].sum())
        n_folds += 1

    if n_folds == 0:
        return None
    tot = sum(acc.values())
    return pd.DataFrame({
        "family": families,
        "mean_abs_shap": [acc[f] / n_folds for f in families],
        "fraction": [acc[f] / tot if tot > 0 else np.nan for f in families],
        "scheme": scheme,
        "n_folds": n_folds,
    })


def figure5(df_rand, df_topo, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = df_rand["family"].tolist()
    x = np.arange(len(order))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # 左：绝对归因量
    ax1.bar(x - w / 2, df_rand["mean_abs_shap"], w,
            color="#A32D2D", label="random split")
    ax1.bar(x + w / 2, df_topo["mean_abs_shap"], w,
            color="#185FA5", label="leave-one-topology-out")
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=18, ha="right", fontsize=9)
    ax1.set_ylabel("mean |SHAP| (per fold)")
    ax1.set_title("Absolute attribution by feature family", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25, lw=0.5)

    # 右：占比（更直观看出"归因转移"）
    ax2.bar(x - w / 2, df_rand["fraction"] * 100, w,
            color="#A32D2D", label="random split")
    ax2.bar(x + w / 2, df_topo["fraction"] * 100, w,
            color="#185FA5", label="leave-one-topology-out")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order, rotation=18, ha="right", fontsize=9)
    ax2.set_ylabel("share of total attribution [%]")
    ax2.set_title("Relative attribution by feature family", fontsize=10.5)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("Where the model looks barely changes — geometry dominates either way\n"
                 "(so the leak is NOT 'the model memorises building-block identity')",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"已保存: {outpath}")


# --------------------------------------------------------------------------
# Figure 6：几何描述符的 SHAP 依赖图
# --------------------------------------------------------------------------
def figure6(X, y, cats, top_geo=4, sample_n=3000, outpath="fig6.png"):
    """在随机划分拟合的模型上，画前几个几何描述符的 SHAP 依赖图。"""
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mdl = _make_model()
    mdl.fit(X, y)

    idx = np.random.RandomState(RANDOM_STATE).choice(
        len(X), size=min(sample_n, len(X)), replace=False)
    Xs = X.iloc[idx]
    try:
        sv = shap.TreeExplainer(mdl).shap_values(Xs)
    except Exception as exc:  # noqa: BLE001
        print(f"  Figure 6 跳过：SHAP 失败 {exc}")
        return
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 0] if sv.shape[-1] == 1 else sv.mean(axis=-1)

    # 只画几何描述符（one-hot 的依赖图没有物理意义）
    geo_cols = [c for c in X.columns if family_of(c, cats) == "structure-derived descriptors"]
    imp = np.abs(sv).mean(axis=0)
    rank = sorted(zip(X.columns, imp), key=lambda t: -t[1])
    picked = [c for c, _ in rank if c in set(geo_cols)][:top_geo]

    n = len(picked)
    if n == 0:
        print("  Figure 6 跳过：没有几何描述符")
        return
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for ax, col in zip(axes[0], picked):
        j = list(X.columns).index(col)
        ax.scatter(Xs[col].values, sv[:, j], s=5, alpha=0.25,
                   color="#185FA5", edgecolors="none", rasterized=True)
        ax.axhline(0, color="#888888", lw=0.7)
        ax.set_xlabel(GEO_LABELS.get(col, col), fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"{GEO_LABELS.get(col, col)}\n"
                     f"mean|SHAP| = {np.abs(sv[:, j]).mean():.3f}", fontsize=9.5)
        ax.grid(alpha=0.2, lw=0.5)
    fig.suptitle("SHAP dependence: are the learned trends physically sensible?",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"已保存: {outpath}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shap-samples", type=int, default=200,
                    help="每折取多少测试样本算 SHAP（越大越准越慢）")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--skip-shap", action="store_true", help="只画 Figure 4")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    print("=" * 72)
    print("Figure 4 / 5 / 6：预测散点 + SHAP 归因分析")
    print("=" * 72)

    df = load_data(args.csv, args.max_rows)
    X, y, groups, cats, dfc = build_features(df, args.target)
    print(f"建模样本 {len(X)} 行，{X.shape[1]} 个特征")
    print(f"拓扑: {groups['topology'].nunique()} 种  "
          f"金属: {groups['metal'].nunique()} 种")

    # ---- Figure 5 ----
    print("\n[1/3] Figure 5：收集两种划分下的预测值 ...")
    t0 = time.time()
    pred_df = collect_predictions(X, y, groups)
    p4 = os.path.join(args.outdir, f"fig5_pred_vs_actual{suffix}.png")
    figure4(pred_df, p4)
    pcsv = os.path.join(args.outdir, f"fig4_predictions{suffix}.csv")
    pred_df.to_csv(pcsv, index=False, encoding="utf-8-sig")
    print(f"已保存: {pcsv}   耗时 {time.time()-t0:.0f} s")

    if args.skip_shap:
        print("\n--skip-shap：已跳过 SHAP 部分")
        return

    # ---- Figure S1 + Table S1 ----
    print("\n[2/3] Figure S1：SHAP 归因按特征族聚合 ...")
    t0 = time.time()
    df_rand = shap_family_attribution(X, y, groups, cats,
                                      shap_samples=args.shap_samples,
                                      scheme="random")
    print(f"  随机划分完成（{time.time()-t0:.0f} s）")
    t0 = time.time()
    df_topo = shap_family_attribution(X, y, groups, cats,
                                      shap_samples=args.shap_samples,
                                      scheme="topology")
    print(f"  留一拓扑完成（{time.time()-t0:.0f} s）")

    if df_rand is None or df_topo is None:
        print("  SHAP 归因失败，跳过 Figure S1")
    else:
        both = pd.concat([df_rand, df_topo], ignore_index=True)
        csv5 = os.path.join(args.outdir, f"fig5_shap_family{suffix}.csv")
        both.to_csv(csv5, index=False, encoding="utf-8-sig")
        print("\n  特征族归因对比（Table S1）：")
        show = both.pivot(index="family", columns="scheme",
                          values="fraction").reset_index()
        show.columns = ["family", "frac_random", "frac_topology"]
        print(show.to_string(index=False))
        print(f"\n已保存: {csv5}")
        figure5(df_rand, df_topo,
                os.path.join(args.outdir, f"figS1_shap_family{suffix}.png"))

    # ---- Figure S2 ----
    print("\n[3/3] Figure S2：几何描述符 SHAP 依赖图 ...")
    figure6(X, y, cats,
            outpath=os.path.join(args.outdir, f"figS2_shap_dependence{suffix}.png"))

    print("\n完成。")


if __name__ == "__main__":
    main()
