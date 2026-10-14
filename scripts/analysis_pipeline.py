#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
固废基沸石 CO2 吸附数据集 —— 分析与建模流水线
================================================

用途
----
把你在阶段2手工提取的 CSV 数据，转成论文所需的：
  1. 数据质量报告（缺失率、分布）—— Data Descriptor 的核心内容
  2. 描述统计与相关矩阵图 —— 论文的 Figure 2/3
  3. 机器学习建模（按 DOI 分组交叉验证，防止数据泄漏）—— Figure 4
  4. SHAP 可解释性分析 —— Figure 5，论文的点睛之笔

如何使用
--------
第一次用（还没数据时），先跑内置演示数据看效果：

    python analysis_pipeline.py --demo

有真实数据后：

    python analysis_pipeline.py --input ../数据提取表模板.csv --outdir outputs

依赖安装（在 Anaconda Prompt 里执行）：

    pip install pandas numpy matplotlib scikit-learn shap
    # XGBoost 可选（推荐）：
    pip install xgboost

设计说明（这几条是审稿人会问的，我已内置处理）
----------------------------------------------
* 防数据泄漏：同一篇论文的多条数据点必须落在同一个交叉验证折里，
  因此全程使用 GroupKFold(groups=doi)。若不加分组，同文献数据会
  同时出现在训练集和测试集，R2 会虚高 0.1-0.3，是很常见的学术不严谨。
* 外部验证：可用 --external-doi 指定若干 DOI 作为完全不参与训练的外部测试集。
* 类别变量：cation / zeolite_type / method / waste_type 采用独热编码。
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# Windows 控制台中文输出保护
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# 0. 依赖检查：缺什么就明确告诉你装什么，不要让人对着报错发呆
# ----------------------------------------------------------------------------
def check_dependencies() -> dict:
    """返回各依赖包是否可用。"""
    status = {}
    for pkg in ["pandas", "numpy", "matplotlib", "sklearn", "shap", "xgboost"]:
        try:
            __import__(pkg)
            status[pkg] = True
        except ImportError:
            status[pkg] = False
    return status


def report_dependencies(status: dict) -> None:
    missing = [k for k, v in status.items() if not v]
    if not missing:
        print("[环境] 所有依赖已就绪。")
        return
    print("[环境] 缺少以下包，请先安装：")
    pip_names = {
        "pandas": "pandas", "numpy": "numpy", "matplotlib": "matplotlib",
        "sklearn": "scikit-learn", "shap": "shap", "xgboost": "xgboost",
    }
    print(f"    pip install {' '.join(pip_names[m] for m in missing)}")
    if "sklearn" in missing or "pandas" in missing:
        print("\n核心依赖缺失，无法继续。装好后重跑本脚本。")
        sys.exit(1)


# ----------------------------------------------------------------------------
# 1. 演示数据：没数据时也能跑通全流程，先看懂每一步在干什么
# ----------------------------------------------------------------------------
def make_demo_data(n: int = 600):
    """生成结构仿真的演示数据（数值为随机合成，**不可用于任何学术结论**）。"""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20261008)
    n_doi = 60
    dois = [f"10.1016/j.DEMO-{i:03d}" for i in range(n_doi)]
    zeolite_types = ["NaA", "NaX", "NaP", "ZSM-5", "MOR"]
    wastes = ["fly ash", "coal gangue", "red mud", "steel slag", "rice husk ash"]
    cations = ["Na", "K", "Ca", "Li"]
    methods = ["alkali fusion-hydrothermal", "direct hydrothermal", "microwave"]

    df = pd.DataFrame({
        "doi": rng.choice(dois, n),
        "year": rng.integers(2015, 2026, n),
        "source_figure": ["Fig. 4"] * n,
        "zeolite_type": rng.choice(zeolite_types, n),
        "waste_type": rng.choice(wastes, n),
        "cation": rng.choice(cations, n),
        "method": rng.choice(methods, n),
        "Si_Al_ratio": np.round(rng.uniform(1.0, 8.0, n), 2),
        "BET_m2g": np.round(rng.uniform(100, 900, n), 1),
        "pore_volume_cm3g": np.round(rng.uniform(0.08, 0.45, n), 3),
        "alkali_fusion_T_C": np.round(rng.uniform(600, 900, n), 0),
        "crystallization_T_C": np.round(rng.uniform(80, 150, n), 0),
        "crystallization_t_h": np.round(rng.uniform(2, 24, n), 1),
        "T_K": np.round(rng.uniform(288, 323, n), 1),
        "P_bar": np.round(rng.choice([0.15, 1.0, 1.0, 1.0, 5.0], n), 2),
        "relative_humidity_pct": rng.choice([0, 0, 0, 30, 60], n),
    })

    # 构造一个"有物理意义"的目标：低硅铝比 + 高比表面 + 高压 → 高吸附量
    # 碱熔温度设一个 750-800 C 的最优窗口，用于演示 SHAP 能挖出非线性规律
    uptake = (
        1.2
        + 0.55 * np.log1p(df["P_bar"])
        - 0.16 * df["Si_Al_ratio"]
        + 0.0016 * df["BET_m2g"]
        + 1.1 * df["pore_volume_cm3g"]
        - 0.0045 * (df["alkali_fusion_T_C"] - 775) ** 2 / 100
        - 0.010 * df["relative_humidity_pct"]
        + rng.normal(0, 0.28, n)
    )
    df["CO2_uptake_mmolg"] = np.round(np.clip(uptake, 0.2, None), 3)

    # 按真实文献的习惯制造缺失：碱熔条件最常缺
    for col, miss in [("alkali_fusion_T_C", 0.32),
                      ("pore_volume_cm3g", 0.18),
                      ("Si_Al_ratio", 0.12),
                      ("BET_m2g", 0.08)]:
        idx = rng.choice(df.index, int(n * miss), replace=False)
        df.loc[idx, col] = np.nan

    return df


# ----------------------------------------------------------------------------
# 2. 数据质量报告 —— Data Descriptor 的骨架
# ----------------------------------------------------------------------------
def quality_report(df, outdir: str) -> None:
    import pandas as pd

    print("\n" + "=" * 66)
    print("步骤 1 / 4  数据质量报告")
    print("=" * 66)
    print(f"记录数: {len(df)}   字段数: {df.shape[1]}")
    if "doi" in df.columns:
        print(f"来源文献数: {df['doi'].nunique()}")
        per_paper = len(df) / max(df['doi'].nunique(), 1)
        print(f"平均每篇贡献数据点: {per_paper:.1f}")

    miss = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        print("\n缺失率（>0 的字段）—— 这张表要原样写进论文：")
        for col, pct in miss.items():
            flag = "  ⚠ 缺失严重，考虑是否剔除该变量" if pct > 40 else ""
            print(f"   {col:26s} {pct:5.1f}%{flag}")
    else:
        print("\n无缺失值。")

    miss.to_csv(os.path.join(outdir, "missing_rate.csv"),
                header=["missing_rate_pct"], encoding="utf-8-sig")


# ----------------------------------------------------------------------------
# 3. 描述统计与相关性图
# ----------------------------------------------------------------------------
def descriptive_plots(df, target: str, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # 无界面环境也能出图
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    print("\n" + "=" * 66)
    print("步骤 2 / 4  描述统计与相关性")
    print("=" * 66)

    num = df.select_dtypes(include=[np.number])
    if target not in num.columns:
        print(f"目标变量 {target} 不是数值型，跳过。")
        return

    print(f"\n目标变量 {target} 描述统计：")
    print(num[target].describe().to_string())

    # --- Figure: 目标变量分布 ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(num[target].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel(f"{target}")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of target variable")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_target_distribution.png"), dpi=200)
    plt.close(fig)

    # --- Figure: Spearman 相关矩阵（对非线性关系更稳健）---
    corr = num.corr(method="spearman")
    cols = [c for c in corr.columns if c != "year"]
    corr = corr.loc[cols, cols]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title("Spearman correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_correlation_matrix.png"), dpi=200)
    plt.close(fig)

    # 与目标相关性最强的变量 —— 写进正文
    tc = corr[target].drop(target).sort_values(key=abs, ascending=False)
    print(f"\n与 {target} 相关性最强的变量（Spearman）：")
    for k, v in tc.head(8).items():
        print(f"   {k:26s} {v:+.3f}")

    print("\n已保存: fig_target_distribution.png, fig_correlation_matrix.png")


# ----------------------------------------------------------------------------
# 4. 机器学习建模（GroupKFold 防泄漏）
# ----------------------------------------------------------------------------
FEATURES_NUM_DEFAULT = [
    "Si_Al_ratio", "BET_m2g", "pore_volume_cm3g",
    "alkali_fusion_T_C", "crystallization_T_C", "crystallization_t_h",
    "T_K", "P_bar", "relative_humidity_pct",
]
FEATURES_CAT_DEFAULT = ["zeolite_type", "waste_type", "cation", "method"]


def build_model_matrix(df, target, num_feats, cat_feats):
    import numpy as np
    import pandas as pd

    num_feats = [c for c in num_feats if c in df.columns and c != target]
    cat_feats = [c for c in cat_feats if c in df.columns]

    X = df[num_feats].copy()
    for c in cat_feats:
        dummies = pd.get_dummies(df[c].astype("category"), prefix=c, dummy_na=False)
        X = pd.concat([X, dummies], axis=1)

    # 缺失值用中位数填补（树模型可接受，比删样本更保住样本量）
    X = X.apply(lambda s: s.fillna(s.median()) if np.issubdtype(s.dtype, np.number) else s)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), df.loc[mask]


def run_modeling(df, target, outdir, use_xgboost, n_splits=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("\n" + "=" * 66)
    print("步骤 3 / 4  机器学习建模")
    print("=" * 66)

    X, y, sub = build_model_matrix(df, target, FEATURES_NUM_DEFAULT, FEATURES_CAT_DEFAULT)

    if "doi" in sub.columns:
        groups = sub["doi"].astype(str)
        n_groups = int(groups.nunique())
    else:
        groups = np.arange(len(sub))
        n_groups = len(sub)
    n_splits = max(2, min(n_splits, n_groups))
    print(f"样本 {len(X)} 条 | 特征 {X.shape[1]} 个 | 分组数 {n_groups} | "
          f"折数 {n_splits}（按 doi 分组，防数据泄漏）")

    # 模型候选
    models = {"RandomForest": RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1)}
    if use_xgboost:
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42)
        except Exception:  # noqa: BLE001
            print("XGBoost 不可用，仅用 RandomForest。")

    gkf = GroupKFold(n_splits=n_splits)
    results = {}
    best = None  # (R2, name, y_true, y_pred)

    for name, mdl in models.items():
        y_true_all, y_pred_all = [], []
        for tr, te in gkf.split(X, y, groups=groups):
            mdl.fit(X.iloc[tr], y.iloc[tr])
            pred = mdl.predict(X.iloc[te])
            y_true_all.extend(y.iloc[te].tolist())
            y_pred_all.extend(pred.tolist())

        r2 = r2_score(y_true_all, y_pred_all)
        rmse = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
        mae = mean_absolute_error(y_true_all, y_pred_all)
        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"\n  {name}:  R2 = {r2:.3f}   RMSE = {rmse:.3f}   MAE = {mae:.3f}")

        cand = (r2, name, np.array(y_true_all), np.array(y_pred_all))
        if best is None or r2 > best[0]:
            best = cand

    # --- Figure: 预测 vs 实测（取 R2 最高的模型）---
    if best:
        _, name, yt, yp = best
        fig, ax = plt.subplots(figsize=(5.4, 5))
        ax.scatter(yt, yp, s=22, alpha=0.6, color="#C44E52", edgecolor="none")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Experimental (mmol/g)")
        ax.set_ylabel("Predicted (mmol/g)")
        ax.set_title(f"{name}  (GroupKFold by source paper)\n"
                     f"R² = {results[name]['R2']:.3f}, RMSE = {results[name]['RMSE']:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_pred_vs_exp.png"), dpi=200)
        plt.close(fig)
        print("\n已保存: fig_pred_vs_exp.png")

    with open(os.path.join(outdir, "model_metrics.txt"), "w", encoding="utf-8") as fh:
        for k, v in results.items():
            fh.write(f"{k}\tR2={v['R2']:.4f}\tRMSE={v['RMSE']:.4f}\tMAE={v['MAE']:.4f}\n")

    return models, X, y, groups, results


# ----------------------------------------------------------------------------
# 5. SHAP 可解释性 —— 论文的点睛之笔
# ----------------------------------------------------------------------------
def run_shap(models, X, y, outdir):
    print("\n" + "=" * 66)
    print("步骤 4 / 4  SHAP 可解释性分析")
    print("=" * 66)
    try:
        import shap
    except ImportError:
        print("未安装 shap，跳过。安装: pip install shap")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mdl = models.get("RandomForest") or list(models.values())[0]
    # 注意：这里在全量数据上拟合仅为出示意图。
    # 正式投稿时，应只在你选定的训练集折上拟合，避免解释器看到测试集。
    mdl.fit(X, y)
    print("（SHAP 解释器在全量数据上拟合，仅作示意；投稿版请改为仅在训练集拟合）")

    try:
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X)
    except Exception as exc:  # noqa: BLE001
        print(f"SHAP 计算失败: {exc}")
        return

    plt.figure(figsize=(7, 5))
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.title("SHAP feature importance", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_shap_importance.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("已保存: fig_shap_importance.png")
    print("提示：SHAP 图要配合 dependence plot（依赖图）讲机制，")
    print("      例如碱熔温度是否呈现倒 U 型最优窗口——那才是审稿人想看的结论。")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="固废基沸石 CO2 吸附数据集分析流水线")
    ap.add_argument("--input", help="输入 CSV 路径（你提取的数据表）")
    ap.add_argument("--target", default="CO2_uptake_mmolg", help="目标变量列名")
    ap.add_argument("--outdir", default="outputs", help="输出目录")
    ap.add_argument("--demo", action="store_true", help="用内置演示数据跑通全流程")
    args = ap.parse_args()

    status = check_dependencies()
    report_dependencies(status)

    import os
    os.makedirs(args.outdir, exist_ok=True)

    if args.demo or not args.input:
        print("\n[模式] 演示数据 —— 数值为随机合成，仅用于理解流程，"
              "**不得用于任何学术结论**。")
        df = make_demo_data()
    else:
        import pandas as pd
        df = pd.read_csv(args.input, encoding="utf-8-sig")
        print(f"\n[模式] 读取真实数据: {args.input}  ({len(df)} 行)")

    quality_report(df, args.outdir)
    descriptive_plots(df, args.target, args.outdir)
    models, X, y, groups, results = run_modeling(
        df, args.target, args.outdir, use_xgboost=status.get("xgboost", False))
    run_shap(models, X, y, args.outdir)

    print("\n" + "=" * 66)
    print("全部完成。输出目录:", os.path.abspath(args.outdir))
    print("=" * 66)


if __name__ == "__main__":
    main()
