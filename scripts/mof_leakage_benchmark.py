#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOF CO2 捕集性能预测：数据泄漏的量化评估（论文级脚本 v2）
==========================================================

科学问题
--------
MOF 的机器学习论文里，绝大多数用**随机划分**训练/测试集。
但 MOF 数据库（如 hMOF）里的结构是用 102 个构筑单元（building block）
按拓扑模板**组合枚举**出来的——同一拓扑、同一金属/有机连接体的 MOF
会同时掉进训练集和测试集。模型于是"背下了这个构筑单元的答案"，
而不是学到了结构-性能关系。这就是**数据泄漏**，会让 R² 虚高。

本脚本量化这件事：同一个模型、同一份数据，只改划分方式，看 R² 掉多少。

v2 相对 v1 的三项升级
---------------------
1. **Leave-One-Group-Out（LOTO）**：实测发现 hMOF 的拓扑总共只有 11 种，
   GroupKFold(5) 每次只留出约 2 种拓扑，方差大且不好解释。
   LOTO（每次留一种拓扑）是 MOF ML 论文的标准做法。
2. **Table 1 分组统计**：输出每种分组的组数、平均/最小/最大组大小。
   审稿人一定会问"你的分组划分到底排除了多少信息"，这张表就是回答。
3. **缓解手段对照（--mitigation）**：测试常见缓解手段是否真的缩小了
   随机划分与严格划分之间的差距。

为什么这个课题适合独立完成
--------------------------
* 数据公开（Materials Cloud，无需导师授权）
* 结果可自证：完整划分下的性能才是对外推能力的诚实估计
* 有真实科学价值：这是 MOF ML 社区的公认方法学问题

数据
----
hMOF（Boyd & Woo, Nature 2019），Materials Cloud:
https://archive.materialscloud.org/records/nsp7v-2dk91/files/screening_data.tar.gz

运行
----
    # 基础：四种划分策略对比
    python mof_leakage_benchmark.py --max-rows 25000

    # 加 LOTO + Table 1（论文主结果）
    python mof_leakage_benchmark.py --max-rows 100000 --loto

    # 第二个 Figure：CO2/N2 选择性（比值型目标必须取 log）
    python mof_leakage_benchmark.py --target "CO2/N2_selectivity" --log-target \
                                    --max-rows 100000 --loto --tag logsel

    # 缓解手段对照（建议 <= 25000 行，否则很慢）
    python mof_leakage_benchmark.py --max-rows 25000 --mitigation

输出
----
  leakage_comparison[_tag].png    四种划分策略的 R² 对比（Figure 1 / 2）
  leakage_results[_tag].csv       数值结果
  loto_results[_tag].csv          留一组的汇总结果（Table 2）
  loto_per_topology[_tag].csv/.png 逐拓扑结果（Figure 3）
  group_stats[_tag].csv           Table 1：分组统计
  mitigation_results[_tag].csv    缓解手段对照（Table 3）
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


# 目标变量：烟气条件下的 CO2 吸附量（0.15 bar, 298 K）
TARGET = "CO2_uptake_P0.15bar_T298K [mmol/g]"
TARGET_SELECTIVITY = "CO2/N2_selectivity"

# 几何/物理描述符（GCMC 计算前就能得到的，保证是"预测"而非"事后诸葛"）
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

# 构筑单元（用于分组划分，验证模型是否真能外推到未见过的化学）
CATEGORICAL_FEATURES = ["functional_groups", "metal_linker", "topology"]

N_ESTIMATORS = 120
RANDOM_STATE = 42
LOTO_MAX_GROUPS = 30      # 超过这个组数就不做留一组（太慢），只做 GroupKFold

# ---------------------------------------------------------------------------
# 匹配臂（2026-10-14 新增，回应"chemistry 臂保留了 topology 块、信息上占优"这一质疑）
# ---------------------------------------------------------------------------
# 按化学组成分组时，模型仍持有 11 列 topology one-hot；而按拓扑分组时那一整块按构造
# 不可用（§4.2 / Table 26 的 availability = 0.000）。于是 Table 7 里
# chemistry 0.6738 对 topology 0.3786 这个差，同时包含"分组变量不同"与"存活 identity
# 区块不同"两件事。--drop-topology-block 把 topology 从 one-hot 块里去掉（分组标签
# 仍保留在 groups 里，划分不受影响），使两条臂在特征上可比。
# 用 `--max-rows N --drop-topology-block --tag notopo<...>` 运行即可。
DROP_CAT_BLOCKS: set[str] = set()
N_JOBS_OVERRIDE: int = -1   # --n-jobs 设置；受限沙箱下必须为 1（joblib 造不出命名管道）

# 模型家族：正文主结果用 RF。MODEL_KIND 由 --model 设置，默认是 "rf"，
# 因此不带该参数运行时与历史产出逐位一致。
MODEL_KIND = "rf"


def _make_model():
    """按 MODEL_KIND 返回估计器。

    rf    —— RandomForestRegressor（正文主结果，历史口径）
    gb    —— HistGradientBoostingRegressor（梯度提升对照臂）
    ridge —— StandardScaler + RidgeCV（线性对照臂）

    两个对照臂是为 §4.5「one model, one feature set」那条限制加的：若
    random 与 grouped 两种划分之间的差距在三种模型上同号、同量级，
    则该差距不是随机森林的专属产物。
    """
    if MODEL_KIND == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS_OVERRIDE)
    if MODEL_KIND == "gb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    if MODEL_KIND == "ridge":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import RidgeCV
        return make_pipeline(
            StandardScaler(with_mean=False),
            RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)))
    raise ValueError(f"未知模型: {MODEL_KIND}")


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_data(csv_path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"原始数据: {len(df)} 行 x {df.shape[1]} 列")
    # 必须随机采样，不能取前 N 行：这个文件是按拓扑/连接体排序的，
    # 直接 head() 会只拿到少数几种拓扑，使分组划分的结果失去代表性。
    if max_rows and max_rows < len(df):
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"随机采样 {max_rows} 行用于建模")
    return df


# --------------------------------------------------------------------------
# 特征构造（四种编码方式，用于缓解手段对照）
# --------------------------------------------------------------------------
def build_features(df, target=TARGET, encoding="onehot"):
    """构造特征矩阵与分组标签。

    encoding:
      onehot  几何 + 类别 one-hot（基线，多数文献的做法）
      geo     只用几何描述符（缓解手段 A：彻底去掉构筑单元身份）
      hash    几何 + 哈希编码（缓解手段 B：低维随机投影）
      target  几何 + 目标编码（缓解手段 C：数值代替高维 one-hot，
              必须用折内 out-of-fold 计算，否则本身又是一个泄漏源）

    返回 X, y, groups, cats, df_clean
    """
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    y = pd.to_numeric(df[target], errors="coerce")

    X = df[cols].copy().astype(float)
    cats = [c for c in CATEGORICAL_FEATURES
            if c in df.columns and c not in DROP_CAT_BLOCKS]

    if encoding == "geo":
        pass
    elif encoding == "hash":
        joined = df[cats[0]].astype(str)
        for c in cats[1:]:
            joined = joined + "|" + df[c].astype(str)
        h = pd.util.hash_pandas_object(joined.astype(str), index=False).astype(np.int64)
        for i in range(32):
            X[f"hash_{i}"] = (((h.values >> i) & 1)).astype(float)
    elif encoding == "target":
        for c in cats:
            X[f"te_{c}"] = np.nan          # 占位，真实值在折内计算
    else:                                   # onehot
        for c in cats:
            dummies = pd.get_dummies(df[c].astype(str), prefix=c).astype(float)
            X = pd.concat([X, dummies], axis=1)

    # 目标编码列此时还是 NaN 占位，不能参与缺失值筛选，否则会把样本全删掉
    placeholder = {f"te_{c}" for c in cats} if encoding == "target" else set()
    check_cols = [c for c in X.columns if c not in placeholder]
    mask = y.notna() & X[check_cols].notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    df_clean = df.loc[mask].reset_index(drop=True)

    groups = {
        "topology": (df_clean["topology"].astype(str)
                     if "topology" in df_clean else None),
        "metal": (df_clean["metal_linker"].astype(str)
                  if "metal_linker" in df_clean else None),
        "linker_pair": (
            (df_clean["organic_linker1"].astype(str) + "_"
             + df_clean["organic_linker2"].astype(str))
            if "organic_linker1" in df_clean else None),
        "chemistry": (
            (df_clean["metal_linker"].astype(str) + "|"
             + df_clean["organic_linker1"].astype(str) + "|"
             + df_clean["organic_linker2"].astype(str))
            if all(c in df_clean for c in ["metal_linker", "organic_linker1", "organic_linker2"])
            else None),
    }
    return X, y, groups, cats, df_clean


def apply_log10(X, y, groups):
    """比值型目标取 log10（丢弃 <=0 的样本）。返回 keep 掩码以便同步 df_clean。"""
    keep = (y > 0).values
    n_bad = int((~keep).sum())
    X = X.loc[keep].reset_index(drop=True)
    y = np.log10(y[keep]).reset_index(drop=True)
    groups = {k: (v[keep].reset_index(drop=True) if v is not None else None)
              for k, v in groups.items()}
    return X, y, groups, n_bad, keep


# --------------------------------------------------------------------------
# 交叉验证评估
# --------------------------------------------------------------------------
def _fold_target_encoding(Xtr, Xte, ytr, raw_tr, raw_te, cats, prior_weight=10.0):
    """折内目标编码：只用训练折的标签算类别均值。"""
    global_mean = float(ytr.mean())
    for c in cats:
        col = f"te_{c}"
        g = ytr.groupby(raw_tr[c].astype(str).values)
        ssum, cnt = g.sum(), g.count()
        smooth = (ssum + prior_weight * global_mean) / (cnt + prior_weight)
        Xtr[col] = raw_tr[c].astype(str).map(smooth).fillna(global_mean).values
        Xte[col] = raw_te[c].astype(str).map(smooth).fillna(global_mean).values
    return Xtr, Xte


def evaluate(X, y, groups_series, n_splits, label,
             target_enc_cols=None, df_raw=None):
    """按给定分组做交叉验证，返回 R2 / RMSE / MAE。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupKFold, KFold

    if groups_series is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        groups_arg = None
    else:
        n_splits = max(2, min(n_splits, groups_series.nunique()))
        splitter = GroupKFold(n_splits=n_splits)
        groups_arg = groups_series.values

    y_true, y_pred = [], []
    for tr, te in splitter.split(X, y, groups=groups_arg):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr = y.iloc[tr]
        if target_enc_cols and df_raw is not None:
            Xtr, Xte = _fold_target_encoding(
                Xtr, Xte, ytr, df_raw.iloc[tr], df_raw.iloc[te], target_enc_cols)
        mdl = _make_model()
        mdl.fit(Xtr, ytr)
        y_pred.extend(mdl.predict(Xte).tolist())
        y_true.extend(y.iloc[te].tolist())

    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"  {label:34s} R2 = {r2:.4f}   RMSE = {rmse:.4f}   MAE = {mae:.4f}")
    return {"strategy": label, "R2": r2, "RMSE": rmse, "MAE": mae}


def evaluate_loto(X, y, groups_series, label):
    """Leave-One-Group-Out：每次留出一种取值，逐组报告 R²。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import LeaveOneGroupOut

    per_group, y_true, y_pred = [], [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=groups_series.values):
        if len(te) == 0:
            continue
        mdl = _make_model()
        mdl.fit(X.iloc[tr], y.iloc[tr])
        pred = mdl.predict(X.iloc[te])
        y_pred.extend(pred.tolist())
        y_true.extend(y.iloc[te].tolist())
        gname = groups_series.iloc[te[0]]
        per_group.append({"group": gname, "n_test": len(te),
                          "R2": r2_score(y.iloc[te].tolist(), pred.tolist())})

    overall = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"  {label:34s} R2 = {overall:.4f}   RMSE = {rmse:.4f}   "
          f"MAE = {mae:.4f}   （{len(per_group)} 组）")
    return {"strategy": label, "R2": overall, "RMSE": rmse, "MAE": mae}, \
        pd.DataFrame(per_group)


# --------------------------------------------------------------------------
# Table 1：分组统计
# --------------------------------------------------------------------------
def group_stats_table(groups):
    rows = []
    for name, s in groups.items():
        if s is None:
            continue
        sizes = s.value_counts()
        rows.append({
            "grouping": name,
            "n_samples": int(len(s)),
            "n_groups": int(s.nunique()),
            "mean_group_size": round(float(sizes.mean()), 1),
            "median_group_size": float(sizes.median()),
            "min_group_size": int(sizes.min()),
            "max_group_size": int(sizes.max()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 出图
# --------------------------------------------------------------------------
def bar_figure(results, title, ylabel, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["strategy"].replace("（", "\n(") for r in results]
    vals = [r["R2"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = ["#A32D2D" if i == 0 else "#185FA5" for i in range(len(vals))]
    bars = ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.008 if v >= 0 else -0.035),
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    lo, hi = min(vals + [0]), max(vals + [0])
    ax.set_ylim(lo - 0.08, hi + 0.06)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"\n已保存: {outpath}")


def loto_figure(per_group, outpath, group_name="topology"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = per_group.sort_values("R2")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(range(len(d)), d["R2"], color="#185FA5", width=0.7)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(d["group"], d["n_test"])],
                       fontsize=7)
    ax.set_ylabel("R² (held-out group)")
    ax.set_title(f"Leave-one-{group_name}-out: per-group R²   "
                 f"mean = {d['R2'].mean():.3f}   sd = {d['R2'].std():.3f}",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"已保存: {outpath}")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    global MODEL_KIND
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--model", default="rf", choices=("rf", "gb", "ridge"),
                    help="模型家族；默认 rf，即正文主结果的历史口径")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--target", default=TARGET,
                    help="目标变量列名，可用 'selectivity' 简写")
    ap.add_argument("--tag", default=None, help="输出文件名后缀")
    ap.add_argument("--log-target", action="store_true",
                    help="对目标变量取 log10（比值型目标建议开启）")
    ap.add_argument("--loto", action="store_true", help="额外做 Leave-One-Group-Out")
    ap.add_argument("--mitigation", action="store_true", help="额外做缓解手段对照")
    ap.add_argument("--drop-topology-block", action="store_true",
                    help="从 one-hot 特征里去掉 topology 块（分组标签仍保留），"
                         "用于与 chemistry 分组做特征上的匹配对照；见 DROP_CAT_BLOCKS 注释")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="RandomForest 并行度；受限沙箱下必须传 1（joblib 无法创建命名管道）")
    args = ap.parse_args()

    global DROP_CAT_BLOCKS, N_JOBS_OVERRIDE
    if args.drop_topology_block:
        DROP_CAT_BLOCKS = {"topology"}
    N_JOBS_OVERRIDE = args.n_jobs

    MODEL_KIND = args.model
    os.makedirs(args.outdir, exist_ok=True)

    raw_target = args.target
    if raw_target.lower() in ("selectivity", "sel", "co2/n2"):
        raw_target = TARGET_SELECTIVITY
    tag = args.tag or ("sel" if raw_target == TARGET_SELECTIVITY else "")
    suffix = f"_{tag}" if tag else ""

    if args.mitigation and (args.max_rows or 0) > 30000:
        print("提示：--mitigation 会跑 4 种特征集 x 2 种划分 x 5 折 = 40 次拟合，")
        print("      样本超过 3 万会非常慢，建议 --max-rows 25000。继续。\n")

    print("=" * 72)
    print("MOF CO2 捕集预测：随机划分 vs 分组划分（数据泄漏量化）")
    print(f"目标变量: {raw_target}")
    print(f"模型家族: {MODEL_KIND}")
    print("=" * 72)

    df = load_data(args.csv, args.max_rows)
    if raw_target not in df.columns:
        print(f"错误：数据中没有列 '{raw_target}'。可用目标变量：")
        for c in df.columns:
            if "uptake" in c or "selectivity" in c or "capacity" in c:
                print("   ", c)
        sys.exit(1)

    X, y, groups, cats, df_clean = build_features(df, raw_target, encoding="onehot")

    if args.log_target:
        X, y, groups, n_bad, keep = apply_log10(X, y, groups)
        df_clean = df_clean.loc[keep].reset_index(drop=True)
        print(f"已对目标变量取 log10（丢弃 {n_bad} 个 <=0 的样本）")

    print(f"\n建模样本: {len(X)} 行, {X.shape[1]} 个特征")
    print(f"目标变量范围: {y.min():.3f} ~ {y.max():.3f}, 均值 {y.mean():.3f}")

    # ---- Table 1 ----
    gstats = group_stats_table(groups)
    print("\n分组统计（Table 1 —— 回答'分组划分到底排除了多少信息'）:")
    print(gstats.to_string(index=False))
    gp = os.path.join(args.outdir, f"group_stats{suffix}.csv")
    gstats.to_csv(gp, index=False, encoding="utf-8-sig")
    print(f"已保存: {gp}")

    # ---- Figure 1 ----
    print("\n交叉验证结果:")
    print("-" * 72)
    results = []
    t0 = time.time()
    results.append(evaluate(X, y, None, 5, "随机划分（多数文献做法）"))
    if groups["topology"] is not None:
        results.append(evaluate(X, y, groups["topology"], 5, "按拓扑分组（GroupKFold）"))
    if groups["metal"] is not None:
        results.append(evaluate(X, y, groups["metal"], 5, "按金属中心分组"))
    if groups["linker_pair"] is not None:
        results.append(evaluate(X, y, groups["linker_pair"], 5, "按有机连接体分组"))
    if groups["chemistry"] is not None:
        results.append(evaluate(X, y, groups["chemistry"], 5, "按化学组成分组（metal+2 linkers）"))
    print("-" * 72)
    print(f"耗时 {time.time() - t0:.0f} s")

    res = pd.DataFrame(results)
    csv_path = os.path.join(args.outdir, f"leakage_results{suffix}.csv")
    res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    r_rand = res.loc[0, "R2"]
    r_group = res["R2"].iloc[1:].min() if len(res) > 1 else None
    print("\n" + "=" * 72)
    print("结论")
    print("=" * 72)
    if r_group is not None:
        drop = r_rand - r_group
        print(f"  随机划分 R2   = {r_rand:.4f}")
        print(f"  最严格分组 R2 = {r_group:.4f}")
        print(f"  虚高幅度 ΔR2  = {drop:.4f}")
        if r_group > 1e-6:
            print(f"  相对幅度      = {drop / r_group * 100:.0f}%")
        else:
            print("  分组划分下 R2 已不优于基线（<=0），此时相对百分比无意义，不报告。")
        print("\n  这个差值就是被随机划分掩盖的虚假性能——")
        print("  它衡量的是模型对'见过的化学'的记忆，而非对新材料的外推能力。")

    short = ("CO2/N2 selectivity" if raw_target == TARGET_SELECTIVITY
             else "CO2 uptake")
    bar_figure(results,
               f"Data leakage in MOF {short} prediction\n"
               "same model, same data, only the split changes",
               "R²  (cross-validated)",
               os.path.join(args.outdir, f"leakage_comparison{suffix}.png"))

    # ---- LOTO ----
    if args.loto:
        print("\n" + "=" * 72)
        print("Leave-One-Group-Out（每次留出一种取值，逐组报告）")
        print("=" * 72)
        loto_rows, loto_detail = [], {}
        for key, gname in [("topology", "拓扑"), ("metal", "金属中心"),
                           ("linker_pair", "有机连接体"), ("chemistry", "化学组成")]:
            g = groups.get(key)
            if g is None:
                continue
            if g.nunique() > LOTO_MAX_GROUPS:
                print(f"  跳过 留一{gname}：{g.nunique()} 组过多（>{LOTO_MAX_GROUPS}），"
                      f"改用上面的 GroupKFold 结果")
                continue
            row, detail = evaluate_loto(X, y, g, f"留一{gname}（{g.nunique()} 组）")
            loto_rows.append(row)
            loto_detail[key] = (detail, gname)

        if loto_rows:
            loto_df = pd.DataFrame(loto_rows)
            lp = os.path.join(args.outdir, f"loto_results{suffix}.csv")
            loto_df.to_csv(lp, index=False, encoding="utf-8-sig")
            print(f"\n已保存: {lp}")
            for key, (detail, gname) in loto_detail.items():
                detail.to_csv(os.path.join(args.outdir,
                                           f"loto_per_{key}{suffix}.csv"),
                              index=False, encoding="utf-8-sig")
                try:
                    loto_figure(detail, os.path.join(
                        args.outdir, f"loto_per_{key}{suffix}.png"), gname)
                except Exception as exc:  # noqa: BLE001
                    print(f"  出图失败: {exc}")

    # ---- 缓解手段对照 ----
    if args.mitigation:
        print("\n" + "=" * 72)
        print("缓解手段对照（每种特征集下，随机 vs 按拓扑分组的差距）")
        print("=" * 72)
        mit_rows = []
        for enc, desc in [("onehot", "几何 + 构筑单元 one-hot（基线）"),
                          ("geo", "仅几何描述符（去掉身份）"),
                          ("hash", "几何 + 哈希编码（32 维）"),
                          ("target", "几何 + 折内目标编码")]:
            Xe, ye, ge, _, dfc = build_features(df, raw_target, encoding=enc)
            if args.log_target:
                Xe, ye, ge, _, keep_e = apply_log10(Xe, ye, ge)
                dfc = dfc.loc[keep_e].reset_index(drop=True)
            print(f"\n  [{desc}]  {Xe.shape[1]} 个特征")
            te_cols = cats if enc == "target" else None
            raw4te = dfc if te_cols else None
            rnd = evaluate(Xe, ye, None, 5, "    随机划分",
                           target_enc_cols=te_cols, df_raw=raw4te)
            lot = evaluate(Xe, ye, ge["topology"], 5, "    按拓扑分组",
                           target_enc_cols=te_cols, df_raw=raw4te)
            mit_rows.append({"feature_set": desc,
                             "R2_random": rnd["R2"],
                             "R2_topology_group": lot["R2"],
                             "gap": rnd["R2"] - lot["R2"]})
        mit = pd.DataFrame(mit_rows)
        print("\n  汇总（gap 越小 = 越不容易被泄漏欺骗）:")
        print(mit.to_string(index=False))
        mp = os.path.join(args.outdir, f"mitigation_results{suffix}.csv")
        mit.to_csv(mp, index=False, encoding="utf-8-sig")
        print(f"\n已保存: {mp}")

    print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
