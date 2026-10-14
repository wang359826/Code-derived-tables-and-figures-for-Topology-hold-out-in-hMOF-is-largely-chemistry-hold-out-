#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
筛选效用（screening utility）分析：把"R² 掉了多少"翻译成"筛出来的材料差多少"
================================================================================

为什么写这个脚本
----------------
R² 是从业者看不懂、也容易被方差骗的指标。前面几轮我们已经被 R² 坑过两次：

  1. 两个测试集的目标方差差 7.7 倍，导致 R² = -0.448 的一侧 MAE 反而更小；
  2. 逐组 R² 的 pooled / weighted / macro 三种口径符号相反。

而真实世界里，做 MOF 高通量筛选的人根本不看 R²，他们看的是：
**按模型排序取前 1% / 5%，丢进下一轮模拟，里面到底有几个是真好的。**

本脚本把口径统一到"同支持集、同目标分布"上，彻底绕开上面两个坑：

  * 用交叉验证得到**每个样本的 out-of-fold 预测**（`cv_predict`），
    所以四种划分策略的预测覆盖的是**完全相同的样本、完全相同的 y**；
  * 于是 precision@k /Spearman/ 选中材料的真实均值等所有指标都可直接相减比较，
    不存在"换支持集就是换分母"的问题。

指标定义
--------
  k = 前 1% / 5% / 10%（按预测值从大到小）
  overlap@k    = |预测 top-k ∩ 真实 top-k| / k
                 随机猜的期望是 k/N，即 0.01 / 0.05 / 0.10
  enrichment@k = overlap@k / (k/N)          —— 相对随机猜的倍数
  topk_true_mean = 模型挑出的 top-k 的真实吸附量均值
  oracle_topk_mean = 真实 top-k 的均值（上界）
  regret@k     = oracle_topk_mean - topk_true_mean   —— 直接回答"损失了多少"

四种划分策略（都对同一样本全集做 out-of-fold 预测）
--------------------------------------------------
  random       KFold(5)                    —— 文献默认做法，泄漏最大
  linker_pair  GroupKFold(5) 按连接体对     —— 留出未见化学
  metal        GroupKFold(5) 按金属中心
  topology     GroupKFold(5) 按拓扑         —— 领域"金标准"

运行
----
    python screening_utility.py --max-rows 25000
    python screening_utility.py --csv data/top_MOFs_screening_data.csv --tag elite

输出
----
    outputs/screening_utility[_tag].csv           逐策略逐 k 的指标
    outputs/fig9_screening_utility[_tag].png      overlap@k 柱状图 + regret 曲线
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

import mof_leakage_benchmark as mb  # 复用基准脚本的数据加载与特征构造

TARGET = mb.TARGET
KS = [0.01, 0.05, 0.10]


def cv_predict(X, y, groups, words=None, n_splits=5, label=""):
    """返回每个样本的 out-of-fold 预测（顺序与输入一致）。

    关键点：每个样本恰好落在一个测试折里一次，
    因此四种策略的预测支持集完全相同，后续指标可直接比较。
    """
    from sklearn.model_selection import GroupKFold, KFold

    if groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True,
                         random_state=mb.RANDOM_STATE)
        garg = None
    else:
        ns = max(2, min(n_splits, groups.nunique()))
        splitter = GroupKFold(n_splits=ns)
        garg = groups.values

    pred = np.full(len(y), np.nan)
    for tr, te in splitter.split(X, y, groups=garg):
        mdl = mb._make_model()
        mdl.fit(X.iloc[tr], y.iloc[tr])
        pred[te] = mdl.predict(X.iloc[te])
    assert not np.isnan(pred).any(), "有样本没有被预测到"
    if label:
        print(f"  [{label}] out-of-fold 预测完成，n={len(pred)}")
    return pred


def add_identity_features(df, cats=None):
    """one-hot 编码构筑单元身份（文献基线做法）。"""
    cats = cats or [c for c in mb.CATEGORICAL_FEATURES if c in df.columns]
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--strategies", default="",
                    help="逗号分隔的策略名，留空则全部（random/topology/metal/linker_pair/chemistry）")
    ap.add_argument("--seed", type=int, default=mb.RANDOM_STATE,
                    help="抽样与模型的随机种子。2026-10-14 新增：正文 §3.11 / Table 16 的 "
                         "screening utility 此前只报单次运行（random_state=42），无法给区间；"
                         "用不同 --seed 跑多次即可给出\"抽样+模型\"的合成不确定度")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="RandomForest 并行度；受限沙箱下 joblib 无法创建命名管道，需传 1")
    ap.add_argument("--no-figure", action="store_true",
                    help="不重画 fig9（多种子重跑时避免产生一堆同名 PNG）")
    args = ap.parse_args()

    # 多种子：抽样层（load_data）与模型层（_make_model）都读 mb.RANDOM_STATE
    mb.RANDOM_STATE = args.seed
    if hasattr(mb, "N_JOBS_OVERRIDE"):
        mb.N_JOBS_OVERRIDE = args.n_jobs

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    csv = args.csv if os.path.isabs(args.csv) else os.path.join(root, args.csv)
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(root, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    df = mb.load_data(csv, max_rows=args.max_rows if args.max_rows > 0 else None)
    X, y, groups, cats, dclean = mb.build_features(df, target=args.target)

    print(f"\n样本: n={len(y)}，目标={args.target}")
    print(f"目标变量: mean={y.mean():.4f} sd={y.std():.4f} var={y.var():.4f} "
          f"min={y.min():.4f} max={y.max():.4f}")
    print("注意：本分析在同支持集上比较，**不需要**也不应该跨种群比较 R²。\n")

    strategies = ordered_strategies(groups)
    if args.strategies:
        keep = {s.strip() for s in args.strategies.split(",") if s.strip()}
        strategies = [(n, g) for n, g in strategies if n in keep]
        if not strategies:
            ap.error(f"--strategies 指定的策略不在可用列表中: {keep}")
    rows = []
    preds = {}
    for name, g in strategies:
        p = cv_predict(X, y, g, label=name)
        preds[name] = p
        print_strategy_metrics(y, p, name, rows)

    out = pd.DataFrame(rows)
    suffix = f"_{args.tag}" if args.tag else ""
    p_csv = os.path.join(outdir, f"screening_utility{suffix}.csv")
    out.to_csv(p_csv, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {p_csv}")

    if not args.no_figure:
        make_figure(y, preds, os.path.join(outdir, f"fig9_screening_utility{suffix}.png"))
    else:
        print("（--no-figure：跳过 fig9 重画）")

    print("\n" + "=" * 74)
    print("读法")
    print("=" * 74)
    print("所有指标的分子分母都建立在**同一批样本**上，所以跨策略可直接相减。")
    print("regret@1% 是给从业者的底线答案：按模型排序挑前 1%，")
    print("拿到的材料平均比真正最好的前 1% 差多少 mmol/g。")


def ordered_strategies(groups):
    out = []
    if "topology" in groups and groups["topology"] is not None:
        out.append(("topology", groups["topology"]))
    if "linker_pair" in groups and groups["linker_pair"] is not None:
        out.append(("linker_pair", groups["linker_pair"]))
    if "metal" in groups and groups["metal"] is not None:
        out.append(("metal", groups["metal"]))
    if "chemistry" in groups and groups["chemistry"] is not None:
        out.append(("chemistry", groups["chemistry"]))
    out.insert(0, ("random", None))  # 随机划分放最前，作为泄漏上限参照
    return out


def print_strategy_metrics(y_true, y_pred, name, rows):
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = len(yt)
    order = np.argsort(-yp)          # 按预测值降序
    true_order = np.argsort(-yt)

    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    rho = spearmanr(yt, yp).correlation

    print(f"\n[{name}]  R2={r2:.4f}  MAE={mae:.4f}  Spearman={rho:.4f}")
    for frac in KS:
        k = max(1, int(round(frac * n)))
        sel = set(order[:k].tolist())
        truth = set(true_order[:k].tolist())
        inter = len(sel & truth)
        overlap = inter / k
        enrich = overlap / frac                       # 相对随机猜
        topk_true = float(yt[order[:k]].mean())
        oracle = float(yt[true_order[:k]].mean())
        regret = oracle - topk_true
        print(f"    k={frac*100:>4.1f}% (n={k:>5d})  overlap@k={overlap:.4f}  "
              f"enrichment={enrich:6.2f}x  topk_true={topk_true:.4f}  "
              f"oracle={oracle:.4f}  regret={regret:.4f}")
        rows.append({
            "strategy": name, "k_frac": frac, "k": k,
            "overlap_at_k": round(overlap, 4),
            "random_expectation": frac,
            "enrichment": round(enrich, 3),
            "topk_true_mean": round(topk_true, 4),
            "oracle_topk_mean": round(oracle, 4),
            "regret": round(regret, 4),
            "R2": round(r2, 4), "MAE": round(mae, 4),
            "spearman": round(float(rho), 4),
        })


def make_figure(y_true, preds, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yt = np.asarray(y_true, dtype=float)
    order_note = ["random", "linker_pair", "metal", "chemistry", "topology"]
    names = [n for n in order_note if n in preds] + \
            [n for n in preds if n not in order_note]
    colors = {"random": "#4C72B0", "linker_pair": "#DD8452",
              "metal": "#937860", "chemistry": "#55A868", "topology": "#C44E52"}
    light = ["#4C72B0", "#DD8452", "#937860", "#55A868", "#C44E52", "#8C8C8C"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # 左图：overlap@k 的富集倍数
    ax = axes[0]
    n = len(yt)
    width = 0.2
    for i, frac in enumerate(KS):
        vals = []
        for nm in names:
            yp = preds[nm]
            k = max(1, int(round(frac * n)))
            sel = set(np.argsort(-yp)[:k].tolist())
            truth = set(np.argsort(-yt)[:k].tolist())
            vals.append(len(sel & truth) / k / frac)
        xs = np.arange(len(names)) + (i - 1) * width
        ax.bar(xs, vals, width, label=f"k={frac*100:.0f}%",
               color=["#4C72B0", "#DD8452", "#937860"][i], alpha=0.9)
    ax.axhline(1.0, ls="--", color="#888888", lw=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Enrichment over random  (x)")
    ax.set_title("Screening enrichment by splitting strategy\n"
                 "(identical support & identical y)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _light_axes(ax)

    # 右图：regret vs k 曲线 + oracle 上界
    ax = axes[1]
    fracs = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    for j, nm in enumerate(names):
        yp = preds[nm]
        order = np.argsort(-yp)
        tru = np.argsort(-yt)
        regrets, oracle = [], []
        for f in fracs:
            k = max(1, int(round(f * n)))
            regrets.append(float(yt[tru[:k]].mean() - yt[order[:k]].mean()))
            oracle.append(float(yt[tru[:k]].mean()))
        ax.plot([f * 100 for f in fracs], regrets, marker="o",
                color=colors.get(nm, light[j % len(light)]), label=nm, lw=1.8)
    ax.set_xlabel("Top fraction selected (%)")
    ax.set_ylabel("Regret vs oracle  (mmol/g)")
    ax.set_title("How much performance is lost by trusting the ranking",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _light_axes(ax)

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    print(f"已保存: {outpath}")


def _light_axes(ax):
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#222222")
    for s in ("bottom", "top", "left", "right"):
        ax.spines[s].set_color("#888888")
    ax.xaxis.label.set_color("#222222")
    ax.yaxis.label.set_color("#222222")
    ax.title.set_color("#222222")


if __name__ == "__main__":
    main()
