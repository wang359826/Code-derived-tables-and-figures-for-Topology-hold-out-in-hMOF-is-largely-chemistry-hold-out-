#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多种子重复：给所有关键数字加误差棒
=================================

为什么要做（2026-10-09）
----------------------
本项目的所有主数字此前都是**单次运行**（seed=42）的结果，没有误差棒。
评审有权要求证明这些数字不是抽样噪声。本脚本对同一目标跑 N 个种子，
每个种子独立重做「随机抽样 → 特征 → 四种划分 → LOTO」，
最后报告均值 ± 标准差，以及 ΔR² 的分布。

⚠️ 与"固定种子只换 CV 折"不同，本脚本**连抽样一起换**，
因此给出的是「抽样 + 划分 + 模型」三者合起来的总不确定度，是更严格的口径。

用法
----
  # 吸附量，25,000 样本 × 5 个种子
  python multi_seed_repeat.py --max-rows 25000 --seeds 5 --tag uptake25k

  # log 选择性
  python multi_seed_repeat.py --target "CO2/N2_selectivity" --log-target \
         --max-rows 25000 --seeds 5 --tag logsel25k
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mof_leakage_benchmark import (  # noqa: E402
    TARGET, TARGET_SELECTIVITY, build_features, load_data, apply_log10,
)


def _cv(X, y, groups, n_splits, seed, n_est, mode):
    """mode: 'random' | 'group' | 'loto'。返回 (pooled R², 逐组 R² list)。

    ⚠️ 逐折 R² 的口径纪律（2026-10-10 修，2026-10-14 更正折的身份）：
    触发零方差的折是 **rht**，不是 nbo：在这个 25,000 样本协议下 rht 在 seed 42 /
    345 / 446 各只落进 1 个结构，np.nanvar == 0 而 r2_score 返回 NaN（pandas 的
    groupby.var() 也给 NaN）；nbo 在五个种子的测试集方差为 0.0107–0.0134，从未为 0。
    主稿 §3.5 与 Table 14 表注此前把该折写成 nbo，并附了来自 §3.4 受控对照测试集的
    sd 比（1/4.4，0.106 对 0.472）作为理由——sd = 0.106 与"方差为 0"自相矛盾：
    那个 sd 属于另一个实验的测试集，不是这条 25,000 样本运行。已一并更正。
    **不能让它静默把 macro 变成 NaN**，否则 "5 个种子只剩 2 个有效"这种事不会被发现。
    故显式记 NaN 并在汇总时报告有效折数，用 nanmean / nanmedian。
    剔除阈值是 np.nanvar(yte) > 1e-12（只排除退化折），不是"方差偏小就丢"。
    """
    if mode == "random":
        sp = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(sp.split(X))
    elif mode == "group":
        k = max(2, min(n_splits, groups.nunique()))
        sp = GroupKFold(n_splits=k)
        splits = list(sp.split(X, y, groups=groups.values))
    else:
        splits = list(LeaveOneGroupOut().split(X, y, groups=groups.values))

    yt, yp, per = [], [], []
    for tr, te in splits:
        if len(te) == 0:
            continue
        m = RandomForestRegressor(n_estimators=n_est, random_state=seed, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        yt.extend(y.iloc[te].tolist())
        yp.extend(p.tolist())
        yte = np.asarray(y.iloc[te], dtype=float)
        # 测试集目标方差为 0 → R² 无定义，显式记 NaN（不要让它污染 macro）
        per.append(r2_score(yte, p) if np.nanvar(yte) > 1e-12 else np.nan)
    return r2_score(yt, yp), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/all_MOFs_screening_data.csv")
    ap.add_argument("--max-rows", type=int, default=25000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-group", action="store_true",
                    help="跳过 linker/metal/topology 三种 GroupKFold，"
                         "只算 random + LOTO（修 macro 口径后重算用，快约 3 倍）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw = args.target
    if raw.lower() in ("selectivity", "sel", "co2/n2"):
        raw = TARGET_SELECTIVITY

    print("=" * 78)
    print(f"多种子重复：{raw}" + ("（log10）" if args.log_target else ""))
    print(f"样本 {args.max_rows} × 种子 {args.seeds} × 4 种划分")
    print("=" * 78)

    rows = []
    t0 = time.time()
    for i in range(args.seeds):
        seed = 42 + i * 101
        # 换抽样种子：直接改 load_data 的随机状态
        import mof_leakage_benchmark as mb
        mb.RANDOM_STATE = seed
        df = load_data(args.csv, args.max_rows)
        X, y, groups, cats, dfc = build_features(df, raw, encoding="onehot")
        if args.log_target:
            X, y, groups, nb, keep = apply_log10(X, y, groups)

        topo = groups["topology"]
        r_rand, _ = _cv(X, y, topo, 5, seed, args.n_estimators, "random")
        if args.skip_group:
            r_lk = r_metal = r_topo = float("nan")
        else:
            r_lk, _ = _cv(X, y, groups["linker_pair"], 5, seed, args.n_estimators, "group")
            r_metal, _ = _cv(X, y, groups["metal"], 5, seed, args.n_estimators, "group")
            r_topo, _ = _cv(X, y, topo, 5, seed, args.n_estimators, "group")
        r_loto, per = _cv(X, y, topo, 11, seed, args.n_estimators, "loto")

        rows.append({
            "seed": seed, "n": len(X),
            "random": r_rand, "linker": r_lk, "metal": r_metal,
            "topology": r_topo,
            "loto_pooled": r_loto,
            "loto_macro": float(np.nanmean(per)),
            "loto_median": float(np.nanmedian(per)),
            "loto_n_valid_folds": int(np.sum(~np.isnan(per))),
            "delta_pooled": r_rand - r_loto,
            "delta_macro": r_rand - float(np.nanmean(per)),
        })
        print(f"  seed={seed:5d}  随机 {r_rand:.4f}  拓扑 {r_topo:.4f}  "
              f"LOTO pooled {r_loto:+.4f} macro {np.nanmean(per):+.4f} "
              f"(有效折 {int(np.sum(~np.isnan(per)))}/11)  "
              f"Δ(pooled) {r_rand - r_loto:.4f}   （{time.time() - t0:.0f} s）")

    d = pd.DataFrame(rows)
    out = os.path.join(args.outdir, f"multi_seed{('_' + args.tag) if args.tag else ''}.csv")
    d.to_csv(out, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("汇总（mean ± sd，n = %d 个种子）" % len(d))
    print("=" * 78)
    for c in ["random", "linker", "metal", "topology", "loto_pooled",
              "loto_macro", "loto_median", "delta_pooled", "delta_macro"]:
        print(f"  {c:14s} {d[c].mean():+.4f} ± {d[c].std():.4f}   "
              f"[{d[c].min():+.4f}, {d[c].max():+.4f}]")
    if "loto_n_valid_folds" in d.columns:
        print(f"  {'有效折数':12s} {d['loto_n_valid_folds'].min()} ~ "
              f"{d['loto_n_valid_folds'].max()}  / 11"
              f"   ← 测试集方差为 0 的折 R² 无定义，已剔除")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
