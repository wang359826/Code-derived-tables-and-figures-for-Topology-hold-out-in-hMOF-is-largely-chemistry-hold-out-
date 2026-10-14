#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已保存的 fig5_shap_family*.csv 重新绘制 Figure S1。
（2026-10-11 图序修订后，此图由原 Figure 5 变为 SI Figure S1；
CSV 文件名沿用存档名不变。SHAP 计算很慢，改图样式时用这个重画即可，不必重跑。）

运行
----
    python plot_shap_family.py --csv outputs/fig5_shap_family_final.csv \
                               --out outputs/figS1_shap_family_final.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# 图内用英文，避免中文字体乱码
# 注意（2026-10-10，C9 更正）：原来的 "geometric descriptors" 名不副实——
# 这 8 个里只有 5 个是纯几何量（weight 是摩尔质量，surface_area / void_volume
# 是每克归一化量），故统一改称 structure-derived scalar descriptors。
FAMILY_EN = {
    "structure-derived descriptors": "structure-derived\nscalar descriptors",
    "functional_groups": "functional\ngroups",
    "metal_linker": "metal\nlinker",
    "topology": "topology",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/fig5_shap_family_final.csv")
    ap.add_argument("--out", default="outputs/figS1_shap_family_final.png")
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    r = d[d["scheme"] == "random"].set_index("family")
    t = d[d["scheme"] == "topology"].set_index("family")

    order = list(r.index)
    x = np.arange(len(order))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.bar(x - w / 2, r.loc[order, "mean_abs_shap"], w,
            color="#A32D2D", label="random split")
    ax1.bar(x + w / 2, t.loc[order, "mean_abs_shap"], w,
            color="#185FA5", label="leave-one-topology-out")
    ax1.set_xticks(x)
    ax1.set_xticklabels([FAMILY_EN.get(f, f) for f in order], fontsize=9)
    ax1.set_ylabel("mean |SHAP| (per fold)")
    ax1.set_title("Absolute attribution by feature family", fontsize=10.5)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25, lw=0.5)

    ax2.bar(x - w / 2, r.loc[order, "fraction"] * 100, w,
            color="#A32D2D", label="random split")
    ax2.bar(x + w / 2, t.loc[order, "fraction"] * 100, w,
            color="#185FA5", label="leave-one-topology-out")
    for xi, f in zip(x, order):
        dr, dt = r.loc[f, "fraction"] * 100, t.loc[f, "fraction"] * 100
        ax2.text(xi, max(dr, dt) + 1.5, f"{dr:.0f}% → {dt:.0f}%",
                 ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([FAMILY_EN.get(f, f) for f in order], fontsize=9)
    ax2.set_ylabel("share of total attribution [%]")
    ax2.set_ylim(0, 105)
    ax2.set_title("Relative attribution by feature family", fontsize=10.5)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("Where the model looks barely changes — structure-derived descriptors\n"
                 "dominate either way (so the leak is NOT 'the model memorises building-block identity')",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"已保存: {args.out}")

    print("\n数值：")
    for f in order:
        print(f"  {f:24s} random {r.loc[f,'fraction']*100:5.1f}%   "
              f"topology {t.loc[f,'fraction']*100:5.1f}%")


if __name__ == "__main__":
    main()
