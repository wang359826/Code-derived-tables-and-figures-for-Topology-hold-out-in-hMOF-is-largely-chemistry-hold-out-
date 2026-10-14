#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 omitted_variable.py 产出的 within_vs_cross.csv 画成 Figure S3
（2026-10-11 图序修订后，此图由原 Figure 8 变为 SI Figure S3）。

运行
----
    python plot_within_cross.py --csv outputs/within_vs_cross_final.csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/within_vs_cross_final.csv")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d = pd.read_csv(args.csv).sort_values("R2_cross")

    x = np.arange(len(d))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, d["R2_within"], w, color="#185FA5",
                label="trained within the same topology")
    b2 = ax.bar(x + w / 2, d["R2_cross"], w, color="#A32D2D",
                label="trained on OTHER topologies")
    ax.axhline(0, color="#333333", lw=0.9)

    for b, v in zip(b1, d["R2_within"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                ha="center", fontsize=8.5)
    for b, v in zip(b2, d["R2_cross"]):
        off = 0.03 if v >= 0 else -0.13
        ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:.2f}",
                ha="center", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(d["topology"], d["n"])],
                       fontsize=9)
    ax.set_ylabel("R² (identical 8 structure-derived scalar descriptors)")
    ax.set_title("The geometry → uptake map is topology-specific\n"
                 f"median within = {d['R2_within'].median():.3f}   "
                 f"median cross = {d['R2_cross'].median():.3f}",
                 fontsize=11.5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    lo = float(min(d["R2_cross"].min(), 0) - 0.6)
    hi = float(max(d["R2_within"].max(), 0) + 0.18)
    ax.set_ylim(lo, hi)

    fig.tight_layout()
    out = os.path.join(args.outdir, "figS3_within_vs_cross.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
