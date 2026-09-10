#!/usr/bin/env python3
"""F6: when does PPI pay? ACT ratio (PPI / humans-only) over rho x N_unlab, one panel per eps, T=90."""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root = os.path.join(HUB, "05_results", "rho_map")
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "rho_map_*.csv"))])
df = df[df["T"] == 90]
colls = sorted(df.collection.unique()); epss = sorted(df.eps.unique())
fig, axes = plt.subplots(len(colls), len(epss), figsize=(3.3 * len(epss), 3.0 * len(colls)), squeeze=False)
for i, c in enumerate(colls):
    for j, e in enumerate(epss):
        ax = axes[i][j]; d = df[(df.collection == c) & (df.eps == e)]
        piv = d.pivot_table(index="N_unlab", columns="rho", values="ratio")
        im = ax.imshow(piv.values, vmin=0.8, vmax=1.8, cmap="RdYlGn", aspect="auto", origin="lower")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{r:.2f}" for r in piv.columns], fontsize=7)
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([str(n) for n in piv.index], fontsize=7)
        for yi in range(piv.shape[0]):
            for xi in range(piv.shape[1]):
                v = piv.values[yi, xi]
                if v == v:
                    ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=7, color="k")
        ax.set_title(f"{c}  ε={e}", fontsize=9); ax.set_xlabel("ρ (judge noise ↓)", fontsize=8); ax.set_ylabel("unlabeled N", fontsize=8)
fig.suptitle("F6 — ACT(PPI) / ACT(humans only) at T=90, α=0.1; wrong-cert ≤ 0.01 in every cell", fontsize=10)
fig.tight_layout(); fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="ACT ratio")
out = os.path.join(root, "F6_rho_map.png"); fig.savefig(out, dpi=150); print("saved", out)
