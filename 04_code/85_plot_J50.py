#!/usr/bin/env python3
"""F7: J50 (human-judged pairs to reach 50% ACT) per method — document-level block, eps=0.02."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(HUB, "05_results", "unified", "J50.csv"))
d = d[(d.block.str.startswith("B")) & (d.eps == 0.02)]
order = ["uniform", "weighted", "strat_pilot", "ai_calib", "ai_resid", "ai_robust_0.5", "weighted_cv"]
label = {"uniform": "uniform", "weighted": "decision-weight IS", "strat_pilot": "stratified (pilot σ)", "ai_calib": "active (calibrated judge)",
         "ai_resid": "active (residual model)", "ai_robust_0.5": "active, robust mix", "weighted_cv": "decision-weight IS + judge CV"}
colls = ["dbpedia-entity", "dl212223"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, c in zip(axes, colls):
    g = d[d.collection == c]
    for i, m in enumerate(order):
        s = g[g.method == m]
        if s.empty:
            continue
        for k, (_, r) in enumerate(s.sort_values("judge").iterrows()):
            val = r.J50_docs; x = i + (k - (len(s) - 1) / 2) * 0.28
            if val != val:
                ax.bar(x, r.max_budget_docs, width=0.26, color="none", edgecolor="gray", hatch="//"); ax.text(x, r.max_budget_docs * 1.01, "not\nreached", ha="center", fontsize=6, color="gray")
            else:
                ax.bar(x, val, width=0.26, color=("C3" if r.judge == "Qwen3-8B" else "C1" if r.judge == "Qwen3-Reranker" else "C0"))
                ax.text(x, val * 1.01, f"{val:,.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([label[m] for m in order], rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{c}  (ε=0.02, α=0.1, precision@cutoff)", fontsize=9); ax.set_ylabel("J50: human-judged pairs to 50% ACT"); ax.grid(axis="y", alpha=.25)
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color="C0", label="humans only"), Patch(color="C3", label="with Qwen3-8B judge"), Patch(color="C1", label="with Qwen3-Reranker judge")], fontsize=7.5)
fig.suptitle("F7 — Cost to certify at equal error control: decision-weighted sampling carries the gain; judge adds a judge-dependent increment", fontsize=10)
fig.tight_layout(); out = os.path.join(HUB, "05_results", "unified", "F7_J50.png"); fig.savefig(out, dpi=150); print("saved", out)
