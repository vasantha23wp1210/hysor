#!/usr/bin/env python3
"""
plot_hazard_analysis.py
Generates hazard frequency comparison chart.
Output: evaluation/figures/hazard_analysis.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

FIG_DIR = Path("evaluation/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

hazards = [
    "Track\nDeviation",
    "FMS\nReprogramming",
    "Go-Around",
    "Lateral/Vertical\nDeviation",
    "Automation\nSetup Error",
    "Unstable\nApproach",
    "ATC Late\nChange",
    "Terrain/Sep\nHazard",
]

rule_based = [58, 44, 50, 51, 43, 21, 16, 22]
semantic   = [32, 28, 25, 27, 27, 11, 18,  6]

x     = np.arange(len(hazards))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))

bars1 = ax.bar(x - width/2, rule_based, width,
               label="Rule-based (n=363)",
               color="#D55E00", hatch="//", edgecolor="white")
bars2 = ax.bar(x + width/2, semantic,   width,
               label="Semantic (n=144)",
               color="#0072B2", hatch="xx", edgecolor="white")

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5, str(int(h)),
            ha="center", va="bottom", fontsize=9
        )

ax.set_xlabel("Hazard / Outcome Type", fontsize=11)
ax.set_ylabel("Incident Count",        fontsize=11)
ax.set_title(
    "Hazard and Outcome Frequency Analysis\n"
    "Representative Approach Phase Operational Scenario\n"
    "(Rule-based vs Semantic Extraction)",
    fontsize=12, pad=12
)
ax.set_xticks(x)
ax.set_xticklabels(hazards, fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10)

plt.tight_layout()

out_png = FIG_DIR / "hazard_analysis.png"
out_pdf = FIG_DIR / "hazard_analysis.pdf"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")