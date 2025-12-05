import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------
# Conceptual groups for NT 2.5B (30 layers)
# ----------------------------------------------------
groups = [
    "0–3\nLow-level\n(k-mers)",
    "4–6\nEarly\ncontext",
    "7–11\nPeak\nsemantic",
    "12–17\nDecline\n(begin)",
    "18–21\nStrong\ndecline",
    "22–30\nLate\nlayers"
]

colors = [
    "#cdd7e0",  # light
    "#9ec5ff",
    "#4d9de0",
    "#88a3c0",
    "#b5b5b5",
    "#dddddd",
]

# Qualitative performance curve (relative values, not your real metrics)
perf = np.array([0.82, 0.93, 0.99, 0.985, 0.96, 0.94])
x = np.arange(len(groups))

# ----------------------------------------------------
# PLOT
# ----------------------------------------------------
plt.figure(figsize=(16, 8))
ax = plt.gca()

# Background colored blocks
for i, color in enumerate(colors):
    ax.add_patch(
        plt.Rectangle((i - 0.45, 0.78), 0.9, 0.25, color=color, alpha=0.35)
    )

# Performance line
ax.plot(x, perf, color="black", linewidth=3, marker="o")

# Labels above blocks
for i, label in enumerate(groups):
    ax.text(i, 1.0, label, ha="center", va="bottom", fontsize=12, weight="bold")

# Axis formatting
ax.set_ylim(0.78, 1.05)
ax.set_xticks(x)
ax.set_xticklabels([""] * len(groups))  # hide numeric ticks
ax.set_ylabel("Relative Performance (qualitative)", fontsize=13)
ax.set_title("Performance Trajectory Across Nucleotide Transformer 500M V2 Layers",
             fontsize=20, pad=50)

ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

