import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------------------------
# Helpers
# ----------------------------
def draw_token(ax, xy, label, color="#87cefa", height=0.8, width=2.0, fontsize=10):
    rect = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.2",
        linewidth=1.2,
        edgecolor="black",
        facecolor=color
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width/2,
        xy[1] + height/2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight="bold"
    )

def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end, xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.7)
    )


# ----------------------------
# Figure layout
# ----------------------------
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.axis("off")

# Fake DNA tokens
seq = "ATGCTTACGATCGGATCCTGAACT"
tokens = ["ATGCTT\n(1024 FP32)", "ACGATC\n1024 FP32", "GGATCC\n1024 FP32", "TGAACT\n1024 FP32"]

# Top row positions
y_top = 6
x0 = 1
w = 2.3
h = 0.9
gap = 0.4

# ----------------------------
# Top row: CLS + 4 sequence tokens
# ----------------------------

draw_token(ax, (x0, y_top), "CLS\n1024 FP32", color="#f4b183")

top_centers = []
for i, t in enumerate(tokens):
    x = x0 + (i+1)*(w + 0.4)
    draw_token(ax, (x, y_top), t, color="#a6cee3")
    top_centers.append((x + w/2, y_top + h/2))

ax.text(1.7, 8, "Processing embeddings for classification", fontsize=32)
ax.text(4.5, 1, "This is just 1 of 30 model layers!", fontsize=20)

# ----------------------------
# Bottom row objects
# ----------------------------
y_bot = 2.5

# Extracted CLS
draw_token(ax, (1, y_bot), "CLS\n1024 FP32", color="#f4b183")

# Mean pooled SEQ token (taller)
draw_token(ax, (8, y_bot), "SEQ MEAN\n1024 FP32", color="#6aaed6", width=2.0, fontsize=9)

# Concatenated token
draw_token(ax, (13, y_bot - 0.3), "CLS\n+\nSEQ MEAN\n\n2048 FP32", color="#f7ce88",
           height=2.0, width=2.2, fontsize=9)

# Symbols between tokens
ax.text(5.0, y_bot + 0.4, "+", fontsize=26, ha="center", va="center")
ax.text(11.5, y_bot + 0.4, "=", fontsize=26, ha="center", va="center")

# ----------------------------
# Arrows
# ----------------------------
# CLS → extracted CLS
arrow(ax, (x0 + w/2, y_top - 0.5), (x0 + w/2, y_bot + 1.3))

# Sequence tokens → mean pooled SEQ
mean_x = 8 + 1.0   # center of mean seq token
mean_y = y_bot + 0.45
for cx, cy in top_centers:
    arrow(ax, (cx, y_top - 0.5), (mean_x, y_bot + 1.3))

# ----------------------------
plt.show()
