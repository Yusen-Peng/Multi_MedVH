import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load results
# ============================================================

df = pd.read_csv("results/main_results.csv")

modalities = [
    "CT",
    "MRI",
    "CXR",
    "ECG",
    "Pathology",
]

# Keep models in the same order as the CSV
heatmap_df = df.set_index("model")[modalities]


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(
    heatmap_df.values,
    aspect="auto",
    vmin=0,
    vmax=80,
    cmap="YlGnBu",
)


# ============================================================
# Axis labels
# ============================================================

ax.set_xticks(np.arange(len(modalities)))
ax.set_xticklabels(
    modalities,
    fontsize=11,
)

ax.set_yticks(np.arange(len(heatmap_df)))
ax.set_yticklabels(
    heatmap_df.index,
    fontsize=10,
)


# ============================================================
# Numerical annotations
# ============================================================

for i in range(len(heatmap_df)):
    for j in range(len(modalities)):

        value = heatmap_df.iloc[i, j]

        # Switch text color for readability
        text_color = "white" if value > 55 else "black"

        ax.text(
            j,
            i,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            color=text_color,
        )


# ============================================================
# Colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.03,
)

cbar.set_label(
    "Accuracy (%)",
    fontsize=11,
)


# ============================================================
# Formatting
# ============================================================

ax.set_title(
    "Performance Across Medical Modalities",
    fontsize=14,
    pad=12,
)

ax.tick_params(
    axis="both",
    length=0,
)

# White cell boundaries
ax.set_xticks(
    np.arange(-0.5, len(modalities), 1),
    minor=True,
)

ax.set_yticks(
    np.arange(-0.5, len(heatmap_df), 1),
    minor=True,
)

ax.grid(
    which="minor",
    color="white",
    linestyle="-",
    linewidth=1.5,
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)


plt.tight_layout()

plt.savefig(
    "results/modality_heatmap.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "results/modality_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()