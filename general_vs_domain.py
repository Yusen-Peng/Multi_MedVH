import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load results
# ============================================================

df = pd.read_csv("results/main_results.csv")

metrics = [
    "Overall",
    "CT",
    "MRI",
    "CXR",
    "ECG",
    "Pathology",
]


# ============================================================
# Matched-size comparisons
# ============================================================

comparisons = [
    {
        "title": "7B Models",
        "general": "LLaVA-1.5-7B",
        "medical": "LLaVA-Med-7B",
    },
    {
        "title": "4B Models",
        "general": "Qwen3-VL-4B",
        "medical": "MedGemma-4B",
    },
]


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    sharey=True,
)

x = np.arange(len(metrics))
width = 0.36


for ax, comparison in zip(axes, comparisons):

    general_row = df[
        df["model"] == comparison["general"]
    ].iloc[0]

    medical_row = df[
        df["model"] == comparison["medical"]
    ].iloc[0]

    general_values = general_row[metrics].astype(float).values
    medical_values = medical_row[metrics].astype(float).values


    # --------------------------------------------------------
    # Bars
    # --------------------------------------------------------

    bars_general = ax.bar(
        x - width / 2,
        general_values,
        width,
        label=comparison["general"],
    )

    bars_medical = ax.bar(
        x + width / 2,
        medical_values,
        width,
        label=comparison["medical"],
    )


    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    ax.bar_label(
        bars_general,
        fmt="%.1f",
        fontsize=8,
        padding=2,
    )

    ax.bar_label(
        bars_medical,
        fmt="%.1f",
        fontsize=8,
        padding=2,
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_title(
        comparison["title"],
        fontsize=13,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        metrics,
        rotation=30,
        ha="right",
    )

    ax.set_ylim(0, 85)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        frameon=False,
        fontsize=9,
    )


axes[0].set_ylabel(
    "Accuracy (%)",
    fontsize=12,
)

fig.suptitle(
    "General-Purpose vs. Medical-Specialized LVLMs",
    fontsize=15,
    y=1.01,
)

plt.tight_layout()

plt.savefig(
    "results/general_vs_medical.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "results/general_vs_medical.png",
    dpi=300,
    bbox_inches="tight",
)



plt.show()