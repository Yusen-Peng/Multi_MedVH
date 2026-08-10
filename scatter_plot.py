import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Load results
# ============================================================

df = pd.read_csv("results/main_results.csv")

# GPT-4o has unknown parameter count, so exclude it from the
# parameter-scaling plot for now.
plot_df = df.dropna(subset=["parameters"]).copy()


# ============================================================
# Define model families
# ============================================================

families = {
    "LLaVA-1.5": [
        "LLaVA-1.5-7B",
        "LLaVA-1.5-13B",
    ],
    "Qwen2-VL": [
        "Qwen2-VL-2B",
        "Qwen2-VL-7B",
    ],
    "Qwen3-VL": [
        "Qwen3-VL-4B",
        "Qwen3-VL-8B",
    ],
    "MedGemma": [
        "MedGemma-4B",
        "MedGemma-27B",
    ],
}

# LLaVA-Med only has one size, so plot it as an isolated point.
isolated_models = [
    "LLaVA-Med-7B",
]


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5.5))

# Connected families
for family, models in families.items():

    family_df = (
        plot_df[plot_df["model"].isin(models)]
        .sort_values("parameters")
    )

    ax.plot(
        family_df["parameters"],
        family_df["Overall"],
        marker="o",
        markersize=8,
        linewidth=2,
        linestyle="--",      # or ls="--"
        label=family,
    )

    # Point labels
    for _, row in family_df.iterrows():
        ax.annotate(
            row["model"],
            (row["parameters"], row["Overall"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )


# Isolated models
for model in isolated_models:

    row = plot_df[plot_df["model"] == model].iloc[0]

    ax.scatter(
        row["parameters"],
        row["Overall"],
        s=70,
        marker="X",
        label=model,
    )

    ax.annotate(
        model,
        (row["parameters"], row["Overall"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )


# ============================================================
# Formatting
# ============================================================

ax.set_xscale("log")

ax.set_xlabel("Number of Parameters (B)", fontsize=12)
ax.set_ylabel("Overall Accuracy (%)", fontsize=12)

ax.set_title(
    "Overall Accuracy vs. Model Size",
    fontsize=14,
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.3,
)

ax.legend(
    frameon=False,
    fontsize=9,
)

plt.tight_layout()

plt.savefig(
    "results/accuracy_vs_parameters.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "results/accuracy_vs_parameters.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()