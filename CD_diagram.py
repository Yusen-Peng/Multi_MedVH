import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, studentized_range


# ============================================================
# Configuration
# ============================================================

OUTPUT_ROOT = "outputs"

OUTPUT_PDF = "results/model_cd_diagram.pdf"

ALPHA = 0.05


# ============================================================
# Models
# ============================================================

# folder_name : display_name
MODEL_MAP = {
    "gpt4o": "GPT-4o",

    "llava15_7b": "LLaVA-1.5-7B",
    "llava15_13b": "LLaVA-1.5-13B",

    "qwen2vl_2b": "Qwen2-VL-2B",
    "qwen2vl_7b": "Qwen2-VL-7B",

    "qwen3vl_4b": "Qwen3-VL-4B",
    "qwen3vl_8b": "Qwen3-VL-8B",

    "llava_med15_7b": "LLaVA-Med-7B",

    "medgemma_4b": "MedGemma-4B",
    "medgemma_27b": "MedGemma-27B",
}


# ============================================================
# Question types
#
# IMPORTANT:
# these must exactly match the CSV column names
# ============================================================

QUESTION_TYPES = [
    "baseline",
    "modality_mismatch",
    "incorrect_premise",
    "false_suggestions",
]


# ============================================================
# Load all model results
# ============================================================

def load_all_results():

    all_results = {}

    expected_tasks = None

    for folder_name, display_name in MODEL_MAP.items():

        csv_path = os.path.join(
            OUTPUT_ROOT,
            folder_name,
            "report",
            "results_modality_x_question_type.csv",
        )

        if not os.path.exists(csv_path):
            print(f"[WARNING] Missing: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # ----------------------------------------------------
        # Verify required columns
        # ----------------------------------------------------

        required_columns = ["modality"] + QUESTION_TYPES

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{csv_path} missing columns: {missing_columns}\n"
                f"Found columns: {list(df.columns)}"
            )

        # ----------------------------------------------------
        # Convert:
        #
        # modality | baseline | mismatch | ...
        #
        # into:
        #
        # CT__baseline            score
        # CT__modality_mismatch   score
        # MRI__baseline           score
        # ...
        #
        # ====================================================
        # 5 modalities x 4 question types = 20 tasks
        # ====================================================
        # ----------------------------------------------------

        model_scores = {}

        for _, row in df.iterrows():

            modality = row["modality"]

            for question_type in QUESTION_TYPES:

                task_name = f"{modality}__{question_type}"

                model_scores[task_name] = row[question_type]

        model_scores = pd.Series(
            model_scores,
            name=display_name,
            dtype=float,
        )

        # ----------------------------------------------------
        # Verify all models have exactly the same tasks
        # ----------------------------------------------------

        current_tasks = set(model_scores.index)

        if expected_tasks is None:
            expected_tasks = current_tasks
        else:
            if current_tasks != expected_tasks:

                missing = expected_tasks - current_tasks
                extra = current_tasks - expected_tasks

                raise ValueError(
                    f"Task mismatch for {display_name}\n"
                    f"Missing: {missing}\n"
                    f"Extra: {extra}"
                )

        all_results[display_name] = model_scores

    # Each row = task
    # Each column = model
    score_matrix = pd.DataFrame(all_results)

    return score_matrix


# ============================================================
# Compute ranks
# ============================================================

def compute_ranks(score_matrix):

    # Higher accuracy = better
    #
    # rank 1 = best
    #
    # method="average" properly handles ties

    rank_matrix = score_matrix.rank(
        axis=1,
        method="average",
        ascending=False,
    )

    return rank_matrix


# ============================================================
# Friedman + Nemenyi
# ============================================================

def run_statistics(
    rank_matrix,
    alpha=0.05,
):

    models = rank_matrix.columns.tolist()

    N = len(rank_matrix)
    k = len(models)

    # --------------------------------------------------------
    # Friedman omnibus test
    # --------------------------------------------------------

    friedman_stat, friedman_p = friedmanchisquare(
        *[
            rank_matrix[model].to_numpy()
            for model in models
        ]
    )

    # --------------------------------------------------------
    # Average ranks
    # --------------------------------------------------------

    avg_ranks = (
        rank_matrix
        .mean(axis=0)
        .sort_values()
    )

    # --------------------------------------------------------
    # Nemenyi Critical Difference
    #
    # CD =
    #
    # q_alpha * sqrt(
    #     k(k+1)
    #     -------
    #       6N
    # )
    #
    # scipy's studentized_range gives the full
    # Studentized Range statistic, so divide by sqrt(2)
    # for the conventional Nemenyi q.
    # --------------------------------------------------------

    q_studentized = studentized_range.ppf(
        1.0 - alpha,
        k,
        np.inf,
    )

    q_nemenyi = (
        q_studentized
        / np.sqrt(2.0)
    )

    cd = (
        q_nemenyi
        * np.sqrt(
            k * (k + 1)
            / (6.0 * N)
        )
    )

    return {
        "N": N,
        "k": k,

        "friedman_stat": friedman_stat,
        "friedman_p": friedman_p,

        "q_studentized": q_studentized,
        "q_nemenyi": q_nemenyi,

        "cd": cd,

        "average_ranks": avg_ranks,
    }


# ============================================================
# Find maximal non-significant groups
# ============================================================

def find_non_significant_groups(
    avg_ranks,
    cd,
):
    """
    Models are not significantly different when

        |rank_i - rank_j| <= CD

    Because the models are ordered by rank, valid groups are
    contiguous intervals.

    We keep only maximal groups.
    """

    names = avg_ranks.index.tolist()
    ranks = avg_ranks.values

    groups = []

    n = len(names)

    for i in range(n):

        for j in range(i + 1, n):

            if ranks[j] - ranks[i] <= cd:

                groups.append((i, j))

            else:
                break

    # Remove groups fully contained inside a larger group
    maximal_groups = []

    for group in groups:

        i, j = group

        contained = False

        for other in groups:

            oi, oj = other

            if (
                other != group
                and oi <= i
                and oj >= j
                and (oi < i or oj > j)
            ):
                contained = True
                break

        if not contained:
            maximal_groups.append(group)

    return maximal_groups


# ============================================================
# Plot CD diagram
# ============================================================

def plot_cd_diagram(
    stats,
    output_pdf,
):

    avg_ranks = stats["average_ranks"]
    cd = stats["cd"]

    k = stats["k"]

    groups = find_non_significant_groups(
        avg_ranks,
        cd,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11.0, 5.2)
    )

    ax.set_xlim(
        0.3,
        k + 0.7,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.axis("off")


    # ========================================================
    # Rank axis
    # ========================================================

    axis_y = 0.65

    ax.plot(
        [1, k],
        [axis_y, axis_y],
        color="black",
        linewidth=1.2,
    )

    for rank in range(1, k + 1):

        ax.plot(
            [rank, rank],
            [axis_y - 0.015, axis_y + 0.015],
            color="black",
            linewidth=1.0,
        )

        ax.text(
            rank,
            axis_y + 0.035,
            str(rank),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.text(
        (1 + k) / 2,
        axis_y + 0.10,
        "Average Rank",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


    # ========================================================
    # Split models left/right
    # ========================================================

    ordered_models = avg_ranks.index.tolist()

    midpoint = int(np.ceil(k / 2))

    left_models = ordered_models[:midpoint]
    right_models = ordered_models[midpoint:]


    left_y_start = 0.44
    left_spacing = 0.075

    right_y_start = 0.44
    right_spacing = 0.075


    # ========================================================
    # Left-side models
    # ========================================================

    left_x_text = 0.25


    for idx, model in enumerate(left_models):

        rank = avg_ranks[model]

        y = left_y_start - idx * left_spacing

        elbow_x = 0.75

        # vertical connector from rank
        ax.plot(
            [rank, rank],
            [axis_y, y],
            color="black",
            linewidth=0.8,
        )

        # horizontal connector
        ax.plot(
            [elbow_x, rank],
            [y, y],
            color="black",
            linewidth=0.8,
        )

        # model name
        ax.text(
            left_x_text,
            y-0.02,
            f"{model}  ({rank:.2f})",
            ha="left",
            va="center",
            fontsize=10.5,
        )


    # ========================================================
    # Right-side models
    # ========================================================

    right_x_text = k + 0.65

    for idx, model in enumerate(right_models):

        rank = avg_ranks[model]

        y = right_y_start - idx * right_spacing

        elbow_x = k + 0.25

        ax.plot(
            [rank, rank],
            [axis_y, y],
            color="black",
            linewidth=0.8,
        )

        ax.plot(
            [rank, elbow_x],
            [y, y],
            color="black",
            linewidth=0.8,
        )

        ax.text(
            right_x_text,
            y-0.02,
            f"({rank:.2f})  {model}",
            ha="right",
            va="center",
            fontsize=10.5,
        )


    # ========================================================
    # Non-significant groups
    # ========================================================

    group_y_start = 0.82
    group_spacing = 0.035

    ranks = avg_ranks.values

    for group_idx, (i, j) in enumerate(groups):

        y = group_y_start + group_idx * group_spacing

        ax.plot(
            [ranks[i], ranks[j]],
            [y, y],
            color="black",
            linewidth=3.0,
            solid_capstyle="butt",
        )


    # ========================================================
    # Critical Difference bar
    # ========================================================

    cd_y = 0.95

    cd_start = 1.0
    cd_end = cd_start + cd

    ax.plot(
        [cd_start, cd_end],
        [cd_y, cd_y],
        color="black",
        linewidth=1.5,
    )

    ax.plot(
        [cd_start, cd_start],
        [cd_y - 0.015, cd_y + 0.015],
        color="black",
        linewidth=1.2,
    )

    ax.plot(
        [cd_end, cd_end],
        [cd_y - 0.015, cd_y + 0.015],
        color="black",
        linewidth=1.2,
    )

    ax.text(
        (cd_start + cd_end) / 2,
        cd_y + 0.02,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )


    # ========================================================
    # Statistics
    # ========================================================

    p = stats["friedman_p"]

    if p < 0.001:
        p_text = r"$p < 0.001$"
    else:
        p_text = f"$p = {p:.3f}$"

    ax.text(
        (1 + k) / 2,
        0.03,
        (
            f"Friedman test: "
            rf"$\chi^2={stats['friedman_stat']:.2f}$, "
            f"{p_text}; "
            f"N={stats['N']} tasks"
        ),
        ha="center",
        va="bottom",
        fontsize=10.5,
    )


    # ========================================================
    # Save
    # ========================================================

    plt.tight_layout()

    plt.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.savefig(
        output_pdf.replace(".pdf", ".png"),
        bbox_inches="tight",
        dpi=300,
    )

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load score matrix
    # --------------------------------------------------------

    score_matrix = load_all_results()

    print("\n============================================")
    print("Score matrix")
    print("============================================")

    print(score_matrix)

    print(
        "\nShape:",
        score_matrix.shape,
    )

    # Should be:
    #
    # (20, number_of_models)


    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    if score_matrix.isna().any().any():

        print("\nMissing values detected:")

        print(
            score_matrix[
                score_matrix.isna().any(axis=1)
            ]
        )

        raise ValueError(
            "Cannot run CD diagram with missing task scores."
        )


    # --------------------------------------------------------
    # Rank models separately on every task
    # --------------------------------------------------------

    rank_matrix = compute_ranks(
        score_matrix
    )


    print("\n============================================")
    print("Per-task ranks")
    print("============================================")

    print(rank_matrix)


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = run_statistics(
        rank_matrix,
        alpha=ALPHA,
    )


    print("\n============================================")
    print("Friedman + Nemenyi")
    print("============================================")

    print(f"N tasks: {stats['N']}")
    print(f"k models: {stats['k']}")

    print(
        f"Friedman statistic: "
        f"{stats['friedman_stat']:.4f}"
    )

    print(
        f"Friedman p-value: "
        f"{stats['friedman_p']:.6g}"
    )

    print(
        f"Nemenyi q: "
        f"{stats['q_nemenyi']:.4f}"
    )

    print(
        f"Critical Difference: "
        f"{stats['cd']:.4f}"
    )


    print("\nAverage ranks:")

    for model, rank in stats["average_ranks"].items():

        print(
            f"{model:<20s} "
            f"{rank:.3f}"
        )


    # --------------------------------------------------------
    # CD diagram
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PDF),
        exist_ok=True,
    )

    plot_cd_diagram(
        stats,
        OUTPUT_PDF,
    )