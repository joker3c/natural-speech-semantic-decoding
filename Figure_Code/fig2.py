"""Plotting script for Figure 2.

Panels:
    A (right): Per-participant word2vec decoding accuracy.
    B: Relative decoding improvement across 5 / 10 / 20 / 30 semantic clusters.
    C: Phoneme-cluster control decoding accuracy.
    D: One combined panel comparing fastText, GloVe, lemmatized content-word
        ("PosTag" pipeline) and semantic-dimension decoding against the shuffle
        control.

Outputs are written to ``Figure/fig2/``. All data paths are
resolved relative to this file's location, so the script can be run from any
working directory.
"""

import json
import sys
from pathlib import Path
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.pyplot import rc

ROOT = Path(__file__).resolve().parents[1]  # repository root
OUTPUT_DIR = ROOT / "Figure" / "fig2"

sys.path.insert(0, str(Path(__file__).resolve().parent))  # vendored Figure_Code/Code
from Code.modules.decodingStatistics import DecodingStatistics

plt.rcParams["figure.figsize"] = [15, 5]
plt.rcParams["font.size"] = "15"
mpl.rcParams["pdf.fonttype"] = 42
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})


def absolutize_config_paths(params: dict) -> dict:
    """Resolve ``../``-prefixed paths in a config dict against ``ROOT``.

    The stats_config.json files were written for a working directory that is a
    direct child of the repository root, so ``../X`` means ``ROOT / X``.
    Trailing separators are preserved because ``decodingStatistics``
    concatenates paths as strings. ``figure_path`` is redirected to
    ``OUTPUT_DIR`` because the plotting methods of ``DecodingStatistics``
    (the only consumers of ``figure_path``) are not used by this script.
    """
    fixed = dict(params)
    for key, value in fixed.items():
        if not isinstance(value, str) or not value.startswith("../"):
            continue
        if key == "figure_path":
            fixed[key] = str(OUTPUT_DIR) + "/"
        else:
            fixed[key] = str(ROOT / value[3:]) + ("/" if value.endswith("/") else "")
    return fixed


# ---------------------------------------------------------------------------
# Panels A (right) / B / C: main decoding statistics
# ---------------------------------------------------------------------------


def plot_embedding_model_comparison(
    decoding_statistics: pd.DataFrame,
    matrix_type: str,
    num_draws: int,
    reg_strength: float,
    time_before: int,
    time_after: int,
    n_cluster: int,
    bootstrap_test: bool = True,
) -> None:
    """Plot per-participant word2vec decoding accuracy (Figure 2A, right).

    Args:
        decoding_statistics: Aggregated decoding statistics DataFrame.
        matrix_type: Feature matrix type used in the analysis.
        num_draws: Number of bootstrap draws.
        reg_strength: Regularization strength.
        time_before: Time window start (ms).
        time_after: Time window end (ms).
        n_cluster: Number of semantic clusters.
        bootstrap_test: Whether bootstrap-based accuracies are used.
    """
    _ = bootstrap_test  # retained for API consistency with related functions

    filtered_df = decoding_statistics[
        (decoding_statistics.n_cluster == n_cluster)
        & (decoding_statistics.matrix_type == matrix_type)
        & (decoding_statistics.num_draws == num_draws)
        & (decoding_statistics.reg_strength == reg_strength)
        & (decoding_statistics.time_before == time_before)
        & (decoding_statistics.time_after == time_after)
    ]
    filtered_df = filtered_df.drop(
        columns=["n_cluster", "matrix_type", "num_draws", "reg_strength", "time_before", "time_after"]
    )

    df_word2vec_per_seed = filtered_df[filtered_df.embedding_model == "word2vec"][
        ["participant_id", "clustering_seed", "accuracy_bootstrap", "accuracy_bootstrap_shuffle"]
    ]
    df_word2vec = filtered_df[filtered_df.embedding_model == "word2vec"][
        ["participant_id", "accuracy_bootstrap", "accuracy_bootstrap_shuffle"]
    ]

    participant_ids = decoding_statistics.participant_id.unique()
    generic_pids = [
        f"ID0{i}" if i < 10 else f"ID{i}" for i in range(1, len(participant_ids) + 1)
    ]

    word2vec_mean = df_word2vec.groupby("participant_id").mean()["accuracy_bootstrap"].values
    shuffle_mean_word2vec = (
        df_word2vec.groupby("participant_id").mean()["accuracy_bootstrap_shuffle"].values
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # word2vec per participant
    plt.bar(
        np.arange(1, len(participant_ids) + 1) - 0.2,
        word2vec_mean * 100,
        label="Actual",
        width=0.4,
        color="#13679e",
        alpha=0.5,
    )
    for pid in participant_ids:
        participant_values = df_word2vec_per_seed[
            df_word2vec_per_seed["participant_id"] == pid
        ].groupby("clustering_seed")["accuracy_bootstrap"].mean() * 100
        x = np.linspace(
            1 + participant_ids.tolist().index(pid) - 0.2 - 0.15,
            1 + participant_ids.tolist().index(pid) - 0.2 + 0.15,
            20,
        )
        plt.scatter(x, participant_values, s=8, color="#13679e", label="_nolegend_")

    plt.bar(
        np.arange(1, len(participant_ids) + 1) + 0.2,
        shuffle_mean_word2vec * 100,
        label="Control",
        width=0.4,
        color="tab:gray",
        alpha=0.5,
    )
    for pid in participant_ids:
        participant_values = df_word2vec_per_seed[
            df_word2vec_per_seed["participant_id"] == pid
        ].groupby("clustering_seed")["accuracy_bootstrap_shuffle"].mean() * 100
        x = np.linspace(
            1 + participant_ids.tolist().index(pid) + 0.2 - 0.15,
            1 + participant_ids.tolist().index(pid) + 0.2 + 0.15,
            20,
        )
        plt.scatter(x, participant_values, s=8, color="tab:gray", label="_nolegend_")

    plt.ylim([0, 40])
    plt.yticks([0, 10, 20, 30, 40])
    plt.xlim([0, 15])
    plt.xticks(np.arange(1, len(participant_ids) + 1), generic_pids)
    plt.hlines(10, 0, len(participant_ids) + 1 + 0.1, colors="black", linestyles="--", label="Chance", alpha=0.3)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Participant")
    plt.ylabel("Decoding accuracy (%)")
    # Explicit handle order: legend entries appear as Actual, Control, Chance
    # (matplotlib would list the chance Line2D first when collecting itself).
    legend_handles = [
        Patch(facecolor="#13679e", label="Actual"),
        Patch(facecolor="tab:gray", label="Control"),
        plt.Line2D([0], [0], color="black", linestyle="--", alpha=0.3, label="Chance"),
    ]
    plt.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=12)
    plt.savefig(OUTPUT_DIR / "A_result_w2c.pdf", dpi=300)
    plt.close("all")  # reset figure state (headless backends do not clear on show)


def plot_phonetic_control_decoding(
    decoding_statistics: pd.DataFrame, bootstrap_test: bool = True
) -> None:
    """Plot decoding accuracy for phoneme-cluster control.

    Args:
        decoding_statistics: Aggregated decoding statistics DataFrame.
        bootstrap_test: Whether bootstrap-based accuracies are used.
    """
    _ = bootstrap_test  # retained for API consistency

    participant_ids = decoding_statistics.participant_id.unique()
    df_phonetic = decoding_statistics[
        ["participant_id", "accuracy_bootstrap", "accuracy_bootstrap_shuffle"]
    ]

    phonetic_mean = df_phonetic.groupby("participant_id").mean()["accuracy_bootstrap"].values
    shuffle_mean_phonetic = (
        df_phonetic.groupby("participant_id").mean()["accuracy_bootstrap_shuffle"].values
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4.5, 5))
    plt.bar(np.arange(1) + 6, phonetic_mean.mean() * 100, width=0.3, alpha=0.7, color="#1F8C2F", label="Actual")
    plt.bar(
        np.arange(1) + 6 + 0.3,
        shuffle_mean_phonetic.mean() * 100,
        width=0.3,
        alpha=0.5,
        color="tab:gray",
        label="Control",
    )
    np.random.seed(0)
    x = np.linspace(5.90, 6.10, 14)
    for pid in participant_ids:
        participant_value = df_phonetic[df_phonetic["participant_id"] == pid][
            "accuracy_bootstrap"
        ].mean() * 100
        plt.plot(x[participant_ids.tolist().index(pid)], participant_value, ".", color="#1F8C2F", label="_nolegend_")
    x2 = np.linspace(6.20, 6.40, 14)
    for pid in participant_ids:
        participant_value = df_phonetic[df_phonetic["participant_id"] == pid][
            "accuracy_bootstrap_shuffle"
        ].mean() * 100
        plt.plot(x2[participant_ids.tolist().index(pid)], participant_value, ".", color="gray", label="_nolegend_")
    plt.ylim([0, 40])
    plt.yticks([0, 20, 40])
    plt.ylabel("Decoding accuracy (%)")
    plt.xlim([5.7, 6.6])
    plt.xticks([])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(["Actual", "Control"], loc="upper left", frameon=False, fontsize=12)
    plt.savefig(OUTPUT_DIR / "C_result_Decodingacc_phonemeclusters.pdf", dpi=300)
    plt.close("all")  # reset figure state (headless backends do not clear on show)


def plot_relative_increase_with_dots(
    decoding_statistics_5: pd.DataFrame,
    decoding_statistics_10: pd.DataFrame,
    decoding_statistics_20: pd.DataFrame,
    decoding_statistics_30: pd.DataFrame,
    cluster_5: pd.DataFrame,
    cluster_10: pd.DataFrame,
    cluster_20: pd.DataFrame,
    cluster_30: pd.DataFrame,
    bootstrap_test: bool = True,
) -> None:
    """Plot relative increase per participant with individual seed dots.

    Args:
        decoding_statistics_5: Statistics for 5 clusters.
        decoding_statistics_10: Statistics for 10 clusters.
        decoding_statistics_20: Statistics for 20 clusters.
        decoding_statistics_30: Statistics for 30 clusters.
        cluster_5: Raw combined data for 5 clusters.
        cluster_10: Raw combined data for 10 clusters.
        cluster_20: Raw combined data for 20 clusters.
        cluster_30: Raw combined data for 30 clusters.
        bootstrap_test: Whether bootstrap-based accuracies are used.
    """
    _ = bootstrap_test  # retained for API consistency

    participant_ids = decoding_statistics_5.participant_id.unique()
    mean_column = "error_mean"

    plt.rcParams["figure.figsize"] = [15, 5]
    plt.rcParams.update({"font.size": 15})
    bar_width = 0.2

    # 5-cluster bars
    for group_idx, group in decoding_statistics_5.groupby(
        ["embedding_model", "matrix_type", "n_cluster", "num_draws", "reg_strength", "time_before", "time_after"]
    ):
        first_run = True
        counter = 1
        for participant_id in participant_ids:
            df_i = group[group.participant_id == participant_id]
            participant_values = cluster_5[cluster_5["participant_id"] == participant_id].groupby("clustering_seed")["error"].mean()
            x = np.linspace(counter - 0.3 - 0.05, counter - 0.3 + 0.05, 20)
            if first_run:
                plt.bar(
                    counter - 0.3,
                    df_i[mean_column],
                    width=bar_width,
                    alpha=0.7,
                    color="#1F77B4",
                    label="5",
                )
                plt.scatter(x, participant_values, s=8, color="#1F77B4", label="_nolegend_")
                first_run = False
            else:
                plt.bar(counter - 0.3, df_i[mean_column], width=bar_width, alpha=0.7, color="#1F77B4")
                plt.scatter(x, participant_values, s=8, color="#1F77B4", label="_nolegend_")
            counter += 1

    # 10-cluster bars
    for group_idx, group in decoding_statistics_10.groupby(
        ["embedding_model", "matrix_type", "n_cluster", "num_draws", "reg_strength", "time_before", "time_after"]
    ):
        plt.rcParams.update({"font.size": 14})
        first_run = True
        counter = 1
        for participant_id in participant_ids:
            df_i = group[group.participant_id == participant_id]
            participant_values = cluster_10[cluster_10["participant_id"] == participant_id].groupby("clustering_seed")["error"].mean()
            x = np.linspace(counter - 0.1 - 0.05, counter - 0.1 + 0.05, 20)
            if first_run:
                plt.bar(
                    counter - 0.1,
                    df_i[mean_column],
                    width=bar_width,
                    alpha=0.7,
                    color="#AEC7E8",
                    label="10",
                )
                plt.scatter(x, participant_values, s=8, color="#AEC7E8", label="_nolegend_")
                first_run = False
            else:
                plt.bar(counter - 0.1, df_i[mean_column], width=bar_width, alpha=0.7, color="#AEC7E8")
                plt.scatter(x, participant_values, s=8, color="#AEC7E8", label="_nolegend_")
            counter += 1

    # 20-cluster bars
    for group_idx, group in decoding_statistics_20.groupby(
        ["embedding_model", "matrix_type", "n_cluster", "num_draws", "reg_strength", "time_before", "time_after"]
    ):
        plt.rcParams.update({"font.size": 14})
        first_run = True
        counter = 1
        for participant_id in participant_ids:
            df_i = group[group.participant_id == participant_id]
            participant_values = cluster_20[cluster_20["participant_id"] == participant_id].groupby("clustering_seed")["error"].mean()
            x = np.linspace(counter + 0.1 - 0.05, counter + 0.1 + 0.05, 20)
            if first_run:
                plt.bar(
                    counter + 0.1,
                    df_i[mean_column],
                    width=bar_width,
                    alpha=0.7,
                    color="#E4E4A0",
                    label="20",
                )
                plt.scatter(x, participant_values, s=8, color="#E4E4A0", label="_nolegend_")
                first_run = False
            else:
                plt.bar(counter + 0.1, df_i[mean_column], width=bar_width, alpha=0.7, color="#E4E4A0")
                plt.scatter(x, participant_values, s=8, color="#E4E4A0", label="_nolegend_")
            counter += 1

    # 30-cluster bars
    for group_idx, group in decoding_statistics_30.groupby(
        ["embedding_model", "matrix_type", "n_cluster", "num_draws", "reg_strength", "time_before", "time_after"]
    ):
        plt.rcParams.update({"font.size": 14})
        first_run = True
        counter = 1
        for participant_id in participant_ids:
            df_i = group[group.participant_id == participant_id]
            participant_values = cluster_30[cluster_30["participant_id"] == participant_id].groupby("clustering_seed")["error"].mean()
            x = np.linspace(counter + 0.3 - 0.05, counter + 0.3 + 0.05, 20)
            if first_run:
                plt.bar(
                    counter + 0.3,
                    df_i[mean_column],
                    width=bar_width,
                    alpha=0.7,
                    color="#A9D9A9",
                    label="30",
                )
                plt.scatter(x, participant_values, s=8, color="#A9D9A9", label="_nolegend_")
                first_run = False
            else:
                plt.bar(counter + 0.3, df_i[mean_column], width=bar_width, alpha=0.7, color="#A9D9A9")
                plt.scatter(x, participant_values, s=8, color="#A9D9A9", label="_nolegend_")
            counter += 1

    generic_pids = [
        f"ID0{i}" if i < 10 else f"ID{i}" for i in range(1, len(participant_ids) + 1)
    ]
    plt.xticks(np.arange(1, len(generic_pids) + 1), generic_pids)
    plt.xlabel("Participant")
    plt.ylabel("Relative increase of accuracy (%)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim(0, len(generic_pids) + 1)
    plt.legend(loc="upper left", ncol=4, frameon=False, fontsize=12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "B_5_30Cluster_RelativeError.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close("all")  # reset figure state (headless backends do not clear on show)


def compute_relative_error(row: pd.Series) -> float:
    """Relative increase of actual over shuffle decoding accuracy, in percent."""
    return (
        (row["accuracy_bootstrap"] - row["accuracy_bootstrap_shuffle"])
        / row["accuracy_bootstrap_shuffle"]
    ) * 100


def plot_cluster_count_comparison(decoding_combined: pd.DataFrame) -> None:
    """Prepare per-cluster-count data and call the relative-increase plots."""
    mask_word2vec = decoding_combined.embedding_model == "word2vec"

    decoding_combined_5 = decoding_combined[(mask_word2vec) & (decoding_combined.n_cluster == 5)].copy()
    decoding_combined_10 = decoding_combined[(mask_word2vec) & (decoding_combined.n_cluster == 10)].copy()
    decoding_combined_20 = decoding_combined[(mask_word2vec) & (decoding_combined.n_cluster == 20)].copy()
    decoding_combined_30 = decoding_combined[(mask_word2vec) & (decoding_combined.n_cluster == 30)].copy()

    for df in [decoding_combined_5, decoding_combined_10, decoding_combined_20, decoding_combined_30]:
        df["error"] = df.apply(compute_relative_error, axis=1)

    cluster_5_raw = decoding_combined_5.copy()
    cluster_10_raw = decoding_combined_10.copy()
    cluster_20_raw = decoding_combined_20.copy()
    cluster_30_raw = decoding_combined_30.copy()

    grouping_columns = [
        "participant_id",
        "embedding_model",
        "matrix_type",
        "n_cluster",
        "num_draws",
        "reg_strength",
        "time_before",
        "time_after",
    ]

    def aggregate_error(df: pd.DataFrame) -> pd.DataFrame:
        mean_df = df.groupby(grouping_columns, as_index=False)["error"].mean()
        mean_df = mean_df.rename(columns={"error": "error_mean"})
        ste_df = df.groupby(grouping_columns, as_index=False)["error"].sem()
        ste_df = ste_df.rename(columns={"error": "error_ste"})
        return pd.merge(mean_df, ste_df, on=grouping_columns, how="left")

    decoding_combined_5 = aggregate_error(decoding_combined_5)
    decoding_combined_10 = aggregate_error(decoding_combined_10)
    decoding_combined_20 = aggregate_error(decoding_combined_20)
    decoding_combined_30 = aggregate_error(decoding_combined_30)

    plot_relative_increase_with_dots(
        decoding_combined_5,
        decoding_combined_10,
        decoding_combined_20,
        decoding_combined_30,
        cluster_5_raw,
        cluster_10_raw,
        cluster_20_raw,
        cluster_30_raw,
    )


# ---------------------------------------------------------------------------
# Panel D: embedding models and lexical subsets vs. shuffle control
# ---------------------------------------------------------------------------


def load_decoding_results(
    configs_folder: Path, stats_folder: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aggregated decoding statistics, computing them first if missing.

    Args:
        configs_folder: Folder containing ``stats_config.json`` for the pipeline.
        stats_folder: Folder holding the cached ``decoding_combined_all.pkl``
            and ``decoding_statistics.csv`` results.

    Returns:
        Tuple of (decoding_statistics, decoding_combined).
    """
    with open(configs_folder / "stats_config.json", "r") as file:
        config = json.load(file)
    statistics = DecodingStatistics(**absolutize_config_paths(config["stats_params"]))

    decoding_statistics_path = stats_folder / "decoding_statistics.csv"
    decoding_combined_path = stats_folder / "decoding_combined_all.pkl"

    if decoding_statistics_path.exists() and decoding_combined_path.exists():
        decoding_statistics = pd.read_csv(decoding_statistics_path)
        decoding_combined = pd.read_pickle(decoding_combined_path)
    else:
        print("Calculating statistics...")
        decoding_combined = statistics.get_decoding_statistics3()
        decoding_combined.to_pickle(decoding_combined_path)
        decoding_statistics = statistics.get_decoding_statistics2()
        decoding_statistics.to_csv(decoding_statistics_path, index=False)

    return decoding_statistics, decoding_combined


def plot_embedding_lexical_subsets(decoding_combined: pd.DataFrame) -> None:
    """Plot merged Figure 2D: FastText / GloVe / Dimension / Content decoding.

    Each group shows the mean decoding accuracy as a bar next to its shuffle
    control, with one dot per participant (per-participant mean). All
    accuracies are stored as fractions and are converted to percent here.
    """
    mask = (
        (decoding_combined.n_cluster == 10)
        & (decoding_combined.matrix_type == "selective")
        & (decoding_combined.num_draws == 10)
        & (decoding_combined.reg_strength == 10e-5)
        & (decoding_combined.time_before == 500)
        & (decoding_combined.time_after == 0)
    )
    main_df = decoding_combined[mask]

    _, pos_tag_combined = load_decoding_results(
        ROOT / "Code" / "configs" / "Control_PosTag",
        ROOT / "Statistics" / "Stats_PosTag",
    )
    _, dimension_combined = load_decoding_results(
        ROOT / "Code" / "configs" / "Control_SemanticDimension",
        ROOT / "Statistics" / "Stats_SemanticDimension",
    )

    groups = [
        ("FastText", "#13679e", main_df[main_df.embedding_model == "fasttext"]),
        ("GloVe", "#8B8000", main_df[main_df.embedding_model == "glove"]),
        ("Dimension", "#A688E0", dimension_combined),
        ("Content", "#C0392B", pos_tag_combined),
    ]

    plt.figure(figsize=(7, 5))
    for group_idx, (group_label, group_color, group_df) in enumerate(groups, start=1):
        actual_means = (
            group_df.groupby("participant_id")["accuracy_bootstrap"].mean() * 100
        )
        shuffle_means = (
            group_df.groupby("participant_id")["accuracy_bootstrap_shuffle"].mean() * 100
        )

        plt.bar(
            group_idx - 0.15,
            actual_means.mean(),
            width=0.3,
            alpha=0.5,
            color=group_color,
        )
        plt.bar(
            group_idx + 0.15,
            shuffle_means.mean(),
            width=0.3,
            alpha=0.5,
            color="tab:gray",
        )

        x_actual = np.linspace(group_idx - 0.25, group_idx - 0.05, len(actual_means))
        plt.scatter(x_actual, actual_means.values, s=8, color=group_color, label="_nolegend_")
        x_shuffle = np.linspace(group_idx + 0.05, group_idx + 0.25, len(shuffle_means))
        plt.scatter(x_shuffle, shuffle_means.values, s=8, color="tab:gray", label="_nolegend_")

    plt.ylim([0, 40])
    plt.yticks([0, 20, 40])
    plt.xlim([0.4, 4.6])
    plt.xticks([])
    plt.ylabel("Decoding accuracy (%)")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Explicit handle order: the column-major legend fill then reproduces the
    # two rows (FastText, GloVe, Dimension / Content, Control).
    handles = [
        Patch(facecolor="#13679e", label="FastText"),
        Patch(facecolor="#C0392B", label="Content"),
        Patch(facecolor="#8B8000", label="GloVe"),
        Patch(facecolor="tab:gray", label="Control"),
        Patch(facecolor="#A688E0", label="Dimension"),
    ]
    plt.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "D_result_embedding_models.pdf", dpi=300)
    plt.close("all")  # reset figure state (headless backends do not clear on show)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all Figure 2 panels (PDFs saved under ``OUTPUT_DIR``)."""
    # Panel C: phoneme-cluster control
    configs_folder = ROOT / "Code" / "configs" / "Control_Phoneme"
    with open(configs_folder / "stats_config.json", "r") as file:
        config = json.load(file)
    statistics = DecodingStatistics(**absolutize_config_paths(config["stats_params"]))
    decoding_combined_phoneme = statistics.get_decoding_statistics3()
    plot_phonetic_control_decoding(decoding_combined_phoneme)

    # Panels A (right) / B: main semantic decoding
    configs_folder = ROOT / "Code" / "configs" / "SemanticDecode"
    with open(configs_folder / "stats_config.json", "r") as file:
        config = json.load(file)
    statistics = DecodingStatistics(**absolutize_config_paths(config["stats_params"]))
    print("Calculating statistics...")

    decoding_statistics_path = ROOT / "Statistics" / "Statistic_Result" / "decoding_statistics.csv"
    decoding_combined_path = ROOT / "Statistics" / "Statistic_Result" / "decoding_combined_all.pkl"

    if decoding_statistics_path.exists() and decoding_combined_path.exists():
        decoding_statistics = pd.read_csv(decoding_statistics_path)
        decoding_combined = pd.read_pickle(decoding_combined_path)
    else:
        decoding_combined = statistics.get_decoding_statistics3()
        decoding_combined_path.parent.mkdir(parents=True, exist_ok=True)
        decoding_combined.to_pickle(decoding_combined_path)
        decoding_statistics = statistics.get_decoding_statistics2()
        decoding_statistics.to_csv(decoding_statistics_path, index=False)

    plot_embedding_model_comparison(
        decoding_combined,
        matrix_type="selective",
        num_draws=10,
        reg_strength=10e-5,
        time_before=500,
        time_after=0,
        n_cluster=10,
    )
    plot_cluster_count_comparison(decoding_combined)

    # Panel D: embedding models + lexical subsets (merged single panel)
    plot_embedding_lexical_subsets(decoding_combined)


if __name__ == "__main__":
    main()
