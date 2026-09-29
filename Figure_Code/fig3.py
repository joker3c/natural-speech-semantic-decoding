"""Plotting functions for Figure 3.

Panels:
    A: Percentage of selected channels.
    C: Decoding using channels from three brain areas.
    D: Frequency-band contributions to decoding / channel selection
       (two PDFs: per-band selection percentage + relative channel importance).

Outputs are written to ``Figure/fig3/``. All data paths are
resolved relative to this file's location, so the script can be run from any
working directory.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.pyplot import rc

ROOT = Path(__file__).resolve().parents[1]  # repository root
OUTPUT_DIR = ROOT / "Figure" / "fig3"

sys.path.insert(0, str(Path(__file__).resolve().parent))  # vendored Figure_Code/Code
from Code.modules.decodingStatistics import DecodingStatistics


def absolutize_config_paths(params: dict) -> dict:
    """Resolve ``../``-prefixed paths in a config dict against ``ROOT``.

    Config dicts were written for a working directory that is a direct child
    of the repository root, so ``../X`` means ``ROOT / X``. Trailing
    separators are preserved because ``decodingStatistics`` and this script
    concatenate paths as strings. ``figure_path`` is redirected to
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

warnings.filterwarnings("ignore")

plt.rcParams["font.size"] = "10"
mpl.rcParams["pdf.fonttype"] = 42
plt.rcParams["figure.figsize"] = [30, 20]
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})


def file_name_valid(file_name: str, params: dict) -> bool:
    """Return True if ``file_name`` contains all expected parameter substrings."""
    for param, value in params.items():
        if str(value) not in file_name:
            return False
        if param == "time_before" and not re.search(f"timeBefore{value}", file_name):
            return False
        if param == "time_after" and not re.search(f"timeAfter{value}", file_name):
            return False
        if param == "detrend" and not re.search(f"detrend{value}", file_name):
            return False
        if param == "normalize" and not re.search(f"normalize{value}", file_name):
            return False
        if param == "n_cluster" and not re.search(f"nbrclusters{value}", file_name):
            return False
        if param == "p_value" and not re.search(f"pvalue{value}", file_name):
            return False
    return True


def get_pid(file_name: str) -> str:
    """Extract the participant ID prefix from a filename."""
    return re.search(r"(.+)(?=clustering)", file_name).group(0)


def get_group_stats(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate electrode counts per participant and brain area.

    Returns:
        Tuple of (parcellation_df, considered_electrodes, broad_area_counts,
        area_counts).
    """
    parcellation_df = pd.read_csv(f'{config["data_path"]}parcellation.csv', index_col=0)
    group_df = parcellation_df.groupby("pid")

    unique_broad_areas = parcellation_df["broad_area"].unique()
    unique_areas = parcellation_df["area"].unique()
    broad_area_combined = pd.DataFrame(columns=np.append("pid", unique_broad_areas))
    area_combined = pd.DataFrame(columns=np.append("pid", unique_areas))

    for pid, group in group_df:
        broad_area_count = group.loc[group["janca_useful"] == True, "broad_area"].value_counts()
        area_count = group.loc[group["janca_useful"] == True, "area"].value_counts()

        broad_area_combined = pd.concat(
            [broad_area_combined, pd.DataFrame({"pid": [pid], **broad_area_count})]
        )
        area_combined = pd.concat(
            [area_combined, pd.DataFrame({"pid": [pid], **area_count})]
        )

    considered_electrodes = group_df.agg({"electrode": "count", "janca_useful": "sum"})
    for pid in considered_electrodes.index:
        n_electrodes = considered_electrodes.loc[pid, "electrode"]
        n_useful = considered_electrodes.loc[pid, "janca_useful"]
        print(
            f"Participant {pid} has {n_electrodes} electrodes out of which "
            f"{n_useful} are considered for the analysis ({6 * n_useful} features)."
        )

    return parcellation_df, considered_electrodes, broad_area_combined, area_combined


def get_selective_stats(
    config: dict,
    condition_config: dict,
    feature_count_df: pd.DataFrame,
    parcellation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter, Counter, int]:
    """Read selective-electrode files and aggregate selection statistics.

    Returns:
        Tuple of (selective_features_combined, selective_stats,
        broad_area_counter, area_counter, n_seeds).
    """
    selective_features_combined = pd.DataFrame()
    selective_stats = pd.DataFrame()

    unique_broad_areas = parcellation_df["broad_area"].unique()
    unique_areas = parcellation_df["area"].unique()
    broad_area_counter = Counter(unique_broad_areas)
    area_counter = Counter(unique_areas)

    n_seeds = 0
    for file in Path(config["data_path"]).glob("*.csv"):
        if not file_name_valid(file.name, condition_config):
            continue

        print(f"For file {file.name}:")
        pid = get_pid(file.name)
        df = pd.read_csv(file)
        df["pid"] = df["participant_id"]
        del df["participant_id"]

        selective_features_combined = pd.concat([selective_features_combined, df])

        n_seeds = len(df["seed"].unique())
        n_freqbands = len(df["freqband"].unique())

        for group_id, group in df.groupby(["pid", "clustering", "n_cluster", "freqband"]):
            for sub_group_id, sub_group in group.groupby(["seed"]):
                n_selective = len(sub_group)
                selective_stats = pd.concat(
                    [
                        selective_stats,
                        pd.DataFrame(
                            {
                                "pid": pid,
                                "clustering": [group_id[1]],
                                "n_cluster": [group_id[2]],
                                "freqband": [group_id[3]],
                                "seed": [sub_group_id[0]],
                                "selective_features_count": [n_selective],
                                "selective_features_percentage": [
                                    n_selective / (n_freqbands * feature_count_df.loc[pid, "janca_useful"])
                                ],
                            }
                        ),
                    ]
                )

                for electrode_i in sub_group["electrode_name"]:
                    _, electrode_i = electrode_i.split("_", 1)
                    broad_area_i, area_i = parcellation_df.loc[
                        (parcellation_df["electrode"] == electrode_i) & (parcellation_df["pid"] == pid),
                        ["broad_area", "area"],
                    ].values[0]
                    broad_area_counter[broad_area_i] += 1
                    area_counter[area_i] += 1

    return selective_features_combined, selective_stats, broad_area_counter, area_counter, n_seeds


def plot_selective_percentage_summary(feature_df: pd.DataFrame, selective_df: pd.DataFrame) -> None:
    """Plot the boxplot of per-participant selected-channel percentages (Fig. 3A)."""
    feature_df = feature_df.copy()
    feature_df["pid_generic"] = [
        f"ID0{i}" if i < 10 else f"ID{i}" for i in range(1, len(feature_df.index) + 1)
    ]

    cluster_together = ["clustering", "n_cluster", "pid"]
    selective_df = selective_df.copy()
    selective_df["selective_features_percentage"] = 100 * selective_df.selective_features_percentage

    selective_stats_mean = selective_df.groupby(cluster_together + ["freqband"]).agg(
        {"selective_features_count": "mean", "selective_features_percentage": "mean"}
    )
    selective_stats_std = selective_df.groupby(cluster_together + ["freqband"]).agg(
        {"selective_features_count": "std", "selective_features_percentage": "std"}
    )

    selective_stats_mean["selective_features_count_mean"] = selective_stats_mean.selective_features_count
    del selective_stats_mean["selective_features_count"]
    selective_stats_mean["selective_features_count_std"] = selective_stats_std.selective_features_count
    selective_stats_mean["selective_features_percentage_mean"] = selective_stats_mean.selective_features_percentage
    selective_stats_mean["selective_features_percentage_std"] = selective_stats_std.selective_features_percentage

    selective_stats_mean_summed = selective_stats_mean.groupby(cluster_together).agg(
        {
            "selective_features_count_mean": "sum",
            "selective_features_count_std": "sum",
            "selective_features_percentage_mean": "sum",
            "selective_features_percentage_std": "sum",
        }
    )

    mean_percentage = selective_stats_mean_summed["selective_features_percentage_mean"].mean()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Boxplot of per-participant percentages
    plt.figure(figsize=(8, 6))
    center_position = 7
    box_width = 0.3

    box = plt.boxplot(
        selective_stats_mean_summed["selective_features_percentage_mean"],
        positions=[center_position],
        widths=box_width,
        showfliers=False,
        patch_artist=True,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="#7AA4D2", alpha=0.7)
    for median in box["medians"]:
        median.set(color="white", alpha=0)

    jitter = box_width / 2 * 0.9
    x = np.linspace(center_position - jitter, center_position + jitter, 14)
    plt.scatter(x, selective_stats_mean_summed["selective_features_percentage_mean"], color="#7AA4D2")

    plt.xlim([6.6, 7.4])
    plt.xticks([])
    plt.yticks([10, 15, 20, 25])
    plt.ylabel("% Selected channel")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_dir / "A_selectiveMeanPercentage_Boxplot.pdf", dpi=300)
    plt.close("all")  # reset figure state (headless backends do not clear on show)


def _read_selective_electrode_file(patient_id: str) -> pd.DataFrame:
    """Read one participant's selective-electrode CSV and pivot by frequency band."""
    file_path = (
        f"{ROOT}/Matrix/Matrix_10Cluster/{patient_id}clustering_selective_electrodes_df_"
        f"timeBefore500_timeAfter0_detrendTrue_normalizeFalse_envelope_word2vec_"
        f"nbrclusters10_pvalue0.05.csv"
    )
    df = pd.read_csv(file_path)
    grouped = df.groupby(["seed", "freqband"])["electrode_name"].apply(list).reset_index()
    pivoted = grouped.pivot(index="seed", columns="freqband", values="electrode_name").reset_index()
    pivoted.columns = ["seed"] + [
        f"selective_electrodes_freqband{col}" for col in pivoted.columns if col != "seed"
    ]
    pivoted["patient_id"] = patient_id
    pivoted["n_cluster"] = 10
    return pivoted


def plot_selection_percentage_by_band() -> None:
    """Plot the percentage of selected channels per frequency band."""
    patient_ids = [f"PT{i:02d}" for i in range(1, 15)]

    all_patients_df = pd.concat(
        [_read_selective_electrode_file(pid) for pid in patient_ids], axis=0
    ).reset_index(drop=True)

    parcellation_df = pd.read_csv(ROOT / "Parcellation" / "parcellation.csv")
    useful_channels = parcellation_df[parcellation_df["janca_useful"] == 1]
    electrode_counts = useful_channels.groupby("pid")["electrode"].count()

    length_columns = []
    for band in range(6):
        column = f"{band}_length"
        length_columns.append(column)
        all_patients_df[column] = all_patients_df[f"selective_electrodes_freqband{band}"].apply(len)

    for pid in patient_ids:
        all_patients_df.loc[all_patients_df["patient_id"] == pid, "total"] = electrode_counts.loc[pid]

    for band in range(6):
        all_patients_df[f"{band}_per"] = all_patients_df[f"{band}_length"] / all_patients_df["total"]

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    band_offsets = [6, 6.6, 7.2, 7.8, 8.4, 9]
    band_labels = ["θ", "α", "β", "low-γ", "mid-γ", "high-γ"]
    participant_ids = all_patients_df.patient_id.unique()

    # Boxplot per frequency band (Fig. 3D, top)
    plt.figure(figsize=(10, 5))
    box_width = 0.3
    box_data = [all_patients_df.groupby("patient_id")[f"{band}_per"].mean() * 100 for band in range(6)]
    box = plt.boxplot(
        box_data,
        positions=band_offsets,
        widths=box_width,
        showfliers=False,
        patch_artist=True,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="#9FBEDF", alpha=0.8)
    for median in box["medians"]:
        median.set(color="white", alpha=0)

    for i, band in enumerate(range(6)):
        x = np.linspace(band_offsets[i] - 0.1, band_offsets[i] + 0.1, 14)
        for pid in participant_ids:
            value = all_patients_df[all_patients_df["patient_id"] == pid][f"{band}_per"].mean() * 100
            plt.plot(x[participant_ids.tolist().index(pid)], value, ".", color="#9FBEDF", label="_nolegend_")

    plt.xlim([5.6, 9.4])
    plt.yticks([0, 10, 20, 30])
    plt.ylabel("% selected channel")
    plt.xticks(band_offsets, band_labels, fontsize=9, rotation=45, ha="right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(output_dir / "E_Selection_Percentage_mean_6Band_BoxPlot_Ver2.pdf", dpi=300)
    plt.close("all")  # reset figure state (headless backends do not clear on show)


def _read_feature_importance_file(patient_id: str, seed: int) -> pd.DataFrame:
    """Read feature-importance data for one participant and seed."""
    file_path = (
        f"{ROOT}/Decoding_Result/SemanticDecoding/featureImportance/{patient_id}_nclusters10_speaking_"
        f"X_selective_timebefore500_timeafter0_detrendTrue_normalizeFalse_envelope_"
        f"word2vec_numdraws10_reg0.0001_p0.05.csv"
    )
    data = np.load(file_path, allow_pickle=True)
    data = data[data["clustering_seed"] == seed]
    importances = data["importances_bootstrap_ratio"]

    selective_electrodes = pd.read_csv(
        f"{ROOT}/Matrix/Matrix_10Cluster/{patient_id}clustering_selective_electrodes_df_"
        f"timeBefore500_timeAfter0_detrendTrue_normalizeFalse_envelope_word2vec_"
        f"nbrclusters10_pvalue0.05.csv"
    )
    selective_electrodes_seed = selective_electrodes[selective_electrodes["seed"] == seed].copy()

    for i in range(50):
        feature_importance = importances.iloc[i]
        mean_importance = np.mean(feature_importance, axis=1)
        selective_electrodes_seed[f"times{i}"] = mean_importance

    max_importance = selective_electrodes_seed.iloc[:, 7:].max(axis=1)
    selective_electrodes_seed = selective_electrodes_seed.drop(selective_electrodes_seed.columns[7:], axis=1)
    selective_electrodes_seed["mean_importance_rate"] = max_importance
    return selective_electrodes_seed


def plot_frequency_band_importance() -> None:
    """Plot relative channel importance across frequency bands."""
    patient_ids = [f"PT{i:02d}" for i in range(1, 15)]

    combined_data = pd.concat(
        [
            _read_feature_importance_file(pid, seed)
            for pid in patient_ids
            for seed in range(20)
        ],
        axis=0,
    ).reset_index(drop=True)

    grouped_mean = combined_data.groupby(["electrode_name", "freqband"], as_index=False)[
        "mean_importance_rate"
    ].mean()

    parcellation_df = pd.read_csv(ROOT / "Parcellation" / "parcellation.csv")
    useful_channels = parcellation_df[parcellation_df["janca_useful"] == 1].copy()
    useful_channels["electrode_name"] = useful_channels["pid"] + "_" + useful_channels["electrode"]
    merged_data = pd.merge(grouped_mean, useful_channels, on="electrode_name", how="left")

    band_dataframes = [merged_data[merged_data["freqband"] == band] for band in range(6)]
    band_means = [df["mean_importance_rate"].mean() for df in band_dataframes]
    band_stds = [df["mean_importance_rate"].sem() for df in band_dataframes]

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    band_offsets = [6, 6.6, 7.2, 7.8, 8.4, 9]
    band_labels = ["θ", "α", "β", "low-γ", "mid-γ", "high-γ"]
    participant_ids = merged_data.pid.unique()

    # Boxplot per frequency band (Fig. 3D, bottom)
    plt.figure(figsize=(10, 5))
    box_width = 0.3
    box_data = [df.groupby("pid")["mean_importance_rate"].mean() * 100 for df in band_dataframes]
    box = plt.boxplot(
        box_data,
        positions=band_offsets,
        widths=box_width,
        showfliers=False,
        patch_artist=True,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="#E6B72E", alpha=0.8)
    for median in box["medians"]:
        median.set(color="white", alpha=0)

    for i, band in enumerate(range(6)):
        x = np.linspace(band_offsets[i] - 0.1, band_offsets[i] + 0.1, 14)
        for pid in participant_ids:
            value = (
                band_dataframes[i][band_dataframes[i]["pid"] == pid]["mean_importance_rate"].mean() * 100
            )
            plt.plot(x[participant_ids.tolist().index(pid)], value, ".", color="#E6B72E", label="_nolegend_")

    plt.xlim([5.6, 9.4])
    plt.yticks([0, 5, 10, 15])
    plt.ylabel("Relative channel importance")
    plt.xticks(band_offsets, band_labels, fontsize=9, rotation=45, ha="right")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(output_dir / "E_Relative_channel_importance_mean_6Band_BoxPlot.pdf", dpi=300)
    plt.close("all")  # reset figure state (headless backends do not clear on show)


def plot_three_area_control_decoding(
    decoding_statistics: pd.DataFrame,
    data_combined: pd.DataFrame,
    bootstrap_test: bool = True,
) -> None:
    """Plot decoding accuracy when using only three pre-defined brain areas.

    Args:
        decoding_statistics: Aggregated decoding statistics DataFrame.
        data_combined: Per-bootstrap decoding results DataFrame.
        bootstrap_test: Whether bootstrap-based accuracies are used.
    """
    _ = bootstrap_test  # retained for API consistency (only the archived extra panels use it)

    participant_ids = decoding_statistics.participant_id.unique()

    plt.rcParams["font.size"] = "15"
    mpl.rcParams["pdf.fonttype"] = 42
    rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

    data_combined["accuracy_bootstrap"] = pd.to_numeric(data_combined["accuracy_bootstrap"])
    data_combined["accuracy_bootstrap_shuffle"] = pd.to_numeric(data_combined["accuracy_bootstrap_shuffle"])

    area3_mean = data_combined.groupby("participant_id")["accuracy_bootstrap"].mean().values * 100
    shuffle_mean_area3 = (
        data_combined.groupby("participant_id")["accuracy_bootstrap_shuffle"].mean().values * 100
    )

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-participant bars
    plt.figure(figsize=(14, 5))
    plt.bar(
        np.arange(1, len(participant_ids) + 1) - 0.2,
        area3_mean,
        label="Actual from 3 areas",
        width=0.4,
        color="#13679e",
        alpha=0.5,
    )
    for pid in participant_ids:
        per_seed = data_combined[data_combined["participant_id"] == pid].groupby("clustering_seed")[
            "accuracy_bootstrap"
        ].mean() * 100
        x = np.linspace(
            1 + participant_ids.tolist().index(pid) - 0.2 - 0.15,
            1 + participant_ids.tolist().index(pid) - 0.2 + 0.15,
            20,
        )
        plt.scatter(x, per_seed, s=8, color="#13679e", label="_nolegend_")

    plt.bar(
        np.arange(1, len(participant_ids) + 1) + 0.2,
        shuffle_mean_area3,
        label="Control",
        width=0.4,
        color="tab:gray",
        alpha=0.5,
    )
    for pid in participant_ids:
        per_seed = data_combined[data_combined["participant_id"] == pid].groupby("clustering_seed")[
            "accuracy_bootstrap_shuffle"
        ].mean() * 100
        x = np.linspace(
            1 + participant_ids.tolist().index(pid) + 0.2 - 0.15,
            1 + participant_ids.tolist().index(pid) + 0.2 + 0.15,
            20,
        )
        plt.scatter(x, per_seed, s=8, color="tab:gray", label="_nolegend_")

    plt.ylim([0, 40])
    plt.yticks([0, 10, 20, 30, 40])
    plt.xlim([0, 15])
    generic_pids = [f"ID0{i}" if i < 10 else f"ID{i}" for i in range(1, len(participant_ids) + 1)]
    plt.xticks(np.arange(1, len(participant_ids) + 1), generic_pids)
    plt.hlines(
        10,
        0,
        len(participant_ids) + 1 + 0.1,
        colors="black",
        linestyles="--",
        label="Chance",
        alpha=0.3,
    )
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Participant")
    plt.ylabel("Decoding accuracy (%)")
    # Explicit handle order: legend entries appear as Actual from 3 areas,
    # Control, Chance (matplotlib would list the chance Line2D first).
    legend_handles = [
        Patch(facecolor="#13679e", label="Actual from 3 areas"),
        Patch(facecolor="tab:gray", label="Control"),
        plt.Line2D([0], [0], color="black", linestyle="--", alpha=0.3, label="Chance"),
    ]
    plt.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=12)
    plt.savefig(output_dir / "D_result_3area.pdf", dpi=300)
    plt.close("all")  # reset figure state (headless backends do not clear on show)


if __name__ == "__main__":
    parcellation_params = absolutize_config_paths({"data_path": "../Parcellation/"})
    parcellation_df, feature_count_df, _, _ = get_group_stats(parcellation_params)

    config = {
        "selectivity_params": absolutize_config_paths({"data_path": "../Matrix/Matrix_10Cluster/"}),
        "condition_params": {
            "feature_name": "envelope",
            "matrix_type": "selective",
            "time_before": 500,
            "time_after": 0,
            "timeseries": False,
            "detrend": True,
            "normalize": False,
            "n_cluster": 10,
            "p_value": 0.05,
            "pretrained_embedding_model": "word2vec",
        },
    }
    _, selective_stats, _, _, _ = get_selective_stats(
        config["selectivity_params"],
        config["condition_params"],
        feature_count_df,
        parcellation_df,
    )
    plot_selective_percentage_summary(feature_count_df, selective_stats)

    plot_selection_percentage_by_band()
    plot_frequency_band_importance()

    configs_folder = ROOT / "Code" / "configs" / "Control_3Area"
    with open(configs_folder / "stats_config.json", "r") as file:
        config = json.load(file)
    statistics = DecodingStatistics(**absolutize_config_paths(config["stats_params"]))

    control_csv = ROOT / "Statistics" / "Stats_3Area" / "decoding_statistics_3Area_control.csv"
    control_pkl = ROOT / "Statistics" / "Stats_3Area" / "decoding_combined_3Area_control.pkl"

    if control_csv.exists() and control_pkl.exists():
        decoding_statistics = pd.read_csv(control_csv)
        decoding_combined = pd.read_pickle(control_pkl)
    else:
        decoding_combined = statistics.get_decoding_statistics3()
        decoding_combined.to_pickle(control_pkl)
        decoding_statistics = statistics.get_decoding_statistics2()
        decoding_statistics.to_csv(control_csv, index=False)

    plot_three_area_control_decoding(decoding_statistics, decoding_combined)
