"""Plotting script for Figure 1.

Panels:
    B: All recorded channels across participants (brain map).
    C: Word counts per participant.

Outputs are written to ``Figure/fig1/``. All data paths are
resolved relative to this file's location, so the script can be run from any
working directory.

Requires nilearn installed:
    python fig1.py
"""

import warnings
from pathlib import Path
from typing import Union

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import rc
from nilearn import plotting

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]  # repository root
OUTPUT_DIR = ROOT / "Figure" / "fig1"

AREA_COLORS = [
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#8EAA11",  # yellow
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#9d03fc",  # violet
    "#7f7f7f",  # gray
]

AREA_ORDER = [
    "middletemporal",
    "hippocampus",
    "rostralmiddlefrontal",
    "superiortemporal",
    "amygdala",
    "caudate",
    "putamen",
    "thalamus-proper",
    "inferiortemporal",
    "precentral",
    "parstriangularis",
    "lateralorbitofrontal",
    "caudalmiddlefrontal",
    "other",
]

# Colorbar read order (reverse of the AREA_ORDER brain areas, "other" last).
# Building area_to_color from this order keeps the colorbar swatches and the
# brain-dot colors consistent.
DISPLAY_ORDER = list(reversed(AREA_ORDER[:-1])) + ["other"]

# Spelled-out colorbar labels; all other names are used as-is.
AREA_DISPLAY_NAMES = {
    "caudalmiddlefrontal": "caudal middle frontal",
    "inferiortemporal": "inferior temporal",
    "superiortemporal": "superior temporal",
    "rostralmiddlefrontal": "rostral middle frontal",
    "middletemporal": "middle temporal",
    "thalamus-proper": "thalamus",
    "lateralorbitofrontal": "lateralorbito frontal",
}

def plot_all_channels() -> None:
    """Plot all useful channels across participants on a brain surface."""
    data = pd.read_csv(ROOT / "Parcellation" / "parcellation.csv")
    data = data[data["janca_useful"] == 1].reset_index(drop=True)

    positions = np.load(ROOT / "Parcellation" / "pos.npy")
    positions = pd.DataFrame(positions, columns=["x", "y", "z"])
    channel_info = pd.concat([data, positions], axis=1)
    channel_info = channel_info.rename(columns={"pid": "participant_id"})

    qualitative_cmap = mcolors.ListedColormap(AREA_COLORS, name="15colors")
    color_stamps = np.linspace(0, 1, 15)[:-1] + (0.5 / 13)
    area_to_color = {DISPLAY_ORDER[i]: color_stamps[i] for i in range(len(DISPLAY_ORDER))}

    channel_info["node_value"] = channel_info["area"].map(area_to_color)
    channel_info["node_value"] = channel_info["node_value"].fillna(0.9645)
    channel_info["node_size"] = 10

    plt.rcParams["figure.figsize"] = [15, 15]
    plt.rcParams["font.size"] = "13"
    mpl.rcParams["pdf.fonttype"] = 42
    rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

    fig = plt.figure(figsize=(18, 11))
    display = plotting.plot_markers(
        node_values=channel_info["node_value"],
        node_coords=channel_info[["x", "y", "z"]],
        node_cmap=qualitative_cmap,
        node_size=channel_info["node_size"],
        node_vmin=0,
        node_vmax=1,
        colorbar=False,
        figure=fig,
        alpha=0.7,
    )

    # Layout: coronal (y), sagittal (x), axial (z) left to right in the
    # upper part of the canvas. plot_markers' "ortho" mode always creates
    # x/y/z slices in that order at x0 = 0 / 0.3 / 0.6, so reposition the
    # axes manually (this nilearn version rejects tuple display modes).
    view_slots = {"y": 0.0, "x": 0.3, "z": 0.6}
    for view, slot_x in view_slots.items():
        display.axes[view].ax.set_position([slot_x, 0.3, 0.3, 0.66])

    # Horizontal colorbar strip below the three brains, with one vertical
    # label per swatch.
    strip_ax = fig.add_axes([0.035, 0.22, 0.93, 0.015])
    strip = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=1), cmap=qualitative_cmap),
        cax=strip_ax,
        orientation="horizontal",
    )
    strip.set_ticks(list(area_to_color.values()))
    strip.ax.set_xticklabels(
        [AREA_DISPLAY_NAMES.get(area, area) for area in area_to_color.keys()],
        rotation=90,
        fontsize=13,
    )
    strip.outline.set_visible(False)
    strip.ax.tick_params(axis="x", length=0, pad=2)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "Total_channels.pdf", dpi=300)
    plt.close("all")  # reset figure state (headless backends do not clear on show)


def plot_conversation_word_counts(
    data_path: Union[str, Path] = ROOT / "Transcription" / "Conversation",
    embeddings_path: Union[str, Path] = ROOT / "Transcription" / "Embedding" / "fasttext",
) -> None:
    """Plot total and unique word counts per participant.

    Derived entirely from the anonymized release (no ``word`` column needed):
    word_count / total_time / speaking_time come from the conversation CSV
    (its rows are already the post-exclusion sid=0 token set); unique_words is
    the number of unique rows in the row-aligned fastText embedding matrix —
    fastText's subword model embeds every token (0 NaN rows for all 14
    participants) and distinct words always map to distinct vectors, so unique
    rows == unique words (verified against the raw transcripts, 14/14 exact).
    The word2vec matrices cannot be used here: their OOV rows are all-NaN and
    collapse into one row.
    """
    data_path = Path(data_path)
    embeddings_path = Path(embeddings_path)
    conversation_stats = pd.DataFrame(
        columns=["pid", "total_time", "speaking_time", "word_count", "unique_words"]
    )

    for file in sorted(data_path.glob("*.csv")):
        conversation_df = pd.read_csv(file)
        word_count = len(conversation_df)
        total_time = conversation_df["end_time"].max() - conversation_df["start_time"].min()
        speaking_time = (conversation_df["end_time"] - conversation_df["start_time"]).sum()

        participant_id = file.stem
        embeddings = np.load(embeddings_path / f"{participant_id}.npy")
        kept = embeddings[~np.isnan(embeddings).any(axis=1)]
        unique_words = len(np.unique(kept, axis=0))

        conversation_stats = pd.concat(
            [
                conversation_stats,
                pd.DataFrame(
                    {
                        "pid": [participant_id],
                        "total_time": [total_time],
                        "speaking_time": [speaking_time],
                        "word_count": [word_count],
                        "unique_words": [unique_words],
                    }
                ),
            ]
        )

    data = conversation_stats.reset_index(drop=True)
    raw_word_count = data["word_count"].astype(int)  # annotations show raw counts
    data["word_count"] = data["word_count"] / 1000.0  # axis: Counts (x 1000)
    data["unique_words"] = data["unique_words"] / 1000.0
    data["pid_generic"] = [
        f"ID0{i}" if i < 10 else f"ID{i}" for i in range(1, len(data) + 1)
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(len(data["pid_generic"]))
    bar_total = ax.bar(
        index, data["word_count"], label="All words", color="#1f77b4"
    )
    ax.bar(index, data["unique_words"], label="Unique words", color="#ff7f0e")

    for bar, raw_count in zip(bar_total, raw_word_count):
        ax.annotate(
            f"{raw_count:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Participants")
    ax.set_ylabel("Counts (x 1000)")
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xticks(index)
    ax.set_xticklabels(data["pid_generic"])
    ax.tick_params(axis="both", which="major")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=12
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "conversation_info.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")  # reset figure state (headless backends do not clear on show)


if __name__ == "__main__":
    plot_all_channels()
    plot_conversation_word_counts()
