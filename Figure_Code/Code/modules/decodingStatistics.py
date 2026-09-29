"""Aggregate per-participant decoding result pickles (used by fig2.py / fig3.py)."""

import os
from pathlib import Path

import pandas as pd


class DecodingStatistics():
    """Aggregate the decoding result pickles stored per participant."""

    def __init__(self, decoding_path: Path, save_path: Path, figure_path: Path) -> None:
        self.decoding_path = decoding_path
        self.save_path = save_path
        self.figure_path = figure_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.figure_path):
            os.makedirs(self.figure_path)

    def get_decoding_statistics2(self) -> pd.DataFrame:
        """Per-condition summary (mean / std / median / min / max) of the bootstrap decoding accuracies."""
        decoding_accuracies = pd.DataFrame(
            columns=['participant_id', 'embedding_model', 'matrix_type', 'n_cluster', 'num_draws', 'reg_strength',
                     'time_before', 'time_after',
                     'accuracy_bootstrap_mean', 'accuracy_bootstrap_shuffle_mean',
                     'accuracy_bootstrap_std', 'accuracy_bootstrap_shuffle_std',
                     'accuracy_bootstrap_median', 'accuracy_bootstrap_shuffle_median',
                     'accuracy_bootstrap_min', 'accuracy_bootstrap_shuffle_min',
                     'accuracy_bootstrap_max', 'accuracy_bootstrap_shuffle_max',
                     'accuracy_bootstrapTrain_mean', 'accuracy_bootstrapTrain_shuffle_mean',
                     'accuracy_bootstrapTrain_std', 'accuracy_bootstrapTrain_shuffle_std',
                     'accuracy_bootstrapTrain_median', 'accuracy_bootstrapTrain_shuffle_median',
                     'accuracy_bootstrapTrain_min', 'accuracy_bootstrapTrain_shuffle_min',
                     'accuracy_bootstrapTrain_max', 'accuracy_bootstrapTrain_shuffle_max'])

        for decoding_file in os.listdir(self.decoding_path):
            if os.path.isfile(os.path.join(self.decoding_path, decoding_file)):
                decoding_df = pd.read_pickle(os.path.join(self.decoding_path, decoding_file))
                decoding_grouped = decoding_df.groupby(['participant_id', 'embedding_model', 'matrix_type', 'n_cluster',
                                                        'num_draws', 'reg_strength', 'time_before', 'time_after'])

                for group in decoding_grouped:
                    accuracies_bootstrap = group[1]['accuracy_bootstrap']
                    accuracies_bootstrap_shuffle = group[1]['accuracy_bootstrap_shuffle']
                    accuracies_bootstrapTrain = group[1]['accuracy_bootstrapTrain']
                    accuracies_bootstrapTrain_shuffle = group[1]['accuracy_bootstrapTrain_shuffle']
                    decoding_accuracies = pd.concat([decoding_accuracies, pd.DataFrame({'participant_id': [group[0][0]],
                                                                                        'embedding_model': group[0][1],
                                                                                        'matrix_type': group[0][2],
                                                                                        'n_cluster': group[0][3],
                                                                                        'num_draws': group[0][4],
                                                                                        'reg_strength': group[0][5],
                                                                                        'time_before': group[0][6],
                                                                                        'time_after': group[0][7],
                                                                                        'accuracy_bootstrap_mean': accuracies_bootstrap.mean(),
                                                                                        'accuracy_bootstrap_shuffle_mean': accuracies_bootstrap_shuffle.mean(),
                                                                                        'accuracy_bootstrap_std': accuracies_bootstrap.std(),
                                                                                        'accuracy_bootstrap_shuffle_std': accuracies_bootstrap_shuffle.std(),
                                                                                        'accuracy_bootstrap_median': accuracies_bootstrap.median(),
                                                                                        'accuracy_bootstrap_shuffle_median': accuracies_bootstrap_shuffle.median(),
                                                                                        'accuracy_bootstrap_min': accuracies_bootstrap.min(),
                                                                                        'accuracy_bootstrap_max': accuracies_bootstrap.max(),
                                                                                        'accuracy_bootstrap_shuffle_min': accuracies_bootstrap_shuffle.min(),
                                                                                        'accuracy_bootstrap_shuffle_max': accuracies_bootstrap_shuffle.max(),
                                                                                        'accuracy_bootstrapTrain_mean': accuracies_bootstrapTrain.mean(),
                                                                                        'accuracy_bootstrapTrain_shuffle_mean': accuracies_bootstrapTrain_shuffle.mean(),
                                                                                        'accuracy_bootstrapTrain_std': accuracies_bootstrapTrain.std(),
                                                                                        'accuracy_bootstrapTrain_shuffle_std': accuracies_bootstrapTrain_shuffle.std(),
                                                                                        'accuracy_bootstrapTrain_median': accuracies_bootstrapTrain.median(),
                                                                                        'accuracy_bootstrapTrain_shuffle_median': accuracies_bootstrapTrain_shuffle.median(),
                                                                                        'accuracy_bootstrapTrain_min': accuracies_bootstrapTrain.min(),
                                                                                        'accuracy_bootstrapTrain_max': accuracies_bootstrapTrain.max(),
                                                                                        'accuracy_bootstrapTrain_shuffle_min': accuracies_bootstrapTrain_shuffle.min(),
                                                                                        'accuracy_bootstrapTrain_shuffle_max': accuracies_bootstrapTrain_shuffle.max()}
                                                                                       )])

        return decoding_accuracies

    def get_decoding_statistics3(self) -> pd.DataFrame:
        """Concatenate every per-participant result pickle into one DataFrame."""
        decoding_combined = []
        decoding_files = [f for f in os.listdir(self.decoding_path) if os.path.isfile(os.path.join(self.decoding_path, f))]

        for decoding_file in decoding_files:
            decoding_df = pd.read_pickle(os.path.join(self.decoding_path, decoding_file))
            decoding_df = decoding_df[['participant_id', 'embedding_model', 'matrix_type', 'n_cluster', 'num_draws', 'reg_strength', 'time_before', 'time_after', 'clustering_seed', 'bootstrap_seed','accuracy', 'accuracy_shuffle', 'accuracy_bootstrap', 'accuracy_bootstrap_shuffle', 'accuracy_bootstrapTrain', 'accuracy_bootstrapTrain_shuffle', 'accuracy_bootstrap_control']]
            decoding_combined.append(decoding_df)

        combined_df = pd.concat(decoding_combined, ignore_index=True)
        return combined_df
