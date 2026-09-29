"""Feature-matrix builders for the SSDD semantic-dimension control (KMeans clustering variant)."""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List
from collections import OrderedDict
from tqdm.auto import tqdm
from scipy.signal import detrend
from scipy.stats import zscore, ttest_ind, ranksums
from sklearn.cluster import KMeans


class FullMatrix():
    '''Full neural-feature matrix (no channel selection).'''

    def __init__(self, transcription_df: pd.DataFrame, time_before: int, time_after: int,
                 n_freqbands: int, electrode_count: int, detrend: bool, normalize: bool, data_path: str,
                 participant_id: str, feature_name: str, timeseries: bool, logger) -> None:
        self.transcription_df = transcription_df
        self.time_before, self.time_after = time_before, time_after
        self.detrend, self.normalize = detrend, normalize
        self.data_path = data_path
        self.participant_id = participant_id
        self.feature_name = feature_name
        self.n_freqbands = n_freqbands
        self.electrode_count = electrode_count
        self.timeseries = timeseries
        self.logger = logger

    def build_matrix(self) -> np.ndarray[np.ndarray]:
        '''Build the word x (electrode x freqband) feature matrix from neural data.'''
        tic = time.time()
        X = 999 * np.ones((len(self.transcription_df), self.electrode_count * self.n_freqbands))
        counter = 0
        for file in tqdm(os.listdir(self.data_path)):
            if file.startswith(self.participant_id) and self.feature_name in file:
                neural_data = np.load(f'{self.data_path}{file}')
                if self.detrend:
                    neural_data = detrend(neural_data, axis=1)
                if self.normalize:
                    neural_data = zscore(neural_data, axis=1)
                for idx, row in self.transcription_df.iterrows():
                    if row.start_time_ms < neural_data.shape[
                        0]:  # do not access data that does not exist
                        X[idx, self.n_freqbands * counter:self.n_freqbands * (counter + 1)] = np.mean(neural_data[int(
                            row['start_time_ms'] - self.time_before):int(row['start_time_ms']) + self.time_after, :],
                                                                                                      axis=0)
                    else:
                        raise ValueError('row.start_time_ms > neural_data.shape[0]')
                counter += 1
        toc = time.time()
        self.logger.info(f'For participant {self.participant_id}: Time to build full matrix: {toc - tic} seconds')
        return X


class SelectiveMatrix():
    '''Neural-feature matrix with channel selection.'''

    def __init__(self, transcription_df: pd.DataFrame, n_cluster: int, time_before: int, time_after: int,
                 n_freqbands: int, electrode_count: int, detrend: bool, normalize: bool, participant_id: str,
                 feature_name: str, p_value: float, seed_i: int, data_path: str, timeseries: bool) -> None:
        self.transcription_df = transcription_df
        self.n_cluster = n_cluster
        self.time_before, self.time_after = time_before, time_after
        self.participant_id = participant_id
        self.n_freqbands, self.electrode_count = n_freqbands, electrode_count
        self.detrend, self.normalize = detrend, normalize
        self.feature_name = feature_name
        self.p_value = p_value
        self.seed_i = seed_i
        self.data_path = data_path
        self.timeseries = timeseries

    def cluster_words(self, cluster_unique_words: bool = True) -> None:
        '''Cluster words into n_cluster semantic groups with KMeans (this control
        uses KMeans, not the main experiment's spectral clustering).'''
        if cluster_unique_words:
            transcription_unique_words = self.transcription_df.drop_duplicates(subset=['word'])
            transcription_unique_words['kmeans_clusters'] = KMeans(n_clusters=self.n_cluster,
                                                                  random_state=self.seed_i).fit_predict(
                np.array(transcription_unique_words['embedding'].tolist()))
            self.transcription_df['kmeans_clusters'] = self.transcription_df['word'].map(
                transcription_unique_words.set_index('word')['kmeans_clusters'])
        else:
            self.transcription_df['kmeans_clusters'] = KMeans(n_clusters=self.n_cluster,
                                                              random_state=self.seed_i).fit_predict(
                np.array(self.transcription_df['embedding'].tolist()))

    def single_neuronal_selectivity(self, probe_np, neuron_data, ttest_or_perm='ranksum') -> np.ndarray:
        '''Channel selectivity per label: one cluster vs. all others (ranksum or ttest).'''
        # neuron_data: n_words x n_electrodes; probe_np: n_words x 1
        # labels are numeric; 999 marks excluded observations
        output_p = np.zeros([neuron_data.shape[1], probe_np.shape[1], self.n_cluster])
        output_p[:] = np.nan
        for ind in range(probe_np.shape[1]):
            probe_value = probe_np[:, ind]
            for neuron_ind in range(neuron_data.shape[1]):
                neuron_sub = neuron_data[:, neuron_ind]
                neuron_non = np.isnan(neuron_sub) == 0
                neuron_sub = neuron_sub[neuron_non]
                probe_sub = probe_value[neuron_non]
                probe_none = probe_sub < 900
                neuron_sub = neuron_sub[probe_none]
                probe_sub = probe_sub[probe_none]
                unique_probe = np.unique(probe_sub)
                for probe_ind1, value1 in enumerate(unique_probe):
                    valuein = neuron_sub[probe_sub == value1]
                    valueout = neuron_sub[probe_sub != value1]

                    if (valuein.shape[0] > 0) and (valueout.shape[0] > 0):
                        if (valuein.sum() / valuein.shape[0] > 0.01) * (valueout.sum() / valueout.shape[0] > 0.01):
                            if ttest_or_perm == 't':
                                output_p[neuron_ind, ind, probe_ind1] = ttest_ind(valuein, valueout)[
                                    1]
                            elif ttest_or_perm == 'ranksum':
                                output_p[neuron_ind, ind, probe_ind1] = ranksums(valuein, valueout)[1]
                            else:
                                raise ValueError('ttest_or_perm should be t or ranksum')
                        else:
                            logging.info(
                                'Failing in SelectiveMatrix.single_neuronal_selectivity: valuein.sum()/valuein.shape[0] or valueout.sum()/valueout.shape[0] is 0')
                    else:
                        logging.info(
                            'Failing in SelectiveMatrix.single_neuronal_selectivity: valuein.shape[0] or valueout.shape[0] is 0')
        return output_p

    def extract_selective_electrodes(self) -> Tuple[pd.DataFrame, List[List[str]], pd.Series]:
        '''Select electrodes with significant label selectivity.'''
        X = [999 * np.ones((len(self.transcription_df), self.electrode_count)) for _ in range(self.n_freqbands)]
        Y = [None] * self.n_freqbands
        sig_idxs = [None] * self.n_freqbands
        p_values = [None] * self.n_freqbands
        selective_electrodes = pd.DataFrame(columns=['electrode_name', 'freqband', 'p_value'])
        selective_electrodes_calculation = [None] * self.n_freqbands

        count_el = -1

        # For silent control, select electrodes that are selective for the speaking condition
        if 'silentControl' in self.data_path:
            data_path_selective = self.data_path.replace('_silentControl', '')
        else:
            data_path_selective = self.data_path

        for file in tqdm(os.listdir(data_path_selective)):
            if file.startswith(self.participant_id) and self.feature_name in file:
                count_el += 1
                neural_data = np.load(f'{data_path_selective}{file}')
                for idx, row in self.transcription_df.iterrows():
                    for i in range(self.n_freqbands):
                        X[i][idx, count_el] = np.mean(neural_data[int(row['start_time_ms'] - self.time_before):int(
                            row['start_time_ms'] + self.time_after), i])

        probe_np = self.transcription_df['kmeans_clusters']
        for i in range(self.n_freqbands):
            X[i] = X[i][~np.isnan(X[i]).any(axis=1)]
            if len(X[i]) > 0:
                probe_np_resized = np.expand_dims(probe_np[:len(X[i])], axis=1)
                Y[i] = self.single_neuronal_selectivity(probe_np_resized, X[i])[:, 0, :]
                # Bonferroni correction across clusters
                condition_fulfilled = np.where(Y[i] < (self.p_value / len(np.unique(probe_np))))
                sig_idxs[i] = np.unique(condition_fulfilled[0])
                p_values_rows = Y[i][sig_idxs[i], :]
                # Min p-value per electrode
                p_values[i] = np.min(p_values_rows, axis=1)
            else:
                raise ValueError('X[i] is empty')

        electrode_names = extract_electrode_names(data_path_selective, self.participant_id, self.feature_name)

        total_electrodes = 0
        for i in range(self.n_freqbands):
            el_names = extract_selective_electrode_names(electrode_names, sig_idxs[i])
            selective_electrodes = pd.concat([selective_electrodes,
                                              pd.DataFrame({'electrode_name': el_names,
                                                            'freqband': [i] * len(
                                                                extract_selective_electrode_names(electrode_names, sig_idxs[i])),
                                                            'p_value': p_values[i]})])
            selective_electrodes_calculation[i] = extract_selective_electrode_names(electrode_names, sig_idxs[i])
            total_electrodes += len(el_names)
        self.total_electrodes = total_electrodes
        logging.info(
            f'Number of selective features: {total_electrodes}, expected at 5% chance level: {int(0.05 * self.electrode_count * self.n_freqbands)}')
        return selective_electrodes, selective_electrodes_calculation, probe_np

    def averaged_matrix(self, selective_electrodes) -> np.ndarray[np.ndarray]:
        '''Averaged feature matrix for time_after > 0.'''
        X = 999 * np.ones((len(self.transcription_df), self.total_electrodes))
        freq_counter = -1
        count_el = -1
        for selective_electrode in selective_electrodes:
            freq_counter += 1
            for file in tqdm(os.listdir(self.data_path)):
                if file.startswith(self.participant_id) and self.feature_name in file:
                    if file.replace('_' + self.feature_name + '.npy', '') in selective_electrode:
                        count_el += 1
                        neural_data = np.load(f'{self.data_path}{file}')[:, freq_counter]
                        if self.detrend:
                            neural_data = detrend(neural_data)
                        if self.normalize:
                            neural_data = zscore(neural_data)
                        for idx, row in self.transcription_df.iterrows():
                            if row.start_time_ms < neural_data.shape[0]:
                                X[idx, count_el] = np.mean(neural_data[int(row['start_time_ms'] - self.time_before):int(
                                    row['start_time_ms'] - self.time_after)])
                            else:
                                raise ValueError('row.start_time_ms > neural_data.shape[0]')
        return X

    def averaged_matrix_negative_time_after(self, selective_electrodes) -> np.ndarray[np.ndarray]:
        '''Averaged feature matrix for time_after < 0.'''
        X = 999 * np.ones((len(self.transcription_df), self.total_electrodes))
        freq_counter = -1
        count_el = -1
        for selective_electrode in selective_electrodes:
            freq_counter += 1
            for file in tqdm(os.listdir(self.data_path)):
                if file.startswith(self.participant_id) and self.feature_name in file:
                    if file.replace('_' + self.feature_name + '.npy', '') in selective_electrode:
                        count_el += 1
                        neural_data = np.load(f'{self.data_path}{file}')[:, freq_counter]
                        if self.detrend:
                            neural_data = detrend(neural_data)
                        if self.normalize:
                            neural_data = zscore(neural_data)
                        for idx, row in self.transcription_df.iterrows():
                            if row.start_time_ms < neural_data.shape[0]:
                                X[idx, count_el] = np.mean(neural_data[int(row['start_time_ms'] - self.time_before):int(
                                    row['start_time_ms'] + self.time_after)])
                            else:
                                raise ValueError('row.start_time_ms > neural_data.shape[0]')
        return X

    def timeseries_matrix(self, selective_electrodes, window_size: int = 50, stride: int = 25) -> np.ndarray[
        np.ndarray[np.ndarray]]:
        time_span = len(range(-np.abs(self.time_before), self.time_after - window_size + 1, stride))
        X = 999 * np.ones((len(self.transcription_df), self.total_electrodes, time_span))
        freq_counter = -1
        count_el = -1
        for selective_electrode in selective_electrodes:
            freq_counter += 1
            for file in tqdm(os.listdir(self.data_path)):
                if file.startswith(self.participant_id) and self.feature_name in file:
                    if file.replace(self.feature_name, '') in selective_electrode:
                        count_el += 1
                        neural_data = np.load(f'{self.data_path}{file}')[:, freq_counter]
                        if self.detrend:
                            neural_data = detrend(neural_data)
                        if self.normalize:
                            neural_data = zscore(neural_data)
                        for idx, row in self.transcription_df.iterrows():
                            if row.start_time_ms < neural_data.shape[0]:
                                time_counter = 0
                                for i in range(-np.abs(self.time_before), self.time_after - window_size + 1, stride):
                                    X[idx, count_el, time_counter] = np.mean(neural_data[
                                                                             int(row['start_time_ms'] + i):int(row[
                                                                                                                   'start_time_ms'] + i + window_size),
                                                                             :], axis=0)
                                    time_counter += 1
                            else:
                                raise ValueError('row.start_time_ms > neural_data.shape[0]')
        return X

    def build_matrix(self, selective_electrodes_df: pd.DataFrame) -> Tuple[np.ndarray[np.ndarray],
    np.ndarray, pd.DataFrame]:
        '''Decoding matrix: KMeans labels -> channel selection -> averaged features.'''
        logging.info(f'Clustering data with seed {self.seed_i}...')
        self.cluster_words()

        selective_electrodes, selective_electrodes_calculation, y = self.extract_selective_electrodes()

        selective_electrodes['participant_id'] = self.participant_id
        selective_electrodes['clustering'] = 'kmeans_clusters'
        selective_electrodes['n_cluster'] = self.n_cluster
        selective_electrodes['seed'] = self.seed_i

        selective_electrodes_df = pd.concat([selective_electrodes_df, selective_electrodes])

        if self.timeseries:
            X = self.timeseries_matrix(selective_electrodes_calculation)
        else:
            if self.time_after < 0:
                X = self.averaged_matrix_negative_time_after(selective_electrodes_calculation)
            else:
                X = self.averaged_matrix(selective_electrodes_calculation)

        return X, y, selective_electrodes_df

# ===== Helper functions =====
def extract_electrode_names(neural_data_path, participant_id, end_name):
    electrode_names = []
    for file in os.listdir(neural_data_path):
        if file.startswith(participant_id) and end_name in file:
            el_name = file.replace('_' + end_name + '.npy', '')
            electrode_names.append(f'{el_name}')
    return electrode_names



def extract_selective_electrode_names(names, sig_idxs):
    electrodes = [names[idx] for idx in sig_idxs]
    # Remove duplicates while preserving order
    return list(OrderedDict.fromkeys(electrodes))