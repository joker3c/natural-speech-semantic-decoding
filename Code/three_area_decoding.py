"""Three-brain-area decoding.

Semantic decoding restricted to channels in middle temporal cortex,
precentral gyrus and pars triangularis (from parcellation.csv). This script
does NOT generate matrices itself (GENERATE_MATRICES = False): it consumes
the X/y matrices and PTxxclustering_selective_electrodes_df_*.csv summaries
written by semantic_decoding.py (configs/SemanticDecode/, word2vec,
10 clusters, seeds 0-19) in ../Matrix/Matrix_10Cluster/, plus
../Parcellation/parcellation.csv. Run semantic_decoding.py FIRST.
Configs: configs/Control_3Area/ + configs_group/config.json.
"""
import os
import json
import re
import time
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from modules.participant_data import ParticipantData
from modules.transcription import Transcription

# Decoding setting
# False: this script never builds matrices; it reads the ones written by
# semantic_decoding.py in ../Matrix/Matrix_10Cluster/ (see module docstring).
GENERATE_MATRICES = False
RUN_DECODING = True

GET_ALL = GENERATE_MATRICES or RUN_DECODING


def run_decoding(config_path: str, logger: logging.Logger, electrode: pd.DataFrame) -> None:
    print('Running decoding.')
    with open(config_path, 'r') as file:
        config = json.load(file)
    print('Config loaded.')
    config["transcription_params"]["participant_id"] = config["participant_params"]["participant_id"]

    if GET_ALL:
        tic = time.time()
        participant_data = ParticipantData(**config["participant_params"])
        toc = time.time()
        print(f'Participant data created in {toc-tic:.2f} seconds.')
        tic = time.time()
        participant_data.transcription = Transcription(**config["transcription_params"])
        toc = time.time()
        print(f'Transcription created in {toc-tic:.2f} seconds.')
    # ===== Generate matrices =====
    if GENERATE_MATRICES:
        participant_data.extract_matrix(save_matrix=GENERATE_MATRICES)
    # ===== Run decoding =====
    if RUN_DECODING:
        participant_data.run_decoding_for_3_area_test(**config["decoding_params"], electrode_info=electrode)
        print('Decoding done.')


def run_all_configs(configs_folder: str, logger: logging.Logger, electrode: pd.DataFrame) -> None:
    print('Running all configs.')
    for config_file in os.listdir(configs_folder):
        if config_file.endswith('.json') and config_file != 'stats_config.json':
            config_path = os.path.join(configs_folder, config_file)
            run_decoding(config_path, logger, electrode)


def get_pid(file_name: str) -> str:
    return re.search(r'(.+)(?=clustering)', file_name).group(0)


def file_name_valid(file_name: str, params: dict) -> bool:
    '''Check if file name contains all params'''
    for param in params.keys():
        if str(params[param]) not in file_name:
            return False
        elif param == 'time_before':
            if not re.search(f'timeBefore{params[param]}', file_name):
                return False
        elif param == 'time_after':
            if not re.search(f'timeAfter{params[param]}', file_name):
                return False
        elif param == 'detrend':
            if not re.search(f'detrend{params[param]}', file_name):
                return False
        elif param == 'normalize':
            if not re.search(f'normalize{params[param]}', file_name):
                return False
        elif param == 'n_cluster':
            if not re.search(f'nbrclusters{params[param]}', file_name):
                return False
        elif param == 'p_value':
            if not re.search(f'pvalue{params[param]}', file_name):
                return False
    return True


def get_selective_stats(config: dict, condition_config: dict, electrode_count_df: pd.DataFrame,
                        parcellation_df: pd.DataFrame) -> tuple:
    # Store all selective features
    selective_features_combined = pd.DataFrame()
    selective_stats = pd.DataFrame()
    # Store location of selective features
    unique_broad_areas = parcellation_df['broad_area'].unique()
    unique_areas = parcellation_df['area'].unique()
    broad_area_counter = Counter(unique_broad_areas)
    area_counter = Counter(unique_areas)
    # Selectivity summaries written by semantic_decoding.py. The matching X/y
    # matrices in the same folder are read back via ParticipantData.load_matrix
    # (modules/participant_data.py) during decoding.
    config["data_path"] = '../Matrix/Matrix_10Cluster/'
    for file in Path(config["data_path"]).glob('*.csv'):
        if file_name_valid(file.name, condition_config):
            print(f'For file {file.name}:')
            pid = get_pid(file.name)
            df = pd.read_csv(file)
            df['pid'] = df['participant_id']
            del df['participant_id']

            selective_features_combined = pd.concat([selective_features_combined, df])

            # Get number of seeds
            n_seeds = len(df['seed'].unique())
            n_freqbands = len(df['freqband'].unique())

            for group_id, group in df.groupby(['pid', 'clustering', 'n_cluster', 'freqband']):
                for sub_group_id, sub_group in group.groupby(['seed']):
                    selective_stats = pd.concat([selective_stats, pd.DataFrame({'pid': pid, 'clustering': [group_id[1]],
                                    'n_cluster': [group_id[2]], 'freqband': [group_id[3]], 'seed': [sub_group_id[0]],
                                    'selective_features_count': [len(sub_group)],
                                    'total_number': [(n_freqbands * electrode_count_df.loc[pid, 'janca_useful'])],
                                    'selective_features_percentage': [len(sub_group)/(n_freqbands*electrode_count_df.loc[pid, 'janca_useful'])]})])

                    for electrode_i in sub_group['electrode_name']:
                        _, electrode_i = electrode_i.split('_', 1)
                        broad_area_i, area_i = parcellation_df.loc[(parcellation_df['electrode'] == electrode_i) &
                                            (parcellation_df['pid'] == pid), ['broad_area', 'area']].values[0]
                        broad_area_counter[broad_area_i] += 1
                        area_counter[area_i] += 1
    return selective_features_combined, selective_stats, broad_area_counter, area_counter, n_seeds


def get_electrode_info(config: dict) -> pd.DataFrame:
    parcellation_df = pd.read_csv(f'{config["data_path"]}/' +
                                  f'parcellation.csv', index_col=0)
    return parcellation_df


def get_group_stats(config: dict) -> tuple:
    parcellation_df = pd.read_csv(f'{config["data_path"]}/' +
                                  f'parcellation.csv', index_col=0)
    # Group by participant
    group_df = parcellation_df.groupby('pid')

    unique_broad_areas = parcellation_df['broad_area'].unique()
    unique_areas = parcellation_df['area'].unique()
    broad_area_combined = pd.DataFrame(columns=np.append('pid', unique_broad_areas))
    area_combined = pd.DataFrame(columns=np.append('pid', unique_areas))

    for pid, group in group_df:
        broad_area_count = group.loc[group['janca_useful'] == True, 'broad_area'].value_counts()
        area_count = group.loc[group['janca_useful'] == True, 'area'].value_counts()

        broad_area_combined = pd.concat([broad_area_combined, pd.DataFrame({'pid': [pid], **broad_area_count})])
        area_combined = pd.concat([area_combined, pd.DataFrame({'pid': [pid], **area_count})])

    considered_electrodes = group_df.agg({'electrode': 'count', 'janca_useful': 'sum'})
    # Print for each pid the number of janca_useful
    for pid in considered_electrodes.index:
        nbr_electrodes = considered_electrodes.loc[pid, "electrode"]
        nbr_useful = considered_electrodes.loc[pid, "janca_useful"]
        print(f'Participant {pid} has {nbr_electrodes} electrodes out of which ' +
              f'{nbr_useful} are considered for the analysis ' +
              f'({6 * nbr_useful} features).')

    return parcellation_df, considered_electrodes, broad_area_combined, area_combined


if __name__ == "__main__":
    print('Running main.')
    with open('configs_group/config.json', 'r') as file:
        config = json.load(file)
    parcellation_df, electrode_count_df, broad_area_combined, area_combined = get_group_stats(
        config["parcellation_params"])
    electrodes_info = get_electrode_info(config["parcellation_params"])
    electrodes_info['name'] = electrodes_info['pid'] + '_' + electrodes_info['electrode']
    selective_features_combined, selective_stats, broad_area_counter, area_counter, n_seeds = get_selective_stats(
        config["selectivity_params"],
        config["condition_params"], electrode_count_df, parcellation_df)
    selected_channel_3area = pd.merge(selective_features_combined, electrodes_info, how='left', left_on='electrode_name', right_on='name')
    logging.basicConfig(filename='speech_decoding_logger.log', level=logging.INFO, filemode='a')
    logger = logging.getLogger(__name__)
    run_all_configs('configs/Control_3Area/', logger, selected_channel_3area)
