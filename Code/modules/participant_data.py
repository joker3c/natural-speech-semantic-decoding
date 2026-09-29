"""Per-participant pipeline: build/load neural feature matrices and run semantic decoding."""

import os
import logging
import warnings

import numpy as np
import pandas as pd

from modules.matrices import SelectiveMatrix
from modules.speech_decoder import SpeechDecoder

warnings.filterwarnings("ignore")

# Output DataFrame schemas for the decoding runs. The column order is part of
# the result-file format — do not reorder.
ACCURACY_COLUMNS = ['participant_id', 'embedding_model', 'matrix_type',
                    'n_cluster', 'num_draws', 'reg_strength', 'time_before', 'time_after',
                    'clustering_seed', 'bootstrap_seed',
                    'y_test', 'y_test_bootstrap',
                    'y_pred', 'y_pred_shuffle', 'y_pred_bootstrap', 'y_pred_bootstrap_shuffle',
                    'y_pred_bootstrapTrain', 'y_pred_bootstrapTrain_shuffle',
                    'accuracy', 'accuracy_shuffle', 'accuracy_bootstrap', 'accuracy_bootstrap_shuffle',
                    'accuracy_bootstrapTrain', 'accuracy_bootstrapTrain_shuffle',
                    'accuracy_bootstrap_control',
                    'f1', 'f1_shuffle', 'f1_bootstrap', 'f1_bootstrap_shuffle',
                    'f1_bootstrapTrain', 'f1_bootstrapTrain_shuffle', 'f1_bootstrap_control',
                    'f1_byClass', 'f1_byClass_shuffle', 'f1_byClass_bootstrap', 'f1_byClass_bootstrap_shuffle',
                    'f1_byClass_bootstrapTrain', 'f1_byClass_bootstrapTrain_shuffle',
                    'confusion_matrix', 'confusion_matrix_shuffle', 'confusion_matrix_bootstrap',
                    'confusion_matrix_bootstrap_shuffle', 'confusion_matrix_bootstrapTrain',
                    'confusion_matrix_bootstrapTrain_shuffle', 'confusion_matrix_bootstrap_control',
                    'correlations', 'correlations_shuffle', 'correlations_bootstrap',
                    'correlations_bootstrap_shuffle', 'correlations_bootstrapTrain',
                    'correlations_bootstrapTrain_shuffle']
LOFO_COLUMNS = ['participant_id', 'embedding_model', 'matrix_type',
                'n_cluster', 'num_draws', 'reg_strength', 'time_before', 'time_after',
                'clustering_seed', 'bootstrap_seed', 'seed_i', 'importances', 'importances_ratio',
                'importances_bootstrap', 'importances_bootstrap_ratio',
                'importances_bootstrapTrain', 'importances_bootstrapTrain_ratio']

class ParticipantData():
    '''One participant's data: matrix construction/loading and decoding runs.'''
    def __init__(self, participant_id: str, condition: str, feature_name: str, matrix_type: str, time_before: int,
                 time_after: int, timeseries: bool, detrend: bool, normalize: bool, data_path: str, save_path: str,  n_cluster: int,
                 n_seeds: int, p_value: float, silent_control_data_path: str = None) -> None:
        self.participant_id =  participant_id
        self.condition = condition
        self.feature_name = feature_name
        self.matrix_type = matrix_type
        self.time_before, self.time_after = time_before, time_after
        self.timeseries = timeseries
        self.detrend, self.normalize = detrend, normalize
        self.data_path = data_path
        self.save_path = save_path
        self.silent_control_data_path = silent_control_data_path
        self.transcription = None
        self.n_cluster = n_cluster
        self.n_seeds, self.p_value = n_seeds, p_value

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.get_electrode_count()

        conditions = {'speaking': 0, 'listening': 1}
        if self.condition not in conditions:
            raise ValueError('Condition must be either "speaking" or "listening"')
        else:
            self.sid = conditions[condition]

    def get_attributes(self) -> str:
        attributes = vars(self)
        return ', '.join(f"{key}: {value}" for key, value in attributes.items())
    
    def get_electrode_count(self) -> None:
        electrode_count = 0
        n_freqbands = 0
        first_run = True
        for file in os.listdir(self.data_path):
            if file.startswith(self.participant_id) and self.feature_name in file:
                if first_run:
                    neural_data = np.load(f'{self.data_path}{file}')
                    n_freqbands = neural_data.shape[1]
                    first_run = False
                electrode_count += 1
        self.electrode_count = electrode_count
        self.n_freqbands = n_freqbands
        self.logger.info(f'For participant {self.participant_id}: Electrode counter: {electrode_count}, n_freqbands: {n_freqbands}')   

    def extract_selective_matrix(self, save_matrix):
        '''Build the selective matrix per clustering seed and save matrices plus electrode table.'''
        columns = ['participant_id', 'clustering', 'n_cluster', 'seed', 'electrode_name', 'freqband', 'p_value']
        selective_electrodes_df = pd.DataFrame(columns = columns)

        for seed_i in range(self.n_seeds):

            X, y, selective_electrodes_df = SelectiveMatrix(self.transcription_df, 
                                self.n_cluster, self.time_before, self.time_after, self.n_freqbands, 
                                self.electrode_count, self.detrend, self.normalize, self.participant_id, 
                                self.feature_name, self.p_value, seed_i, self.data_path, 
                                self.timeseries).build_matrix(selective_electrodes_df)
            if save_matrix:
                np.save(f'{self.save_path}{self.participant_id}_{self.condition}_X_{self.matrix_type}_timeBefore{self.time_before}'+
                        f'_timeAfter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}_{self.feature_name}_{self.transcription.pretrained_embedding_model}_nbrclusters{self.n_cluster}'+
                        f'_seed{seed_i}_pvalue{self.p_value}.npy', X) 
                np.save(f'{self.save_path}{self.participant_id}_{self.condition}_y_{self.matrix_type}_timeBefore{self.time_before}'+
                        f'_timeAfter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}_{self.feature_name}_{self.transcription.pretrained_embedding_model}_nbrclusters{self.n_cluster}'+
                        f'_seed{seed_i}_pvalue{self.p_value}.npy', y) 
        selective_electrodes_df.to_csv(f'{self.save_path}{self.participant_id}clustering_selective_electrodes_df_timeBefore{self.time_before}'+
                                        f'_timeAfter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}_{self.feature_name}_{self.transcription.pretrained_embedding_model}'+
                                        f'_nbrclusters{self.n_cluster}_pvalue{self.p_value}.csv', index=False)

    def default_case(self):
        raise ValueError('Matrix type not recognized.')

    def extract_matrix(self, save_matrix: bool = True) -> None:
        print(f'Extracting matrix for participant {self.participant_id}...')
        transcription_df = self.transcription.transcription_df
        if 'sid' in transcription_df.columns:
            # raw CSVs: keep only this condition's rows; anonymized packages
            # are pre-filtered (sid==0 only) and carry no 'sid' column
            transcription_df = transcription_df[transcription_df['sid'] == self.sid]
        self.transcription_df = transcription_df.reset_index(drop=True)
        switch = {
            'selective': self.extract_selective_matrix}
        # lazy dispatch: calling the methods inside the dict would execute both branches
        handler = switch.get(self.matrix_type)
        if handler is None:
            self.default_case()
        handler(save_matrix)

    def load_matrix(self, seed_i: int = None) -> None:
        '''Load a saved matrix (and cluster labels for selective).'''
        transcription_df = self.transcription.transcription_df
        if 'sid' in transcription_df.columns:
            # raw CSVs: keep only this condition's rows; anonymized packages
            # are pre-filtered (sid==0 only) and carry no 'sid' column
            transcription_df = transcription_df[transcription_df['sid'] == self.sid]
        self.transcription_df = transcription_df.reset_index(drop=True)
        if self.matrix_type == "full":
            X = np.load(f'{self.save_path}{self.participant_id}_{self.condition}_X_{self.matrix_type}_timeBefore{self.time_before}_timeAfter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}_{self.feature_name}_{self.transcription.pretrained_embedding_model}.npy')
            self.X = X
        elif self.matrix_type == "selective":
            self.seed_i = seed_i
            X = np.load(f'{self.save_path}{self.participant_id}_{self.condition}_X_{self.matrix_type}_timeBefore{self.time_before}_timeAfter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}_{self.feature_name}_{self.transcription.pretrained_embedding_model}_nbrclusters{self.n_cluster}_seed{seed_i}_pvalue{self.p_value}.npy')
            y = np.load(f'{self.save_path}{self.participant_id}_{self.condition}_y_{self.matrix_type}_timeBefore{self.time_before}_timeAfter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}_{self.feature_name}_{self.transcription.pretrained_embedding_model}_nbrclusters{self.n_cluster}_seed{seed_i}_pvalue{self.p_value}.npy')
            self.transcription_df['cluster_labels_seed_i'] = y
            self.X = X
        else:
            raise ValueError('Matrix type not recognized')

    def run_decoding(self, regularization_strength: float, num_draws: int, n_seeds_bootstrap: int, save_path: str) -> None:
        '''Run decoding over clustering seeds; save accuracy and feature-importance results.'''
        accuracies_df_grouped = pd.DataFrame(columns=ACCURACY_COLUMNS)

        lofo_df = pd.DataFrame(columns=LOFO_COLUMNS)

        speechdecoder = SpeechDecoder(regularization_strength, num_draws, n_seeds_bootstrap, self.matrix_type) 

        if self.matrix_type == 'full':
            self.load_matrix()
            accuracies_df_grouped, lofo_df = speechdecoder.decode(accuracies_df_grouped, lofo_df, self.X, self.transcription_df, self.seed_i, 
                                                    self.participant_id, self.transcription.pretrained_embedding_model, 
                                                    self.matrix_type, self.n_cluster, self.time_before, 
                                                    self.time_after)
        elif self.matrix_type == 'selective':                                                  
            for seed_i in range(self.n_seeds): 
                print(f'Running decoding for clustering seed {seed_i}...')
                self.load_matrix(seed_i)
                # phoneme-control flag inferred from the save path
                phoneme_control = 'SemanticDecoding_Phoneme' in save_path
                accuracies_df_grouped, lofo_df = speechdecoder.decode(accuracies_df_grouped, lofo_df, self.X, self.transcription_df, self.seed_i,
                                                    self.participant_id, self.transcription.pretrained_embedding_model,
                                                    self.matrix_type, self.n_cluster, self.time_before,
                                                    self.time_after, phoneme_control=phoneme_control)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if not os.path.exists(f'{save_path}featureImportance/'):
            os.makedirs(f'{save_path}featureImportance/')

        accuracies_df_grouped.to_pickle(f'{save_path}{self.participant_id}_nclusters{self.n_cluster}_{self.condition}_X_{self.matrix_type}'+
                                        f'_timebefore{self.time_before}_timeafter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}'+
                                        f'_{self.feature_name}_{self.transcription.pretrained_embedding_model}_numdraws{num_draws}_reg{regularization_strength}_p{self.p_value}.csv')
        lofo_df.to_pickle(f'{save_path}featureImportance/{self.participant_id}_nclusters{self.n_cluster}_{self.condition}_X_{self.matrix_type}'+
                          f'_timebefore{self.time_before}_timeafter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}'+
                          f'_{self.feature_name}_{self.transcription.pretrained_embedding_model}_numdraws{num_draws}_reg{regularization_strength}_p{self.p_value}.csv')

    def run_decoding_for_3_area_test(self, regularization_strength: float, num_draws: int, n_seeds_bootstrap: int,
                     save_path: str, electrode_info: pd.DataFrame) -> None:

        '''Run decoding restricted to electrodes of three anatomical areas.'''
        accuracies_df_grouped = pd.DataFrame(columns=ACCURACY_COLUMNS)

        lofo_df = pd.DataFrame(columns=LOFO_COLUMNS)

        speechdecoder = SpeechDecoder(regularization_strength, num_draws, n_seeds_bootstrap, self.matrix_type)

        if self.matrix_type == 'full':
            self.load_matrix()
            accuracies_df_grouped, lofo_df = speechdecoder.decode(accuracies_df_grouped, lofo_df, self.X,
                                                                  self.transcription_df, self.seed_i,
                                                                  self.participant_id,
                                                                  self.transcription.pretrained_embedding_model,
                                                                  self.matrix_type, self.n_cluster, self.time_before,
                                                                  self.time_after)
        elif self.matrix_type == 'selective':

            for seed_i in range(self.n_seeds):
                print(f'Running decoding for clustering seed {seed_i}...')
                self.load_matrix(seed_i)
                previous_X = self.X
                phoneme_control = False
                electrode_info_seed = electrode_info[(electrode_info['seed'] == seed_i) & (electrode_info['pid_x'] == self.participant_id)].reset_index(drop=True)
                three_area = ['parstriangularis','precentral','middletemporal']
                filtered_df = electrode_info_seed[electrode_info_seed['area'].isin(three_area)]
                indices = filtered_df.index
                selected_X = previous_X[:, indices]

                self.X = selected_X
                accuracies_df_grouped, lofo_df = speechdecoder.decode(accuracies_df_grouped, lofo_df, self.X,
                                                                      self.transcription_df, self.seed_i,
                                                                      self.participant_id,
                                                                      self.transcription.pretrained_embedding_model,
                                                                      self.matrix_type, self.n_cluster,
                                                                      self.time_before,
                                                                      self.time_after, phoneme_control=phoneme_control)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not os.path.exists(f'{save_path}featureImportance/'):
            os.makedirs(f'{save_path}featureImportance/')

        accuracies_df_grouped.to_pickle(
            f'{save_path}{self.participant_id}_nclusters{self.n_cluster}_{self.condition}_X_{self.matrix_type}' +
            f'_timebefore{self.time_before}_timeafter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}' +
            f'_{self.feature_name}_{self.transcription.pretrained_embedding_model}_numdraws{num_draws}_reg{regularization_strength}_p{self.p_value}.csv')
        lofo_df.to_pickle(
            f'{save_path}featureImportance/{self.participant_id}_nclusters{self.n_cluster}_{self.condition}_X_{self.matrix_type}' +
            f'_timebefore{self.time_before}_timeafter{self.time_after}_detrend{self.detrend}_normalize{self.normalize}' +
            f'_{self.feature_name}_{self.transcription.pretrained_embedding_model}_numdraws{num_draws}_reg{regularization_strength}_p{self.p_value}.csv')
