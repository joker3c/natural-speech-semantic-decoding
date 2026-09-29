"""SSDD semantic-dimension decoding.

Clusters words by the 6 human-rated semantic dimensions (K-means, 10
clusters) instead of Word2Vec spectral clustering.
Configs: configs/Control_SemanticDimension/.
"""
import os
import json
import time
import logging
from modules.participant_data_semantic_dimension import ParticipantData
from modules.transcription_semantic_dimension import Transcription

# Decoding setting
GENERATE_MATRICES = True
RUN_DECODING = True

GET_ALL = GENERATE_MATRICES or RUN_DECODING


def run_decoding(config_path: str, logger: logging.Logger) -> None:
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
        participant_data.run_decoding(**config["decoding_params"])
        print('Decoding done.')


def run_all_configs(configs_folder: str, logger: logging.Logger) -> None:
    print('Running all configs.')
    for config_file in os.listdir(configs_folder):
        if config_file.endswith('.json') and config_file != 'stats_config.json':
            config_path = os.path.join(configs_folder, config_file)
            run_decoding(config_path, logger)


if __name__ == "__main__":
    print('Running main.')
    logging.basicConfig(filename='speech_decoding_logger.log', level=logging.INFO, filemode='a')
    logger = logging.getLogger(__name__)
    run_all_configs('configs/Control_SemanticDimension/', logger)
