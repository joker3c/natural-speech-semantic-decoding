"""Phoneme control decoding.

Same pipeline as semantic_decoding.py, but each word is relabeled by the
consonant category of its first phoneme (7 classes) instead of its semantic
cluster. The relabeling is triggered when the config's save_path contains
"SemanticDecoding_Phoneme" (see modules/participant_data.py).
Configs: configs/Control_Phoneme/.
"""
import os
import json
import time
import logging

from modules.participant_data import ParticipantData
from modules.transcription import Transcription

# Decoding setting
GENERATE_MATRICES = True  # must run at least once; decoding loads cached matrices
RUN_DECODING = True

GET_ALL = GENERATE_MATRICES or RUN_DECODING


def run_decoding(config_path: str, logger: logging.Logger) -> None:
    """Run the matrix/decoding pipeline for one participant config."""
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
    """Run every participant config in the folder (skips stats_config.json)."""
    print('Running all configs.')
    reversed_files = list(reversed(os.listdir(configs_folder)))
    for config_file in reversed_files:
        if config_file.endswith('.json') and config_file != 'stats_config.json':
            config_path = os.path.join(configs_folder, config_file)
            print(config_file)
            run_decoding(config_path, logger)


if __name__ == "__main__":
    print('Running main.')
    logging.basicConfig(filename='speech_decoding_logger.log', level=logging.INFO, filemode='a')
    logger = logging.getLogger(__name__)
    run_all_configs('configs/Control_Phoneme/', logger)
