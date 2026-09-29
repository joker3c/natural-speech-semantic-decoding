"""Load a transcript and attach embeddings (Word2Vec/FastText/GloVe or SSDD ratings).

Two data sources, selected automatically (same scheme as modules/transcription.py):
  raw CSV     — Conversation/PTxx.csv with a 'word' column: lowercase, drop
                confidential words, embed (gensim models, or the SSDD
                6-dimension rating table for 'SemanticDimension'), drop OOV
                rows, optionally normalize. Needs gensim for the gensim models.
  anonymized  — Conversation/PTxx.csv WITHOUT a 'word' column plus a
                row-aligned Embedding/<model>/PTxx.npy matrix of FINAL
                normalized vectors (all-NaN rows = OOV); for
                'SemanticDimension' it is the (n_rows, 6) SSDD rating-vector
                export. Opaque word IDs are synthesized from identical
                vectors (same vector -> same ID). No gensim needed; vectors
                must NOT be re-normalized.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from modules.transcription import (load_transcription_csv, preprocess_transcription_df,
                                   embedding_package_path, anonymized_word_ids)

# gensim is imported lazily inside the raw-CSV branch only, so machines running
# anonymized packages do not need gensim installed.


class Transcription():
    '''Load a transcript, preprocess words, and attach embeddings.'''

    def __init__(self, participant_id: str, transcription_path: str, pretrained_embedding_model: str,
                 normalize_embeddings: bool) -> None:
        self.participant_id = participant_id
        self.transcription_path = transcription_path
        self.pretrained_embedding_model = pretrained_embedding_model
        self.normalize_embeddings = normalize_embeddings

        self.transcription_df = self.get_embedding(self.preprocess_transcription(self.get_transcription()))

    def get_attributes(self) -> str:
        attributes = vars(self)
        return ', '.join(f"{key}: {value}" for key, value in attributes.items())

    def get_transcription(self) -> pd.DataFrame:
        transcription_df = load_transcription_csv(self.transcription_path, self.participant_id)
        if 'word' in transcription_df.columns:
            return transcription_df

        # Anonymized package: attach the model's vector matrix, drop OOV rows
        # and synthesize word IDs. Everything downstream is then source-agnostic.
        emb_path = embedding_package_path(self.transcription_path, self.participant_id,
                                          self.pretrained_embedding_model)
        embeddings = np.load(emb_path)
        if embeddings.shape != (len(transcription_df), embeddings.shape[-1]):
            raise ValueError(f'{emb_path}: {embeddings.shape[0]} rows do not align with '
                             f'{len(transcription_df)} CSV rows')
        keep = ~np.isnan(embeddings).any(axis=1)
        transcription_df = transcription_df[keep]
        transcription_df['embedding'] = list(embeddings[keep])
        transcription_df['word'] = anonymized_word_ids(embeddings[keep])
        return transcription_df

    def preprocess_transcription(self, transcription_df: pd.DataFrame) -> pd.DataFrame:
        if 'embedding' in transcription_df.columns:
            # anonymized package: already preprocessed, OOV-free and embedded
            return transcription_df

        return preprocess_transcription_df(transcription_df)

    def get_embedding(self, transcription_df: pd.DataFrame) -> pd.DataFrame:
        if 'embedding' in transcription_df.columns:
            # anonymized package: vectors are already the final normalized
            # output — do not re-normalize or modify them in any way
            return transcription_df

        # raw-CSV path: gensim is only needed for the word2vec/fasttext/glove models
        from gensim.models import KeyedVectors
        from gensim.models.fasttext import load_facebook_vectors

        if self.pretrained_embedding_model == 'word2vec':
            # Trained on google news
            model_embedding = KeyedVectors.load_word2vec_format(
                '../Embedding_model/Word2Vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
        elif self.pretrained_embedding_model == 'fasttext':
            # Trained on common crawl
            model_embedding = load_facebook_vectors('../Embedding_model/FastText/crawl-300d-2M-subword.bin')
        elif self.pretrained_embedding_model == 'glove':
            # Trained on Common Crawl; first run glove2word2vec to create glove_readable.txt
            model_embedding = KeyedVectors.load_word2vec_format('../Embedding_model/Glove/glove_readable.txt',
                                                                binary=False)
        elif self.pretrained_embedding_model == 'SemanticDimension':
            print('Using SemanticDimension embedding')
            model_embedding = pd.read_csv('../Embedding_model/SemanticDimension/Estimated_semantic_dimensions_word2vec_English.csv')
        else:
            raise ValueError('Pretrained embedding model not recognized')

        if self.pretrained_embedding_model == 'SemanticDimension':
            transcription_df['embedding'] = transcription_df['word'].apply(
                lambda word: get_semantic_dimension_embedding(word, model_embedding))
        else:
            transcription_df['embedding'] = transcription_df['word'].apply(
                lambda word: get_word_embedding(word, model_embedding))
        # Drop out-of-vocabulary words (NaN embeddings)
        transcription_df = transcription_df[~(transcription_df['embedding'].apply(lambda x: np.any(np.isnan(x))))]

        if self.normalize_embeddings:
            normalized_embeddings = normalize(np.array(transcription_df['embedding'].tolist()), axis=0)
            transcription_df['embedding'] = [normalized_embeddings[i, :] for i in range(normalized_embeddings.shape[0])]

        transcription_df.reset_index(drop=True)
        return transcription_df


# ===== Helper functions =====
def get_semantic_dimension_embedding(word, target_df):
    match = target_df[target_df['word'] == word]
    if not match.empty:
        # Columns c1-c6 hold the six dimension ratings
        return match.iloc[0, 1:7].astype(float).values
    else:
        arr = np.zeros((6,))
        arr[:] = np.nan
        return arr

def get_word_embedding(word, model_embedding):
    try:
        return model_embedding[word]
    except KeyError:
        # NaN vector marks out-of-vocabulary words
        arr = np.zeros((300,))
        arr[:] = np.nan
        return arr
