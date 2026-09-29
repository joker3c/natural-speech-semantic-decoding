"""Load a conversation transcript and attach pretrained word embeddings.

Two data sources, selected automatically:
  raw CSV     — Conversation/PTxx.csv with a 'word' column: lowercase, drop
                confidential words, look up embeddings in a gensim model
                (Word2Vec / GloVe / FastText), drop OOV rows, optionally
                normalize. Needs gensim + Embedding_model/.
  anonymized  — Conversation/PTxx.csv WITHOUT a 'word' column (already
                sid-filtered) plus a row-aligned Embedding/<model>/PTxx.npy
                matrix of FINAL normalized vectors (all-NaN rows = OOV).
                The loader attaches the vectors, drops the NaN rows and
                synthesizes opaque word IDs from identical vectors
                (same vector -> same ID) — the same partition the original
                words had. No gensim needed; vectors must NOT be
                re-normalized (they are the final get_embedding output).
"""

import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

# gensim is imported lazily inside the raw-CSV branch only, so machines running
# anonymized packages do not need gensim installed.

CONFIDENTIAL_WORDS = ['alex', 'boston', 'alex\'s', 'ha', 'haha', 'hahaha', 'er', 'ck', 'b', 'he']


def _load_facebook_vectors(path):
    from gensim.models.fasttext import load_facebook_vectors
    return load_facebook_vectors(path)


# ===== Shared loaders =====
def load_transcription_csv(transcription_path: str, participant_id: str) -> pd.DataFrame:
    '''Read a transcript CSV (raw or anonymized) and compute start_time_ms.'''
    transcription_df = pd.read_csv(f'{transcription_path}{participant_id}.csv')
    transcription_df['start_time_ms'] = transcription_df['start_time'] * 1000
    return transcription_df


def preprocess_transcription_df(transcription_df: pd.DataFrame) -> pd.DataFrame:
    '''Raw-CSV preprocessing: lowercase and remove confidential replacement words.'''
    transcription_df['word'] = transcription_df['word'].apply(lambda x: str(x).lower())

    transcription_df = transcription_df[~transcription_df['word'].isin(CONFIDENTIAL_WORDS)]

    transcription_df.reset_index(drop=True)
    return transcription_df


def embedding_package_path(transcription_path: str, participant_id: str, model: str) -> str:
    '''Anonymized-package matrix path: <root>/Embedding/<model>/<PTxx>.npy,
    where <root> is the parent of the Conversation/ folder transcription_path points to.'''
    root = os.path.dirname(transcription_path.rstrip('/').rstrip('\\'))
    return os.path.join(root, 'Embedding', model, f'{participant_id}.npy')


def anonymized_word_ids(embeddings: np.ndarray) -> list:
    '''Opaque ID per distinct embedding row (w0000, ...), first-occurrence order.
    Same vector -> same ID, so the ID partition equals the original word partition.'''
    seen = {}
    codes = np.empty(len(embeddings), dtype=int)
    for i, row in enumerate(embeddings):
        key = row.tobytes()
        codes[i] = seen.setdefault(key, len(seen))
    return [f'w{c:04d}' for c in codes]


class Transcription():
    '''Load a transcript, preprocess words, and attach pretrained embeddings.'''
    def __init__(self, participant_id: str, transcription_path: str, pretrained_embedding_model: str, normalize_embeddings: bool) -> None:
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

        # Anonymized package: attach the model's embedding matrix, drop OOV rows
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

        # raw-CSV path: needs a gensim embedding model
        from gensim.models import KeyedVectors

        if self.pretrained_embedding_model == 'word2vec':
            # Trained on google news
            model_embedding = KeyedVectors.load_word2vec_format('../Embedding_model/Word2Vec/GoogleNews-vectors-negative300.bin.gz', binary=True)
        elif self.pretrained_embedding_model == 'fasttext':
            # Trained on common crawl
            model_embedding = _load_facebook_vectors('../Embedding_model/FastText/crawl-300d-2M-subword.bin')
        elif self.pretrained_embedding_model == 'glove':
            # Trained on Common Crawl; first run glove2word2vec to create glove_readable.txt
            model_embedding = KeyedVectors.load_word2vec_format('../Embedding_model/Glove/glove_readable.txt',
                                binary=False)
        else:
            raise ValueError('Pretrained embedding model not recognized')

        transcription_df['embedding'] = transcription_df['word'].apply(lambda word: get_word_embedding(word, model_embedding))
        # Drop out-of-vocabulary words (NaN embeddings)
        transcription_df = transcription_df[~(transcription_df['embedding'].apply(lambda x: np.any(np.isnan(x))))]

        if self.normalize_embeddings:
            normalized_embeddings = normalize(np.array(transcription_df['embedding'].tolist()), axis=0)
            transcription_df['embedding'] = [normalized_embeddings[i,:] for i in range(normalized_embeddings.shape[0])]

        transcription_df.reset_index(drop=True)
        return transcription_df


# ===== Helper functions =====
def get_word_embedding(word, model_embedding):
    try:
        return model_embedding[word]
    except KeyError:
        # NaN vector marks out-of-vocabulary words
        arr = np.zeros((300,))
        arr[:] = np.nan
        return arr
