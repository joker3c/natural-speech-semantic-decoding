"""Transcript loader that keeps only content words (NN/VB/JJ/RB) and lemmatizes.

Two data sources, selected automatically (same convention as
modules/transcription.py):
  raw CSV     — Conversation/PTxx.csv with a 'word' column: lowercase, drop
                confidential words, embed (gensim), drop OOV rows, POS-tag,
                keep content words, lemmatize (VB->v, NN->n), normalize.
                Needs gensim + nltk + Embedding_model/.
  anonymized  — Conversation/PTxx.csv WITHOUT a 'word' column plus a
                row-aligned Embedding/<model>_postag/PTxx.npy matrix of FINAL
                normalized lemma vectors (all-NaN rows = dropped by the
                content-word filter). The loader keeps the non-NaN rows and
                takes the opaque lemma IDs from the CSV's pos_tag_lemma_id
                column: one lemma spans several word-form vectors, so
                vector-keyed IDs would split it into several "words" and
                change the SpectralClustering input. No gensim/nltk needed;
                vectors must NOT be re-normalized.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from modules.transcription import (CONFIDENTIAL_WORDS, anonymized_word_ids,
                                   embedding_package_path, load_transcription_csv)

# gensim / nltk are imported lazily inside the raw-CSV branch only, so machines
# running anonymized packages do not need them installed.
_lemmatizer = None


def _ensure_nltk_resources() -> None:
    '''Download the NLTK data and create the lemmatizer (raw-CSV path only).'''
    global _lemmatizer
    if _lemmatizer is not None:
        return
    import nltk
    from nltk.stem import WordNetLemmatizer

    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')
    _lemmatizer = WordNetLemmatizer()


class Transcription():
    '''Load a transcript, attach embeddings, keep content words, and lemmatize.'''

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

        # Anonymized package: attach the content-word lemma matrix, drop the
        # rows the filter removed and synthesize lemma IDs. Everything
        # downstream is then source-agnostic.
        emb_path = embedding_package_path(self.transcription_path, self.participant_id,
                                          self.pretrained_embedding_model + '_postag')
        embeddings = np.load(emb_path)
        if embeddings.shape != (len(transcription_df), embeddings.shape[-1]):
            raise ValueError(f'{emb_path}: {embeddings.shape[0]} rows do not align with '
                             f'{len(transcription_df)} CSV rows')
        keep = ~np.isnan(embeddings).any(axis=1)
        if 'pos_tag_keep' in transcription_df.columns:
            # export-time marker: 1 exactly on the rows with a stored vector
            marker = transcription_df['pos_tag_keep'].to_numpy().astype(int)
            if not np.array_equal(marker, keep.astype(int)):
                raise ValueError(f'{emb_path}: pos_tag_keep column disagrees with the matrix NaN mask')
        transcription_df = transcription_df[keep]
        transcription_df['embedding'] = list(embeddings[keep])
        if 'pos_tag_lemma_id' in transcription_df.columns:
            # Lemma-level opaque IDs; see module docstring for why vector-keyed
            # IDs would split lemmas and change the clustering input.
            lemma_ids = transcription_df['pos_tag_lemma_id'].tolist()
            if any(not isinstance(v, str) or not v for v in lemma_ids):
                raise ValueError(f'{emb_path}: pos_tag_lemma_id has empty or non-string '
                                 f'values on kept rows')
            transcription_df['word'] = lemma_ids
        else:
            # packages without pos_tag_lemma_id: fall back to vector-keyed
            # IDs (splits lemmas; see module docstring)
            transcription_df['word'] = anonymized_word_ids(embeddings[keep])
        return transcription_df

    def preprocess_transcription(self, transcription_df: pd.DataFrame) -> pd.DataFrame:
        if 'embedding' in transcription_df.columns:
            # anonymized package: already filtered and embedded
            return transcription_df

        transcription_df['word'] = transcription_df['word'].apply(lambda x: str(x).lower())

        # Remove words replaced for confidentiality
        transcription_df = transcription_df[~transcription_df['word'].isin(CONFIDENTIAL_WORDS)]

        transcription_df.reset_index(drop=True)
        return transcription_df

    def get_embedding(self, transcription_df: pd.DataFrame) -> pd.DataFrame:
        if 'embedding' in transcription_df.columns:
            # anonymized package: vectors are already the final normalized
            # output — do not re-normalize or modify them in any way
            return transcription_df

        _ensure_nltk_resources()
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
            # Trained on common crawl
            model_embedding = KeyedVectors.load_word2vec_format('../Embedding_model/Glove/glove_readable.txt',
                                                                binary=False)
        else:
            raise ValueError('Pretrained embedding model not recognized')

        transcription_df['embedding'] = transcription_df['word'].apply(
            lambda word: get_word_embedding(word, model_embedding))
        # Drop out-of-vocabulary words (NaN embeddings)
        transcription_df = transcription_df[~(transcription_df['embedding'].apply(lambda x: np.any(np.isnan(x))))]
        # pos_tag takes a list of words and returns [(word, tag)]
        transcription_df['tag'] = transcription_df['word'].apply(get_single_word_pos)
        # Keep only content words
        transcription_df = transcription_df[transcription_df['tag'].apply(is_keep_tag)].reset_index(drop=True)
        transcription_df['normalized_word'] = transcription_df.apply(lemmatize_word, axis=1)
        transcription_df['word'] = transcription_df['normalized_word']
        transcription_df.drop(columns=['normalized_word', 'tag'], inplace=True)
        if self.normalize_embeddings:
            normalized_embeddings = normalize(np.array(transcription_df['embedding'].tolist()), axis=0)
            transcription_df['embedding'] = [normalized_embeddings[i, :] for i in range(normalized_embeddings.shape[0])]

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

def is_keep_tag(tag):
    return tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ') or tag.startswith('RB')

def get_single_word_pos(word):
    if pd.isna(word):
        return None
    _ensure_nltk_resources()
    from nltk import pos_tag
    tagged = pos_tag([str(word)])
    return tagged[0][1]

def lemmatize_word(row):
    word = row['word']
    tag = row['tag']
    _ensure_nltk_resources()
    if tag.startswith('VB'):
        return _lemmatizer.lemmatize(str(word), pos='v')
    elif tag.startswith('NN'):
        return _lemmatizer.lemmatize(str(word), pos='n')
    return word  # non-verbs are not lemmatized
