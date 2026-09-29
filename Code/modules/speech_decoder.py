"""Core logistic-regression semantic decoder: 5-fold CV, bootstrap, shuffle/permutation controls, phoneme-control relabeling."""
import numpy as np
import pandas as pd

from typing import Tuple

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import eng_to_ipa as ipa

class SpeechDecoder():
    """Word-level semantic decoding from EEG with bootstrap and shuffle controls."""

    def __init__(self, regularization_strength: float, num_draws: int, n_seeds_bootstrap: int, matrix_type: str) -> None:
        self.regularization_strength = regularization_strength
        self.num_draws = num_draws
        self.n_seeds_bootstrap = n_seeds_bootstrap
        self.matrix_type = matrix_type
    
    def train_test_split(self, n_splits: int = 5):
        skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)
        y_w_word_label = np.concatenate((np.array(self.transcription_df['embedding'].tolist()), 
                                         np.array(self.transcription_df['word'].tolist()).reshape(-1,1), 
                                         np.array(self.transcription_df['cluster_labels_seed_i'].tolist()).reshape(-1,1)), axis=1)
        indices = np.where(np.isnan(self.X).any(axis=1))[0]
        if len(indices) != 0:
            non_nan_indices = np.where(~np.isnan(self.X).any(axis=1))[0]
            self.X = self.X[non_nan_indices]
            y_w_word_label = y_w_word_label[non_nan_indices]
            self.y = y_w_word_label
            splits = skf.split(self.X, y_w_word_label[:, -1])
        else:
            self.y = y_w_word_label
            splits = skf.split(self.X, y_w_word_label[:, -1])
        return splits

    def train_test_split_for_control(self, n_splits: int = 5):
        skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)
        y_w_word_label = np.concatenate((np.array(self.transcription_df['embedding'].tolist()),
                                         np.array(self.transcription_df['word'].tolist()).reshape(-1,1),
                                         np.array(self.transcription_df['phoneme_label'].tolist()).reshape(-1,1)), axis=1)
        self.y = y_w_word_label
        splits = skf.split(self.X, y_w_word_label[:, -1])
        return splits

    def decode(self, accuracies_df: pd.DataFrame, lofo_df: pd.DataFrame, X: np.ndarray[np.ndarray], transcription_df: pd.DataFrame, 
               seed: int, participant_id: str, embedding_model: str, matrix_type: str, n_cluster:int, 
               time_before: int, time_after: int, phoneme_control: bool = False
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.X = X
        self.transcription_df = transcription_df
        self.seed = seed
        if phoneme_control:
            if 'phoneme_group' in transcription_df.columns:
                # Precomputed phoneme labels (same ipa.convert +
                # get_consonant_group logic as the recomputation below).
                transcription_df['phoneme_label'] = transcription_df['phoneme_group'].to_numpy().astype(int)
            else:
                pho_by_word = []
                for ii in transcription_df.word.values:
                    pho_by_word.append(ipa.convert(ii))

                pho_clus = []
                for ind, ii in enumerate(pho_by_word):
                    pho_clus.append(get_consonant_group(ii))
                first_pho_clus = np.zeros(len(pho_clus))
                for ind in range(len(pho_clus)):
                    if len(pho_clus[ind]) > 0:
                        first_pho_clus[ind] = pho_clus[ind][0]
                first_pho_clus = first_pho_clus.astype(int)
                transcription_df['phoneme_label'] = first_pho_clus
        # Train/test splits
        if phoneme_control:
            splits = self.train_test_split_for_control()
        else:
            splits = self.train_test_split()

        y_pred_tot, y_pred_tot_shuffle = [], []
        y_test_tot = []
        y_test_tot_bootstrap_dict = {seed_i_bootstrap: [] for seed_i_bootstrap in range(self.n_seeds_bootstrap)}
        y_pred_tot_bootstrap_dict = {seed_i_bootstrap: [] for seed_i_bootstrap in range(self.n_seeds_bootstrap)}
        y_pred_tot_bootstrap_shuffle_dict = {seed_i_bootstrap: [] for seed_i_bootstrap in range(self.n_seeds_bootstrap)}
        y_pred_tot_bootstrapTrain_dict = {seed_i_bootstrap: [] for seed_i_bootstrap in range(self.n_seeds_bootstrap)}
        y_pred_tot_bootstrapTrain_shuffle_dict = {seed_i_bootstrap: [] for seed_i_bootstrap in range(self.n_seeds_bootstrap)}
        y_pred_tot_bootstrap_control_dict = {seed_i_bootstrap: [] for seed_i_bootstrap in range(self.n_seeds_bootstrap)}
        for seed_i, (train_index, test_index) in enumerate(splits):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            X_train = np.concatenate((X_train, y_train), axis=1)
            X_test = np.concatenate((X_test, y_test), axis=1)
            np.random.seed(42+seed_i)
            np.random.shuffle(X_train)
            np.random.seed(42+seed_i)
            np.random.shuffle(X_test)
            y_train = X_train[:, self.X.shape[1]:] 
            y_test = X_test[:, self.X.shape[1]:] 
            X_train = X_train[:, :self.X.shape[1]].astype(float)
            X_test = X_test[:, :self.X.shape[1]].astype(float)
            y_train_label = y_train[:, -1]
            y_test_label = y_test[:, -1]
            y_train = y_train[:, :-2].astype(float)
            y_test = y_test[:, :-2].astype(float)

            # Without bootstrapping
            clf = LogisticRegression(random_state=0, multi_class='multinomial', C=self.regularization_strength).fit(X_train, y_train_label)
            y_pred = clf.predict(X_test)
            acc_i = metrics.balanced_accuracy_score(y_test_label, y_pred)

            # Shuffle control
            X_train_shuffle = X_train.copy()
            np.random.seed(22+seed_i)
            np.random.shuffle(X_train_shuffle)
            clf_shuffle = LogisticRegression(random_state=0, multi_class='multinomial', C=self.regularization_strength).fit(X_train_shuffle, y_train_label)
            y_pred_shuffle = clf_shuffle.predict(X_test)

            if len(y_pred_tot) == 0:
                y_pred_tot, y_pred_tot_shuffle = y_pred, y_pred_shuffle
                y_test_tot = y_test_label
            else:
                y_pred_tot = np.append(y_pred_tot, y_pred)
                y_pred_tot_shuffle = np.append(y_pred_tot_shuffle, y_pred_shuffle)
                y_test_tot = np.append(y_test_tot, y_test_label)
  
            for seed_i_bootstrap in range(self.n_seeds_bootstrap):
                X_train_, y_train_ = bootstrap_data(X_train, y_train_label, self.num_draws, seed_i_bootstrap)
                X_test_, y_test_ = bootstrap_data(X_test, y_test_label, self.num_draws, seed_i_bootstrap)


                X_train_shuffle_, y_train_shuffle_ = bootstrap_data(X_train_shuffle, y_train_label, self.num_draws, seed_i_bootstrap)
                X_test_shuffle_, y_test_shuffle_ = bootstrap_data(X_test, y_test_label, self.num_draws, seed_i_bootstrap)

                # With bootstrapping
                clf_bootstrap = LogisticRegression(random_state=0, multi_class='multinomial', C=self.regularization_strength).fit(X_train_, y_train_)
                y_pred_bootstrap = clf_bootstrap.predict(X_test_)
                y_pred_bootstrapTrain = clf_bootstrap.predict(X_test)
                acc_i_bootstrap = metrics.balanced_accuracy_score(y_test_, y_pred_bootstrap)
                acc_i_bootstrapTrain = metrics.balanced_accuracy_score(y_test_label, y_pred_bootstrapTrain)

                # Shuffle control
                np.random.seed(22+seed_i)
                np.random.shuffle(X_train_)
                clf_shuffle = LogisticRegression(random_state=0, multi_class='multinomial', C=self.regularization_strength).fit(X_train_, y_train_)
                y_pred_bootstrap_shuffle = clf_shuffle.predict(X_test_)
                y_pred_bootstrapTrain_shuffle = clf_shuffle.predict(X_test)

                # Control for bootstrapping
                clf_bootstrap_shuffle = LogisticRegression(random_state=0, multi_class='multinomial', C=self.regularization_strength).fit(X_train_shuffle_, y_train_shuffle_)
                y_pred_bootstrap_control = clf_bootstrap_shuffle.predict(X_test_shuffle_)

                y_test_tot_bootstrap_dict[seed_i_bootstrap] = np.append(np.array(y_test_tot_bootstrap_dict[seed_i_bootstrap]), y_test_)
                y_pred_tot_bootstrap_dict[seed_i_bootstrap] = np.append(np.array(y_pred_tot_bootstrap_dict[seed_i_bootstrap]), y_pred_bootstrap)
                y_pred_tot_bootstrap_shuffle_dict[seed_i_bootstrap] = np.append(np.array(y_pred_tot_bootstrap_shuffle_dict[seed_i_bootstrap]), y_pred_bootstrap_shuffle)
                y_pred_tot_bootstrapTrain_dict[seed_i_bootstrap] = np.append(np.array(y_pred_tot_bootstrapTrain_dict[seed_i_bootstrap]), y_pred_bootstrapTrain)
                y_pred_tot_bootstrapTrain_shuffle_dict[seed_i_bootstrap] = np.append(np.array(y_pred_tot_bootstrapTrain_shuffle_dict[seed_i_bootstrap]), y_pred_bootstrapTrain_shuffle)
                y_pred_tot_bootstrap_control_dict[seed_i_bootstrap] = np.append(np.array(y_pred_tot_bootstrap_control_dict[seed_i_bootstrap]), y_pred_bootstrap_control)

                lofo = permutation_importance(clf, X_test, y_test_label, n_repeats=10, random_state=0)
                lofo_bootstrap = permutation_importance(clf_bootstrap, X_test_, y_test_, n_repeats=10, random_state=0)
                lofo_bootstrapTrain = permutation_importance(clf_bootstrap, X_test, y_test_label, n_repeats=10, random_state=0)
                lofo_df = pd.concat([lofo_df, pd.DataFrame({'participant_id': participant_id, 
                            'embedding_model': embedding_model, 'matrix_type': matrix_type, 
                            'n_cluster': n_cluster, 'num_draws': self.num_draws, 
                            'reg_strength': self.regularization_strength, 'time_before': time_before, 
                            'time_after': time_after, 'clustering_seed': seed, 'bootstrap_seed': seed_i_bootstrap,
                            'seed_i': seed_i, 'importances': [lofo.importances], 
                            'importances_ratio': [lofo.importances/acc_i],
                            'importances_bootstrap': [lofo_bootstrap.importances], 'importances_bootstrap_ratio': [lofo_bootstrap.importances/acc_i_bootstrap],
                            'importances_bootstrapTrain': [lofo_bootstrapTrain.importances], 
                            'importances_bootstrapTrain_ratio': [lofo_bootstrapTrain.importances/acc_i_bootstrapTrain]})])


        for key in y_test_tot_bootstrap_dict.keys():
            y_test_tot_bootstrap = y_test_tot_bootstrap_dict[key]
            y_pred_tot_bootstrap = y_pred_tot_bootstrap_dict[key]
            y_pred_tot_bootstrap_shuffle = y_pred_tot_bootstrap_shuffle_dict[key]
            y_pred_tot_bootstrapTrain = y_pred_tot_bootstrapTrain_dict[key]
            y_pred_tot_bootstrapTrain_shuffle = y_pred_tot_bootstrapTrain_shuffle_dict[key]
            y_pred_tot_bootstrap_control = y_pred_tot_bootstrap_control_dict[key]

            corrs_array = np.corrcoef(y_pred_tot.astype(int), y_test_tot.astype(int))[0, 1]
            corrs_shuffle_array = np.corrcoef(y_pred_tot_shuffle.astype(int), y_test_tot.astype(int))[0, 1]
            corrs_bootstrap_array = np.corrcoef(y_pred_tot_bootstrap.astype(int), y_test_tot_bootstrap.astype(int))[0, 1]
            corrs_bootstrap_shuffle_array = np.corrcoef(y_pred_tot_bootstrap_shuffle.astype(int), y_test_tot_bootstrap.astype(int))[0, 1]
            corrs_bootstrapTrain_array = np.corrcoef(y_pred_tot_bootstrapTrain.astype(int), y_test_tot.astype(int))[0, 1]
            corrs_bootstrapTrain_shuffle_array = np.corrcoef(y_pred_tot_bootstrapTrain_shuffle.astype(int), y_test_tot.astype(int))[0, 1]

            acc_ = metrics.balanced_accuracy_score(y_test_tot, y_pred_tot)
            print("acc_:",acc_)
            acc_shuffle_ = metrics.balanced_accuracy_score(y_test_tot, y_pred_tot_shuffle)
            acc_bootstrap_ = metrics.balanced_accuracy_score(y_test_tot_bootstrap, y_pred_tot_bootstrap)
            print("acc_bootstrap:",acc_bootstrap_)
            acc_bootstrap_shuffle_ = metrics.balanced_accuracy_score(y_test_tot_bootstrap, y_pred_tot_bootstrap_shuffle)
            acc_bootstrapTrain_ = metrics.balanced_accuracy_score(y_test_tot, y_pred_tot_bootstrapTrain)
            acc_bootstrapTrain_shuffle_ = metrics.balanced_accuracy_score(y_test_tot, y_pred_tot_bootstrapTrain_shuffle)
            acc_bootstrap_control = metrics.balanced_accuracy_score(y_test_tot_bootstrap, y_pred_tot_bootstrap_control)
        
            f1_ = metrics.f1_score(y_test_tot, y_pred_tot, average='macro')
            f1_shuffle_ = metrics.f1_score(y_test_tot, y_pred_tot_shuffle, average='macro')
            f1_bootstrap_ = metrics.f1_score(y_test_tot_bootstrap, y_pred_tot_bootstrap, average='macro')
            f1_bootstrap_shuffle_ = metrics.f1_score(y_test_tot_bootstrap, y_pred_tot_bootstrap_shuffle, average='macro')
            f1_bootstrapTrain_ = metrics.f1_score(y_test_tot, y_pred_tot_bootstrapTrain, average='macro')
            f1_bootstrapTrain_shuffle_ = metrics.f1_score(y_test_tot, y_pred_tot_bootstrapTrain_shuffle, average='macro')
            f1_bootstrap_control = metrics.f1_score(y_test_tot_bootstrap, y_pred_tot_bootstrap_control, average='macro')

            cm_tot = metrics.confusion_matrix(y_test_tot, y_pred_tot)
            cm_tot_shuffle = metrics.confusion_matrix(y_test_tot, y_pred_tot_shuffle)
            cm_tot_bootstrap = metrics.confusion_matrix(y_test_tot_bootstrap, y_pred_tot_bootstrap)
            cm_tot_bootstrap_shuffle = metrics.confusion_matrix(y_test_tot_bootstrap, y_pred_tot_bootstrap_shuffle)
            cm_tot_bootstrapTrain = metrics.confusion_matrix(y_test_tot, y_pred_tot_bootstrapTrain)
            cm_tot_bootstrapTrain_shuffle = metrics.confusion_matrix(y_test_tot, y_pred_tot_bootstrapTrain_shuffle)
            cm_tot_bootstrap_control = metrics.confusion_matrix(y_test_tot_bootstrap, y_pred_tot_bootstrap_control)

            _, _, f1, _ = metrics.precision_recall_fscore_support(y_test_tot, y_pred_tot)
            _, _, f1_shuffle, _ = metrics.precision_recall_fscore_support(y_test_tot, y_pred_tot_shuffle)
            _, _, f1_bootstrap, _ = metrics.precision_recall_fscore_support(y_test_tot_bootstrap, y_pred_tot_bootstrap)
            _, _, f1_bootstrap_shuffle, _ = metrics.precision_recall_fscore_support(y_test_tot_bootstrap, y_pred_tot_bootstrap_shuffle)
            _, _, f1_bootstrapTrain, _ = metrics.precision_recall_fscore_support(y_test_tot, y_pred_tot_bootstrapTrain)
            _, _, f1_bootstrapTrain_shuffle, _ = metrics.precision_recall_fscore_support(y_test_tot, y_pred_tot_bootstrapTrain_shuffle)

            accuracies_df = pd.concat([accuracies_df, pd.DataFrame({'participant_id': participant_id, 
                                'embedding_model': embedding_model, 'matrix_type': matrix_type, 'n_cluster': n_cluster, 'num_draws': self.num_draws, 
                                'reg_strength': self.regularization_strength, 'time_before': time_before, 'time_after': time_after,
                                'clustering_seed': seed, 'bootstrap_seed': key, 
                                'accuracy': [acc_], 'accuracy_shuffle': [acc_shuffle_], 
                                'accuracy_bootstrap': [acc_bootstrap_], 'accuracy_bootstrap_shuffle': [acc_bootstrap_shuffle_],
                                'accuracy_bootstrapTrain': [acc_bootstrapTrain_], 'accuracy_bootstrapTrain_shuffle': [acc_bootstrapTrain_shuffle_],
                                'accuracy_bootstrap_control': [acc_bootstrap_control],
                                'f1': [f1_], 'f1_shuffle': [f1_shuffle_], 
                                'f1_bootstrap': [f1_bootstrap_], 'f1_bootstrap_shuffle': [f1_bootstrap_shuffle_],
                                'f1_bootstrapTrain': [f1_bootstrapTrain_], 'f1_bootstrapTrain_shuffle': [f1_bootstrapTrain_shuffle_],
                                'f1_bootstrap_control': [f1_bootstrap_control],
                                'correlations': [corrs_array], 'correlations_shuffle': [corrs_shuffle_array], 
                                'correlations_bootstrap': [corrs_bootstrap_array], 'correlations_bootstrap_shuffle': [corrs_bootstrap_shuffle_array],
                                'correlations_bootstrapTrain': [corrs_bootstrapTrain_array], 'correlations_bootstrapTrain_shuffle': [corrs_bootstrapTrain_shuffle_array],
                                'y_test': [y_test_tot], 'y_test_bootstrap': [y_test_tot_bootstrap],
                                'y_pred': [y_pred_tot], 'y_pred_shuffle': [y_pred_tot_shuffle], 
                                'y_pred_bootstrap': [y_pred_tot_bootstrap], 'y_pred_bootstrap_shuffle': [y_pred_tot_bootstrap_shuffle],
                                'y_pred_bootstrapTrain': [y_pred_tot_bootstrapTrain], 'y_pred_bootstrapTrain_shuffle': [y_pred_tot_bootstrapTrain_shuffle],
                                'confusion_matrix': [cm_tot], 'confusion_matrix_shuffle': [cm_tot_shuffle],
                                'confusion_matrix_bootstrap': [cm_tot_bootstrap], 'confusion_matrix_bootstrap_shuffle': [cm_tot_bootstrap_shuffle],
                                'confusion_matrix_bootstrapTrain': [cm_tot_bootstrapTrain], 'confusion_matrix_bootstrapTrain_shuffle': [cm_tot_bootstrapTrain_shuffle],
                                'confusion_matrix_bootstrap_control': [cm_tot_bootstrap_control],
                                'f1_byClass': [f1], 'f1_byClass_shuffle': [f1_shuffle],
                                'f1_byClass_bootstrap': [f1_bootstrap], 'f1_byClass_bootstrap_shuffle': [f1_bootstrap_shuffle],
                                'f1_byClass_bootstrapTrain': [f1_bootstrapTrain], 'f1_byClass_bootstrapTrain_shuffle': [f1_bootstrapTrain_shuffle]})])

        return accuracies_df, lofo_df


# ===== Helper functions =====
def bootstrap_data(X:np.ndarray[np.ndarray], y: np.ndarray, num_draws: int,  bootstrap_seed: int,
                   shuffle: bool=True, time_series:bool = False) -> Tuple[np.ndarray[np.ndarray], np.ndarray]:
    """Class-balanced bootstrap: per repeat, average num_draws random samples of each class into one pseudo-sample."""
    unique_y = np.unique(y)
    result = []
    new_y = []

    np.random.seed(bootstrap_seed+115)
    num_repeats = X.shape[0]

    for i in range(num_repeats):
        mean_draws = []
        for val in unique_y:
            indices = np.where(y == val)[0]
            draw_indices = np.random.choice(indices, num_draws, replace=True)
            mean_draw = np.mean(X[draw_indices], axis=0)
            mean_draws.append(mean_draw)
            new_y.append(val)
        result.append(np.array(mean_draws))

    result = np.array(result)
    new_y = np.array(new_y)
    if time_series:
        result = result.reshape(-1, X.shape[1], X.shape[2])
    else:
        result = result.reshape(-1, X.shape[1])
        
    if shuffle:
        indices = np.arange(len(result))
        np.random.seed(32)
        np.random.shuffle(indices)
        result = result[indices]
        new_y = new_y[indices]

    return result, new_y


def get_consonant_group(word):
    """Map consonants of an IPA word to manner-of-articulation group indices (0-6), two-char phonemes first."""
    group0 = ['b', 'd', 'g', 'k', 'p', 't']  # plosive
    group1 = ['f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h']  # Fricative
    group2 = ['ʧ', 'ʤ']  # Affricate
    group3 = ['n', 'm', 'ŋ']  # Nasal
    group4 = ['l']  # Lateral
    group5 = ['r']  # Rhotic
    group6 = ['w', 'j']  # Glide

    result = []
    i = -1
    while i <= len(word):
        i += 1
        if i + 1 < len(word):
            sub = word[i:(i + 2)]
            if sub in group0:
                result.append(0)
                i += 1
                continue
            elif sub in group1:
                result.append(1)
                i += 1
                continue
            elif sub in group2:
                result.append(2)
                i += 1
                continue
            elif sub in group3:
                result.append(3)
                i += 1
                continue
            elif sub in group4:
                result.append(4)
                i += 1
                continue
            elif sub in group5:
                result.append(5)
                i += 1
                continue
            elif sub in group6:
                result.append(6)
                i += 1
                continue
        if i < len(word):
            sub = word[i]
            if sub in group0:
                result.append(0)
                continue
            elif sub in group1:
                result.append(1)
                continue
            elif sub in group2:
                result.append(2)
                continue
            elif sub in group3:
                result.append(3)
                continue
            elif sub in group4:
                result.append(4)
                continue
            elif sub in group5:
                result.append(5)
                continue
            elif sub in group6:
                result.append(6)
                continue
    return result