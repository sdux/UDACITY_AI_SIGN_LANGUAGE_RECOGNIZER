import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def train_cv_model(self, n_states, X_train, length_train, X_test, length_test):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
              random_state=self.random_state, verbose=False).fit(X_train, length_train)

            score = hmm_model.score(X_test, length_test)

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_states))
                print("model score: {}".format(score))

            return hmm_model, score

        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n_states))

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    #parameters = n_comp ** 2 + 2 * n_comp *n_features-1
    #score = (-2 *logLikelihood + parameters *math.log(number of datapoints))
    """
    def train_model(self, n_states):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
              random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

            score = hmm_model.score(self.X, self.lengths)

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_states))
                print("model score: {}".format(score))

            return hmm_model, score

        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n_states))

    def select(self):
        '''
        returns selected model based on lowest BIC score
        '''
        best_BIC_score = np.inf
        selected_model = None

        for n_comp in range(self.min_n_components,self.max_n_components+1):
            # score is the logLikelyhood
            # model is the HMM model created for n_components
            try:
                hmm_model = self.base_model(n_comp)
                score = hmm_model.score(self.X, self.lengths)
                        # Calculate BIC score
                parameters = n_comp **2 + 2 * n_comp * len(self.X[0]) - 1
                BIC_score = (-2 * score + parameters * math.log(len(self.X)))

                if BIC_score < best_BIC_score:
                    selected_model = hmm_model
                    best_BIC_score = BIC_score
            except:
                pass


        # TODO implement model selection based on BIC scores
        return selected_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def train_model(self, n_states):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
              random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

            score = hmm_model.score(self.X, self.lengths)

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_states))
                print("model score: {}".format(score))

            return hmm_model, score

        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n_states))

    def select(self):
        '''
        returns selected model based on highest DIC score
        '''
        best_DIC_score = -np.inf
        selected_model = None

        for n_comp in range(self.min_n_components,self.max_n_components+1):
            # score is the logLikelyhood
            # model is the HMM model created for n_components
            try:
                hmm_model = self.base_model(n_comp)
                score = hmm_model.score(self.X, self.lengths)

                # Calculate DIC score
                # Sum all scores for words other than current word.
                summed_score = 0
                n_words = 0
                for word in self.hwords:
                    # pass through if word is the current word
                    if word == self.this_word:
                        continue
                    else:
                        n_words += 1
                        summed_score += hmm_model.score(self.X, self.lengths)

                DIC_score = score - (summed_score/n_words)

                if DIC_score > best_DIC_score:
                    selected_model = hmm_model
                    best_DIC_score = DIC_score

            except:
                pass


        # TODO implement model selection based on BIC scores
        return selected_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def train_cv_model(self, n_states, X_train, length_train, X_test, length_test):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            hmm_model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000,
              random_state=self.random_state, verbose=False).fit(X_train, length_train)

            score = hmm_model.score(X_test, length_test)

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, n_states))
                print("model score: {}".format(score))

            return hmm_model, score

        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, n_states))

    def select(self):
        '''
        returns selected model based on highest Cross Validation score
        '''
        best_CV_score = -np.inf
        selected_model = None
        best_n_comp = 0
        summed_score = 0
        CV_score = -np.inf

        # number of splits, either 3 or number of samples
        n_splits = min(len(self.lengths), 3)
        if n_splits < 2:
            n_splits = 2
            return
        #print("Value of n_split: ",n_splits)
        #print("value of self.lengths: ",len(self.lengths))
        for n_comp in range(self.min_n_components,self.max_n_components+1):

            for cv_train_idx, cv_test_idx in KFold(n_splits).split(self.sequences):
                # Split data set into train and test sets
                X_train, length_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, length_test = combine_sequences(cv_test_idx, self.sequences)

                # model is the HMM model created for n_components
                try:
                    hmm_model, test_score = self.train_cv_model(n_comp, X_train, length_train, X_test, length_test)

                    # sum scores for each state
                    summed_score += test_score

                    # average score per state
                    CV_score = summed_score / n_comp

                except:
                    pass

            if CV_score > best_CV_score:
                best_CV_score = CV_score
                best_n_comp = n_comp
                selected_model = hmm_model
        try:
            # Re-calculate selected model and score with full dataset
            selected_model, score = self.train_cv_model(best_n_comp, self.X, self.lengths, self.X, self.lengths)
            # TODO implement model selection based on BIC scores
        except:

            pass
        return selected_model