from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from cachetools import LRUCache
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.externals import joblib
from textstat.textstat import textstat as ts

from constants import (FEAT_WC_CATEGORIES, FEAT_WC_CONCERN, FEAT_WC_LING,
                       FEAT_WC_PSYCH, NON_ALNUM_CHARS, STOPWORDS)
from LiwcUtil import LiwcUtil


class AirProfile(object):
    """Automatic analysis of Airbnb Host profiles.

    To Use:
    >>> from AirProfile import AirProfile
    >>> ap = AirProfile(liwc_path='../LIWC2007/liwc_2007.trie')
    >>> host_profile = "I have spent my life in the service industry." \
        "I look forward to being your host and I look forward to meeting you."
    >>> ap.predict_topics(input)
    >>> ap.predict_trust(input)

    Attributes:
        Prediction: A named tuple of the prediction result and probability.
    """

    __liwc_internal__ = None
    __vec_internal__ = None
    __classifier_cat_internal__ = None
    __classifier_trust_internal__ = {}

    __lemm_n = LRUCache(
        maxsize=256, missing=partial(WordNetLemmatizer().lemmatize, pos='n'))
    __lemm_v = LRUCache(
        maxsize=256, missing=partial(WordNetLemmatizer().lemmatize, pos='v'))

    __SentenceToken = namedtuple('SentenceToken', ['raw', 'clean'])
    Prediction = namedtuple('Prediction', ['prob', 'predict'])

    def __init__(self, liwc_path='', model_dir_path='./AirProfile/models'):
        """
        Args:
            liwc_path: The path to the LIWC 2007 trie. Trust prediction will
                not work unless this is specified.
            model_dir_path: The directory path for the AirProfile models.
                Defaults to './AirProfile/models'.
        """
        self.__liwc_path = liwc_path if isinstance(liwc_path,
                                                   Path) else Path(liwc_path)
        # TODO(kenlimmj): This should perform a high-level integrity check on
        #  the directory structure.
        self.__model_dir_path = model_dir_path if isinstance(
            model_dir_path, Path) else Path(model_dir_path)

    @property
    def __liwc(self):
        if not self.__liwc_internal__:
            self.__liwc_internal__ = LiwcUtil(self.__liwc_path)
        return self.__liwc_internal__

    @property
    def __vec(self):
        """Lazily loads the sentence vectorizer model from disk."""
        if not self.__vec_internal__:
            vec_path = (self.__model_dir_path / 'sentence_vectorizer' /
                        'vectorizer.pkl')
            if not (vec_path.exists() and vec_path.is_file()):
                raise ValueError

            self.__vec_internal__ = joblib.load(vec_path)

        return self.__vec_internal__

    @property
    def __classifier_cat(self):
        """Lazily loads the sentence category models from disk."""
        if not self.__classifier_cat_internal__:
            cat_model_dir = self.__model_dir_path / 'sentence_categories'
            if not (cat_model_dir.exists() and cat_model_dir.is_dir()):
                raise ValueError

            cat_models = []
            for cat_model_path in cat_model_dir.glob('*.pkl'):
                cat_models.append(joblib.load(cat_model_path))

            self.__classifier_cat_internal__ = cat_models

        return self.__classifier_cat_internal__

    def __get_classifier_trust(self, fname):
        """Lazily loads the trust model from disk."""
        if fname not in self.__classifier_trust_internal__:
            trust_model_path = self.__model_dir_path / 'trust' / fname
            if not (trust_model_path.exists() and trust_model_path.is_file()):
                raise ValueError

            self.__classifier_trust_internal__[fname] = joblib.load(
                trust_model_path)

        return self.__classifier_trust_internal__[fname]

    def predict_trust(self, input, strip_html=True):
        """Predicts the trustworthiness of a profile.

        Segments the input with sentence-level granularity, returning the
        probability that the profile represented by the input is perceived to
        be more trustworthy compared to other profiles of similar length.

        Args:
            input: An Airbnb host profile, as a string.
            strip_html: Whether HTML tags in the input should be stripped. True
                by default, but can be disabled for speed if the input is known
                to be sanitized.

        Returns:
            An AirProfile.Prediction object for trustworthiness of the profile.

        Raises:
            ValueError: If the input is an invalid or empty string.
            IOError: If the LIWC trie is not available.
        """
        if not (input and input.strip()):
            raise ValueError

        if not (self.__liwc_path.exists() and self.__liwc_path.is_file()):
            raise IOError

        sentence_tokens = self.__preprocess(input, strip_html)
        liwc_features = self.__liwc.summarize(input, sentence_tokens)

        wc = liwc_features['WC']
        liwc_features['wc_log'] = np.log(wc)
        liwc_features['readability'] = ts.flesch_kincaid_grade(
            input.decode('utf-8'))

        prediction_agg = np.empty(len(self.__classifier_cat))
        for s in sentence_tokens:
            prediction_agg += np.array(
                [c.predict for c in self.__classify_sentence(s)])

        for i in xrange(len(FEAT_WC_CATEGORIES)):
            liwc_features[FEAT_WC_CATEGORIES[i]] = prediction_agg[i]

        X = [
            liwc_features[f]
            for f in AirProfile.__get_trust_model_feat_cols(wc)
        ]
        X_shape = np.array(X).reshape(1, -1)
        model = self.__get_classifier_trust(
            AirProfile.__get_trust_model_fname(wc))

        return self.Prediction(
            np.round(model.predict_proba(X_shape)[0][1], 2),
            model.predict(X_shape)[0])

    def __classify_sentence(self, tokens):
        """Classifies a sentence based on the trust category models.

        Args:
            tokens: A list of AirProfile.SentenceToken, corresponding to a
                sentence.

        Returns:
            A list of AirProfile.Prediction. Each entry in the list corresponds
            to the prediction made by a category model.
        """
        vector = self.__vec.transform([' '.join(tokens.clean)])

        # TODO(kenlimmj): This should be an ordered or keyed data structure,
        # since the category classifiers are reconciled later on.
        output = []
        for classifier in self.__classifier_cat:
            prediction = self.Prediction(
                np.round(classifier.predict_proba(vector)[0], 2),
                classifier.predict(vector)[0])
            output.append(prediction)

        return output

    def predict_topics(self, input, strip_html=True):
        """Predicts the trust evaluation topics for a profile.

        Args:
            input: An Airbnb host profile, as a string.
            strip_html: Whether HTML tags in the input should be stripped. True
                by default, but can be disabled for speed if the input is known
                to be sanitized.

        Returns:
            A list of prediction probabilities

        Raises:
            ValueError: If the input is an invalid or empty string.
        """
        if not (input and input.strip()):
            raise ValueError

        sentence_tokens = self.__preprocess(input, strip_html)

        results = []
        for s in sentence_tokens:
            probs = np.array([c.prob[1] for c in self.__classify_sentence(s)])
            predictions = {}
            for i in range(len(FEAT_WC_CATEGORIES)):
                predictions[FEAT_WC_CATEGORIES[i]] = np.round(probs[i], 2)
            results.append([' '.join(s.raw), predictions])

        return results

    # TODO(kenlimmj): Consider making this a decorator, since it's used by both
    # `predict_topics` and `predict_trust`.
    def __preprocess(self, input, strip_html=True):
        """Cleans and tokenizes an input string for trust prediction.

        Performs the follow operations, in order:
        1. Strips HTML tags, if requested.
        2. Removes non alphanumeric characters.
        3. Converts to lowercase.
        4. Tokenizes on whitespace.
        5. Removes stopwords (see `constants.STOPWORDS`).
        6. Lemmatizes each token.

        Args:
            input: The string to be pre-processed.
            strip_html: Whether HTML tags in the input should be stripped. True
                by default, but can be disabled for speed if the input is known
                to be sanitized.

        Returns:
            A list of AirProfile.SentenceToken.
        """
        stripped_html = BeautifulSoup(
            input, 'lxml').get_text() if strip_html else input

        sentences = [
            str(s.decode('utf-8')).translate(None, NON_ALNUM_CHARS).lower()
            for s in sent_tokenize(stripped_html)
        ]

        output = []
        for s in sentences:
            tokens = s.split()
            tokens_stop = [t for t in tokens if t not in STOPWORDS]

            tokens_lemm_verb = [self.__lemm_v[t] for t in tokens_stop]
            tokens_lemm_noun = [self.__lemm_n[t] for t in tokens_lemm_verb]

            output.append(self.__SentenceToken(tokens, tokens_lemm_noun))

        return output

    @staticmethod
    def __get_trust_model_feat_cols(wc):
        """Gets the LIWC features to be evaluated based on word count."""
        feat_cols = FEAT_WC_LING + FEAT_WC_PSYCH + FEAT_WC_CONCERN
        if wc > 19:
            feat_cols += FEAT_WC_CATEGORIES
        return feat_cols

    @staticmethod
    def __get_trust_model_fname(wc):
        """Gets the trust model to be used based on word count."""
        if wc <= 19:
            return 'trust0.pkl'
        elif wc <= 36:
            return 'trust1.pkl'
        elif wc <= 58:
            return 'trust2.pkl'
        elif wc <= 88:
            return 'trust3.pkl'
        else:
            return 'trust4.pkl'
