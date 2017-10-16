"""Provides the AirProfile class for analyzing Airbnb host profiles."""

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

from .constants import (FEAT_WC_CATEGORIES, FEAT_WC_CONCERN, FEAT_WC_LING,
                        FEAT_WC_PSYCH, NON_ALNUM_CHARS, STOPWORDS)
from .LiwcUtil import LiwcUtil


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

    def predict_trust(self, profile, strip_html=True):
        """Predicts the trustworthiness of a profile.

        Segments the input with sentence-level granularity, returning the
        probability that the profile represented by the input is perceived to
        be more trustworthy compared to other profiles of similar length.

        Args:
            profile: An Airbnb host profile, as a string.
            strip_html: Whether HTML tags in the input should be stripped. True
                by default, but can be disabled for speed if the input is known
                to be sanitized.

        Returns:
            An AirProfile.Prediction object for trustworthiness of the profile.

        Raises:
            ValueError: If the input is an invalid or empty string.
            IOError: If the LIWC trie is not available.
        """
        if not (profile and profile.strip()):
            raise ValueError

        if not (self.__liwc_path.exists() and self.__liwc_path.is_file()):
            raise IOError

        sentence_tokens = self.__preprocess(profile, strip_html)
        liwc_features = self.__liwc.summarize(profile, sentence_tokens)

        word_count = liwc_features['WC']
        liwc_features['wc_log'] = np.log(word_count)
        liwc_features['readability'] = ts.flesch_kincaid_grade(
            profile.decode('utf-8'))

        prediction_agg = np.empty(len(self.__classifier_cat))
        for sent in sentence_tokens:
            prediction_agg += np.array(
                [c.predict for c in self.__classify_sentence(sent)])

        for idx, cat in enumerate(FEAT_WC_CATEGORIES):
            liwc_features[cat] = prediction_agg[idx]

        feats = [
            liwc_features[f]
            for f in AirProfile.__get_trust_model_feat_cols(word_count)
        ]
        feats_shape = np.array(feats).reshape(1, -1)
        model = self.__get_classifier_trust(
            AirProfile.__get_trust_model_fname(word_count))

        return self.Prediction(
            np.round(model.predict_proba(feats_shape)[0][1], 2),
            model.predict(feats_shape)[0])

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

    def predict_topics(self, profile, strip_html=True):
        """Predicts the trust evaluation topics for a profile.

        Args:
            profile: An Airbnb host profile, as a string.
            strip_html: Whether HTML tags in the input should be stripped. True
                by default, but can be disabled for speed if the input is known
                to be sanitized.

        Returns:
            A list of prediction probabilities

        Raises:
            ValueError: If the input is an invalid or empty string.
        """
        if not (profile and profile.strip()):
            raise ValueError

        sentence_tokens = self.__preprocess(profile, strip_html)

        results = []
        for token in sentence_tokens:
            probs = np.array(
                [c.prob[1] for c in self.__classify_sentence(token)])
            predictions = {}
            for idx, cat in enumerate(FEAT_WC_CATEGORIES):
                predictions[cat] = np.round(probs[idx], 2)
            results.append([' '.join(token.raw), predictions])

        return results

    # TODO(kenlimmj): Consider making this a decorator, since it's used by both
    # `predict_topics` and `predict_trust`.
    def __preprocess(self, profile, strip_html=True):
        """Cleans and tokenizes an input string for trust prediction.

        Performs the follow operations, in order:
        1. Strips HTML tags, if requested.
        2. Removes non alphanumeric characters.
        3. Converts to lowercase.
        4. Tokenizes on whitespace.
        5. Removes stopwords (see `constants.STOPWORDS`).
        6. Lemmatizes each token.

        Args:
            profile: The string to be pre-processed.
            strip_html: Whether HTML tags in the input should be stripped. True
                by default, but can be disabled for speed if the input is known
                to be sanitized.

        Returns:
            A list of AirProfile.SentenceToken.
        """
        stripped_html = BeautifulSoup(
            profile, 'lxml').get_text() if strip_html else profile

        sentences = [
            str(s.decode('utf-8')).translate(None, NON_ALNUM_CHARS).lower()
            for s in sent_tokenize(stripped_html)
        ]

        output = []
        for sent in sentences:
            tokens = sent.split()
            tokens_stop = [t for t in tokens if t not in STOPWORDS]

            tokens_lemm_verb = [self.__lemm_v[t] for t in tokens_stop]
            tokens_lemm_noun = [self.__lemm_n[t] for t in tokens_lemm_verb]

            output.append(self.__SentenceToken(tokens, tokens_lemm_noun))

        return output

    @staticmethod
    def __get_trust_model_feat_cols(word_count):
        """Gets the LIWC features to be evaluated based on word count."""
        feat_cols = FEAT_WC_LING + FEAT_WC_PSYCH + FEAT_WC_CONCERN
        if word_count > 19:
            feat_cols += FEAT_WC_CATEGORIES
        return feat_cols

    @staticmethod
    def __get_trust_model_fname(word_count):
        """Gets the trust model to be used based on word count."""
        if word_count <= 19:
            return 'trust0.pkl'
        elif word_count <= 36:
            return 'trust1.pkl'
        elif word_count <= 58:
            return 'trust2.pkl'
        elif word_count <= 88:
            return 'trust3.pkl'
        return 'trust4.pkl'
