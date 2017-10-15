from collections import Counter
from itertools import chain
from pathlib import Path

import ujson

from constants import (LIWC_CATEGORY_KEYS, LIWC_META_KEYS, LIWC_PUNCT,
                       LIWC_PUNCT_KEYS)


class LiwcUtil:
    """A utility class for working with LIWC 2007."""
    __liwc_trie_internal__ = None

    def __init__(self, liwc_trie_path):
        self.__liwc_trie_path = liwc_trie_path if isinstance(
            liwc_trie_path, Path) else Path(liwc_trie_path)

        self.__norm_keys = set(LIWC_META_KEYS + LIWC_CATEGORY_KEYS +
                               LIWC_PUNCT_KEYS)

        # Drop 'WC' and 'WPS', because normalization occurs across sentences
        # and should intentionally not touch those fields.
        self.__norm_keys.discard('WC')
        self.__norm_keys.discard('WPS')

        # Exclude 'filler' to avoid skewing the normalization counts.
        self.__norm_keys.discard('filler')

    @property
    def __liwc_trie(self):
        """Lazily loads the LIWC trie from disk."""
        if not self.__liwc_trie_internal__:
            with self.__liwc_trie_path.open() as liwc_trie:
                self.__liwc_trie_internal__ = ujson.load(liwc_trie)
        return self.__liwc_trie_internal__

    def summarize(self, input, sent_tokens, normalize=True):
        """Constructs a summary of LIWC attributes.

        Args:
            input: A string to summarize.
            sent_tokens: A list of AirProfile.SentenceToken corresponding to
                the input.
            normalize: Whether the word frequency counts should be normalized
                to unit density. True by default.

        Returns:
            A dictionary of LIWC attributes to scores.
        """
        raw_tokens = list(chain.from_iterable([t.raw for t in sent_tokens]))

        counts = Counter(self.__read_tokens(raw_tokens))
        counts['Dic'] = sum(counts.values())
        counts['WC'] = len(raw_tokens)
        counts['WPS'] = counts['WC'] / float(len(sent_tokens))
        counts['Sixltr'] = sum(len(t) > 6 for t in raw_tokens)
        counts['Numerals'] = sum(t.isdigit() for t in raw_tokens)

        character_counts = Counter(input)
        for name, chars in LIWC_PUNCT:
            counts[name] = sum(character_counts[char] for char in chars)
        counts['Parenth'] = counts['Parenth'] / 2.0
        counts['AllPct'] = sum(counts[name] for name, _ in LIWC_PUNCT)

        if normalize:
            for column in self.__norm_keys:
                counts[column] = float(counts[column]) / float(counts['WC'])

        result = dict.fromkeys(LIWC_CATEGORY_KEYS + ['Dic'], 0)
        result.update(counts)
        return result

    def __walk_trie(self, token, token_i=0, trie_cursor=None):
        """Walks the LIWC trie.

        Args:
            token: An AirProfile.SentenceToken to be found in the trie.
            token_i: The index of the token in the sentence. Defaults to 0, the
                start of the sentence.
            trie_cursor: Specifies the location in the trie to start walking
                from. If not specified, starts from the root of the trie.

        Returns:
            The LIWC category for the word found in the trie.
        """
        if trie_cursor is None:
            trie_cursor = self.__liwc_trie

        if '*' in trie_cursor:
            for cat in trie_cursor['*']:
                yield cat
        elif '$' in trie_cursor and token_i == len(token):
            for cat in trie_cursor['$']:
                yield cat
        elif token_i < len(token):
            letter = token[token_i]
            if letter in trie_cursor:
                new_cursor = trie_cursor[letter]
                for cat in self.__walk_trie(token, token_i + 1, new_cursor):
                    yield cat

    def __read_tokens(self, tokens):
        """Walks the LIWC trie for a set of tokens."""
        for t in tokens:
            for cat in self.__walk_trie(t):
                yield cat
