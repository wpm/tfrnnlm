import re
from collections import Counter
from itertools import takewhile

import numpy as np


def whitespace_word_tokenization(text):
    """
    Tokenize on whitespace, dropping punctuation and normalizing to lower-case, e.g.

        Call me Ishmael. Some years ago--never mind how long precisely...

    goes to

        call me ishmael some years ago never mind how long precisely...

    :param text: text to tokenize
    :type text: str
    :return: tokens in the text file
    :rtype: [str]
    """
    return re.sub("[^\w]", " ", text.lower()).strip().split()


class IndexedVocabulary(object):
    """
    Given a sequence of tokens, create a unique integer mapping from an integer to a type. Out of vocabulary types map
    to a default value of zero.

    If a minimum token frequency is specified, all tokens with a lower frequency are mapped to out of vocabulary. If a
    maximum vocabulary size is specified, only the most frequent types will be indexed.
    """

    def __init__(self, tokens, min_frequency=None, max_vocabulary=None):
        """
        :param tokens: sequence of natural language tokens
        :type tokens: iterator of str
        :param min_frequency: minimum token frequency for inclusion in the vocabulary
        :type min_frequency: int or None
        :param max_vocabulary: maximum vocabulary size
        :type max_vocabulary: int or None
        """
        # Sort in descending order by frequency and then by token so that the composition of the vocabulary is
        # deterministic.
        types = sorted(Counter(tokens).items(), key=lambda t: (-t[1], t[0]))
        if min_frequency is not None:
            types = list(takewhile(lambda t: t[1] >= min_frequency, types))
        types = types[:max_vocabulary]
        self.index_to_type = dict(enumerate((t for t, _ in types), 1))
        self.type_to_index = dict(map(reversed, self.index_to_type.items()))

    def __repr__(self):
        return "Indexed Vocabulary, size %d" % len(self)

    def __str__(self):
        return "%s: %s ..." % (
            repr(self), " ".join("%s:%d" % (t, i) for i, t in sorted(self.index_to_type.items())[:5]))

    def __len__(self):
        # Add 1 for the out of vocabulary type.
        return len(self.type_to_index) + 1

    def index(self, type):
        """
        :param type: a type listed in the vocabulary
        :type type: str
        :return: index of the type
        :rtype: int
        """
        return self.type_to_index.get(type, 0)

    def type(self, index):
        """
        :param index: index of a type in the vocabulary
        :type index: int
        :return: the type
        :rtype: str
        """
        return self.index_to_type.get(index, None)


def language_model_batches(data, batch_size):
    """Create batched language model data with padding.

    This returns two arrays of size total_time x batch_size. The first is the inputs to the model and the second are
    the expected targets.

    >>> language_model_batches([79, 90, 136, 33, 5, 137], 3)
    (array([[ 79,  90, 136],
       [ 90, 136,  33],
       [136,  33,   5],
       [ 33,   5, 137],
       [  5, 137,   0],
       [137,   0,   0]]),
    array([[ 90, 136,  33],
       [136,  33,   5],
       [ 33,   5, 137],
       [  5, 137,   0],
       [137,   0,   0],
       [  0,   0,   0]]))
    """
    total_time = len(data)

    def batches():
        """Enumerate all data sequences of length batch_size in order, padding the end with zeros as necessary"""
        for i in range(total_time):
            batch = data[i:i + batch_size]
            batch = np.pad(batch, (0, batch_size - len(batch)), mode="constant")
            yield batch

    # Concatenate the batches into a single array to get the input.
    batched_input = np.concatenate([batch for batch in batches()]).reshape(total_time, batch_size)
    # Shift the input one time point forward and pad the end with zeroes to get the targets.
    batched_targets = np.roll(batched_input, -1, axis=0)
    batched_targets[-1] = 0
    return batched_input, batched_targets


def batches(data, time_steps, batch_size):
    total_time = len(data)

    def unrolled_sequence_pairs():
        def unrolled_sequences():
            for i in range(total_time):
                batch = data[i:i + time_steps]
                batch = np.pad(batch, (0, time_steps - len(batch)), mode="constant")
                yield batch

        prev = None
        for unrolled_sequence in unrolled_sequences():
            if prev is not None:
                yield prev, unrolled_sequence
            prev = unrolled_sequence
        pad = np.zeros(time_steps, dtype=int)
        yield prev, pad
        extra = total_time % batch_size
        for _ in range(extra):
            yield pad, pad

    contexts = []
    targets = []
    for context, target in unrolled_sequence_pairs():
        contexts += [context]
        targets += [target]
        if len(contexts) == batch_size:
            contexts = np.concatenate(contexts).reshape(-1, time_steps)
            targets = np.concatenate(targets).reshape(-1, time_steps)
            yield contexts, targets
            contexts = []
            targets = []
