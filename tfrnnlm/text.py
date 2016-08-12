import re
from collections import Counter
from itertools import takewhile

import numpy as np


class WhitespaceWordTokenization(object):
    """
    Tokenize on whitespace, dropping punctuation and normalizing to lower-case, e.g.

        Call me Ishmael. Some years ago--never mind how long precisely...

    goes to

        call me ishmael some years ago never mind how long precisely...
    """

    def __str__(self):
        return "Whitespace Word Tokenization"

    def __call__(self, text):
        return re.sub("[^\w]", " ", text.lower()).strip().split()


class PennTreebankTokenization(object):
    """
    Tokenize Penn Treebank Data

    Split on whitespace and insert an <eos> symbol at the end of each line.

    Penn Treebank uses <unk> to mark unknown words. Specify this as the optional out of vocabulary token when creating
    an IndexedVocabulary.
    """

    def __str__(self):
        return "Penn Treebank Tokenization"

    def __call__(self, text):
        for line in text.split("\n"):
            for token in line.strip().split():
                yield token
            yield "<eos>"


class IndexedVocabulary(object):
    """
    Given a sequence of tokens, create a unique integer mapping from an integer to a type. Out of vocabulary types map
    to the index zero. By default, the out of vocabulary type is None. Optionally, a string can be supplied (e.g.
    "<UNK>") if the text already has out of vocabulary types marked.

    If a minimum token frequency is specified, all tokens with a lower frequency are mapped to out of vocabulary. If a
    maximum vocabulary size is specified, only the most frequent types will be indexed.
    """

    @classmethod
    def factory(cls, min_frequency=None, max_vocabulary=None, out_of_vocabulary=None):
        return lambda tokens: cls(tokens, min_frequency, max_vocabulary, out_of_vocabulary)

    def __init__(self, tokens, min_frequency=None, max_vocabulary=None, out_of_vocabulary=None):
        """
        :param tokens: sequence of natural language tokens
        :type tokens: iterator of str
        :param min_frequency: minimum token frequency for inclusion in the vocabulary
        :type min_frequency: int or None
        :param max_vocabulary: maximum vocabulary size
        :type max_vocabulary: int or None
        :param out_of_vocabulary out of vocabulary type
        :type out_of_vocabulary str or None
        """
        # Sort in descending order by frequency and then by token so that the composition of the vocabulary is
        # deterministic.
        self.out_of_vocabulary = out_of_vocabulary
        if self.out_of_vocabulary is not None:
            tokens = (token for token in tokens if token != out_of_vocabulary)
        types = sorted(Counter(tokens).items(), key=lambda t: (-t[1], t[0]))
        if min_frequency is not None:
            types = list(takewhile(lambda t: t[1] >= min_frequency, types))
        types = types[:max_vocabulary]
        self.index_to_type = dict(enumerate((t for t, _ in types), 1))
        self.type_to_index = dict(map(reversed, self.index_to_type.items()))

    def __repr__(self):
        return "Indexed Vocabulary, size %d" % len(self)

    def __str__(self):
        n = min(5, len(self))
        items = ["%s:%d" % (self.type(i), i) for i in range(n)]
        s = "%s: %s" % (repr(self), " ".join(items))
        if len(self) > 5:
            s += " ..."
        return s

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
        return self.index_to_type.get(index, self.out_of_vocabulary)

    def index_tokens(self, tokens):
        """
        Convert a sequence of tokens to their corresponding indexes

        :param tokens: sequence of tokens
        :type tokens: iterable of str
        :return: the token indexes
        :rtype: iterable of int
        """
        return (self.index(token) for token in tokens)


def language_model_batches(data, time_steps, batch_size):
    """
    Yield pairs of minibatches of the data where minibatches are arrays with the shape (batch_size x time_steps). The
    second element of the pair is equal to the first one shifted ahead one place. Pad with zeros as necessary.

    Each minibatch may be used as input for tf.nn.dynamic_rnn.
    """
    total_time = len(data)

    def unrolled_sequence_pairs():
        """
        Yield a pair of sequences. The first element of the pair subsequence of the data of length time_steps in order.
        The second element is the first element shifted ahead by one. Pad with zeros as necessary.

        e.g. data = [1, 2, 3], time_steps = 2 yields

            ([1, 2], [2, 3])
            ([2, 3], [3, 0])
            ([3, 0], [0, 0])
        """

        def unrolled_sequences():
            """
            Yield every subsequence of the data of length time_steps in order, padding the end with zeros.

            e.g. data = [1, 2, 3], time_steps = 2 yields [1, 2], [2, 3], [3, 0]
            """
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
