import re
from collections import Counter
from itertools import takewhile

import numpy as np
from tfrnnlm import logger


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


def epochs(documents, time_steps, batch_size, max_epochs=None):
    """
    Enumerator over documents, yielding pairs of batches of the data where batches are arrays with the shape
    (batch_size x time_steps). The second element of the pair is equal to the first one shifted ahead one place.
    Pad with zeros at the end of each document as necessary.

    Documents are sequences of integers. By definition, the sequence of integers in one document is independent of the
    ones in the previous document.

    This will run for the specified number of epochs. If max_epochs is not specified, it will run forever.

    :param documents: documents to enumerate over
    :type documents: sequence of numpy.array of int
    :param time_steps: number of time steps to unroll
    :type time_steps: int
    :param batch_size: number of unrolled sequences to combine into a single batch
    :type batch_size: int
    :param max_epochs: maximum number of epochs
    :type max_epochs: int or None
    :return: epoch number, whether this is a new epoch and/or new document, batch context and target
    :rtype: iterator of (int, bool, bool, numpy.array, numpy.array)
    """
    logger.info("Epoch size %d" % sum(len(document) for document in documents))
    epoch = 1
    while True:
        if max_epochs is not None and epoch > max_epochs:
            break
        new_epoch = True
        for document in documents:
            new_document = True
            for context, target in language_model_batches(document, time_steps, batch_size):
                yield epoch, new_epoch, new_document, context, target
                new_epoch = False
                new_document = False
        epoch += 1


def language_model_batches(data, time_steps, batch_size):
    """
    Arranges a sequence of data into a form for use in batched language model training. This returns pairs of arrays.
    The first is the language model context, the second is the context. The data is mapped into the context in order
    in arrays with the shape (batch_size x time_steps). The target is the same shifted ahead by a single time step. The
    end of the data is padded with zeros as necessary.

    Each batch may be used as input for tf.nn.dynamic_rnn.

    :param data: data to emit as language model batches
    :type data: numpy.array of int
    :param time_steps: number of time steps to unroll
    :type time_steps: int
    :param batch_size: number of unrolled sequences to combine into a single batch
    :type batch_size: int
    :return: batches of contexts and their targets
    :rtype: iterator over (numpy.array, numpy.array)
    """
    # Divide the data up into batches of size time_steps * batch_size.
    n = len(data)
    m = time_steps * batch_size
    # Pad the end of the data with zeros to make its length a multiple of time_steps * batch_size, then add one
    # additional padding zero for the targets time shift.
    p = [(n + i) % m for i in range(m)].index(0)
    padded_data = np.pad(data, (0, (p + 1)), mode="constant")
    instances = np.math.ceil((len(padded_data) - 1) / time_steps)
    # Context and targets are arrays of shape (k x batch_size x time_steps).
    xs = padded_data[:-1].reshape(instances // batch_size, batch_size, time_steps)
    ys = padded_data[1:].reshape(instances // batch_size, batch_size, time_steps)
    # Enumerate over the batches.
    return zip(xs, ys)
