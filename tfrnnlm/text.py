"""
Utilities for converting text files into arrays of integers that can be handled by recursive neural networks.
"""

import glob
import itertools
import math
import os
import pickle
import re
from collections import Counter

import numpy as np


class Tokenizer(object):
    """
    Abstract tokenizer class

    The default method of a tokenizer maps a string to a sequence of string tokens.
    The join_tokens method converts a sequence of tokens back into a single string. Depending on the tokenizer, some
    token separate information may be lost, so the two methods are not guaranteed to be exact inverses of each other.
    """

    def __init__(self, lowercase):
        self.lowercase = lowercase

    def _case_normalize(self, text):
        if self.lowercase:
            text = text.lower()
        return text

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.lowercase == other.lowercase

    def __call__(self, text):
        raise NotImplementedError()

    @staticmethod
    def join_tokens(tokens):
        return " ".join(tokens)


class WhitespaceTokenizer(Tokenizer):
    def __init__(self, lowercase=False):
        super(WhitespaceTokenizer, self).__init__(lowercase)

    def __call__(self, text):
        return self._case_normalize(text).strip().split()


class WordTokenizer(Tokenizer):
    """
    Tokenize on whitespace, dropping punctuation and optionally normalizing to lower-case, e.g.

        Call me Ishmael. Some years ago--never mind how long precisely

    goes to

        call me ishmael some years ago never mind how long precisely
    """

    def __init__(self, lowercase=False):
        super(WordTokenizer, self).__init__(lowercase)

    def __call__(self, text):
        return re.sub("[^\w]", " ", self._case_normalize(text)).strip().split()


class CharacterTokenizer(Tokenizer):
    """
    Break text into characters, retaining whitespace and optionally normalizing to lower-case.
    """

    def __init__(self, lowercase=False):
        super(CharacterTokenizer, self).__init__(lowercase)

    def __call__(self, text):
        return list(self._case_normalize(text))

    @staticmethod
    def join_tokens(tokens):
        return "".join(tokens)


class Vocabulary(object):
    """
    A mapping of string types to integer indexes.

    Indexing is 2-based because index 0 is reserved for padding buffers. 1 is reserved as a special out-of-vocabulary
    index.
    """
    OUT_OF_VOCABULARY = 1

    @classmethod
    def factory(cls, tokenizer, min_frequency=None, max_vocabulary=None, out_of_vocabulary=None):
        return lambda documents: cls(documents, tokenizer, min_frequency, max_vocabulary, out_of_vocabulary)

    def __init__(self, documents, tokenizer, min_frequency=None, max_vocabulary=None, out_of_vocabulary=None):
        """
        :param documents: set of documents
        :type documents: iterator of str
        :param tokenizer: tokenizer
        :type tokenizer: Tokenizer
        :param min_frequency: optional minimum token frequency for inclusion in the vocabulary
        :type min_frequency: int or None
        :param max_vocabulary: optional maximum vocabulary size
        :type max_vocabulary: int or None
        :param out_of_vocabulary optional out of vocabulary type
        :type out_of_vocabulary str or None
        """
        self.tokenizer = tokenizer
        self.out_of_vocabulary = out_of_vocabulary
        # Sort in descending order by frequency and then by token so that the composition of the vocabulary is
        # deterministic.
        tokens = itertools.chain(*(tokenizer(document) for document in documents))
        if self.out_of_vocabulary is not None:
            tokens = (token for token in tokens if token != out_of_vocabulary)
        types = sorted(Counter(tokens).items(), key=lambda t: (-t[1], t[0]))
        if min_frequency is not None:
            types = list(itertools.takewhile(lambda t: t[1] >= min_frequency, types))
        types = types[:max_vocabulary]
        self.index_to_type = dict(enumerate((t for t, _ in types), 2))
        self.type_to_index = dict(map(reversed, self.index_to_type.items()))

    def __eq__(self, other):
        return self.tokenizer == other.tokenizer and \
               self.out_of_vocabulary == other.out_of_vocabulary and \
               self.index_to_type == other.index_to_type and \
               self.type_to_index == other.type_to_index

    def __repr__(self):
        return "Vocabulary, size %d" % len(self)

    def __str__(self):
        n = min(6, len(self))
        items = ["%s:%d" % (self.type(i), i) for i in range(1, n)]
        s = "%s: %s" % (repr(self), " ".join(items))
        if len(self) > 6:
            s += " ..."
        return s

    def __len__(self):
        return len(self.type_to_index)

    # noinspection PyShadowingBuiltins
    def index(self, type):
        """
        :param type: a type listed in the vocabulary
        :type type: str
        :return: index of the type
        :rtype: int
        """
        return self.type_to_index.get(type, Vocabulary.OUT_OF_VOCABULARY)

    def type(self, index):
        """
        :param index: index of a type in the vocabulary
        :type index: int
        :return: the type
        :rtype: str
        """
        if index < 1:
            raise ValueError("Index %d must be greater than zero" % index)
        return self.index_to_type.get(index, self.out_of_vocabulary)

    def index_string(self, s):
        """
        Tokenize and index a string.

        :param s: string to convert to indexes
        :type s: str
        :return: the token indexes
        :rtype: numpy.array of int
        """
        return np.array([self.index(token) for token in self.tokenizer(s)])


class PartitionedData(object):
    """
    Sets of documents used to train or evaluate a language model.

    Documents are sequences of integers. (A text document must by indexed before being used here.) Each document is
    treated as an independent Markov process. Documents are divided into partitions (e.g. train, validate, test).

    The epoch member function enumerates over the data in the documents, returning batches suitable for training an RNN.
    """
    VOCABULARY = "vocabulary.pkl"

    def __init__(self, vocabulary, document_partitions):
        """
        :param vocabulary: the vocabulary used to index the documents in the partitions
        :type vocabulary: Vocabulary
        :param document_partitions: map partition names to partitions
        :type document_partitions: map of str to DocumentSet
        """
        self.vocabulary = vocabulary
        self.document_partitions = document_partitions

    def __repr__(self):
        return "Partitioned Data: %s" % ", ".join(self)

    def __str__(self):
        def partition_20_tokens(partition):
            types = filter(None, [self.vocabulary.type(token) for token in self[partition].documents[0][:20]])
            return self.vocabulary.tokenizer.join_tokens(types)

        partitions = ["%s: %s\n%s ...\n" % (partition,
                                            self[partition],
                                            partition_20_tokens(partition))
                      for partition in self]
        return "%s\n%s" % (self.vocabulary, "\n".join(partitions))

    def __eq__(self, other):
        return self.vocabulary == other.vocabulary and self.document_partitions == other.document_partitions

    def __getitem__(self, partition):
        return self.document_partitions[partition]

    def __iter__(self):
        """
        :return: partition names ordered by number of tokens and then name
        :rtype: iterator of str
        """
        return iter(sorted(self.document_partitions.keys(), key=lambda p: (-len(self[p]), p)))

    @classmethod
    def from_text(cls, text_partitions, vocabulary_partitions, vocabulary_factory):
        # Ensure all vocabulary partitions reference text partitions.
        bad_vocabulary_partitions = \
            [partition for partition in vocabulary_partitions if partition not in text_partitions.keys()]
        if bad_vocabulary_partitions:
            raise ValueError("Vocabulary partitions %s do not appear in the data set"
                             % ",".join(bad_vocabulary_partitions))
        document_lists = [text_partitions[partition] for partition in text_partitions
                          if partition in vocabulary_partitions]
        documents = itertools.chain(*document_lists)
        vocabulary = vocabulary_factory(documents)
        document_partitions = {}
        for partition, documents in text_partitions.items():
            document_partitions[partition] = DocumentSet(
                [vocabulary.index_string(document) for document in sorted(documents)])
        return cls(vocabulary, document_partitions)

    def serialize(self, directory):
        with open(os.path.join(directory, PartitionedData.VOCABULARY), "wb") as f:
            pickle.dump(self.vocabulary, f)
        for partition, documents in self.document_partitions.items():
            partition_file = os.path.join(directory, partition)
            documents.serialize(partition_file)

    @classmethod
    def deserialize(cls, directory):
        with open(os.path.join(directory, PartitionedData.VOCABULARY), "rb") as f:
            vocabulary = pickle.load(f)
        document_partitions = {}
        for partition_file in glob.glob(os.path.join(directory, "*.npy")):
            partition = os.path.basename(partition_file).split(".")[0]
            document_partitions[partition] = DocumentSet.deserialize(partition_file)
        return cls(vocabulary, document_partitions)


class DocumentSet(object):
    """
    Set of documents used to train or evaluate a language model.

    Documents are numpy.array objects. Each is treated as an independent Markov process.

    The epoch member function enumerates over the data in the documents, returning batches suitable for training an RNN.
    """

    def __init__(self, documents):
        """
        :param documents: documents in this data set
        :type documents: sequence of numpy.array
        """
        self.documents = documents

    def serialize(self, filename):
        np.save(filename, self.documents)

    @classmethod
    def deserialize(cls, filename):
        return cls(np.load(filename))

    def __len__(self):
        """
        :return: number of tokens in the document set
        :rtype: int
        """
        return sum(len(document) for document in self.documents)

    def __str__(self):
        return "%d documents, %d tokens" % (len(self.documents), len(self))

    def __iter__(self):
        return iter(self.documents)

    def __eq__(self, other):
        # numpy.array objects are unhashable, so convert them into tuples.
        return set(map(tuple, self.documents)) == set(map(tuple, other.documents))

    def epoch(self, time_steps, batch_size):
        """
        Iterator over a single epoch of the document set.

        :param time_steps: number of time steps to unroll
        :type time_steps: int
        :param batch_size: number of unrolled sequences to combine into a single batch
        :type batch_size: int
        :return: whether this batch is the start of a new document, the context and targets, and the portion of the
        epoch completed
        :rtype: iterator over (bool, numpy.array, numpy.array, float)
        """
        batches = 0
        total_batches = self.total_batches(time_steps, batch_size)
        for document in self.documents:
            start_document = True
            for x, y in self.language_model_batches(document, time_steps, batch_size):
                batches += 1
                yield start_document, x, y, batches / total_batches
                start_document = False

    def total_batches(self, time_steps, batch_size):
        return sum(math.ceil(len(document) / (time_steps * batch_size)) for document in self.documents)

    @staticmethod
    def language_model_batches(data, time_steps, batch_size):
        """
        Arranges a sequence of data into a form for use in batched language model training. This returns pairs of
        arrays. The first is the language model context, the second is the context. The data is mapped into the context
        in order in arrays with the shape (batch_size x time_steps). The target is the same shifted ahead by a single
        time step. The end of the data is padded with zeros as necessary.

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
        # Context and targets are arrays of shape (total_batches x batch_size x time_steps).
        xs = padded_data[:-1].reshape(instances // batch_size, batch_size, time_steps)
        ys = padded_data[1:].reshape(instances // batch_size, batch_size, time_steps)
        # Enumerator over the batches.
        return zip(xs, ys)
