import math

import numpy as np


class DocumentSet(object):
    """
    Set of documents used to train or evaluate a language model.

    Documents are sequences of integers. (A text document must by indexed before being used here.) Each document is
    treated as an independent Markov process.

    The epoch member function enumerates over the data in the documents, returning batches suitable for training an RNN.
    """

    def __init__(self, documents):
        """
        :param documents: documents in this data set
        :type documents: list of lists, where the inner lists can be converted to numpy.array objects
        """
        self.documents = [np.array(document) for document in documents]

    def __len__(self):
        """
        :return: number of tokens in the document set
        :rtype: int
        """
        return sum(len(document) for document in self.documents)

    def __str__(self):
        return "%d documents, %d tokens" % (len(self.documents), len(self))

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
        total_batches = self._total_batches(time_steps, batch_size)
        for document in self.documents:
            start_document = True
            for x, y in self.language_model_batches(document, time_steps, batch_size):
                batches += 1
                yield start_document, x, y, batches / total_batches
                start_document = False

    def _total_batches(self, time_steps, batch_size):
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
