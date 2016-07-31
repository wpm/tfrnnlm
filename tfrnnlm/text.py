import re
from collections import Counter
from itertools import takewhile


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
