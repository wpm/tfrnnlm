import collections
import textwrap
from unittest import TestCase

import numpy as np
from tfrnnlm.document_set import language_model_batches
from tfrnnlm.prepare_data import vocabulary_from_documents
from tfrnnlm.text import IndexedVocabulary, WhitespaceWordTokenization, PennTreebankTokenization


class TestTokenization(TestCase):
    def test_whitespace_word_tokenization(self):
        tokens = WhitespaceWordTokenization()("\nCall me Ishmael. Some years ago--never mind how long precisely  ")
        self.assertEqual(tokens,
                         ["call", "me", "ishmael", "some", "years", "ago", "never", "mind", "how", "long", "precisely"])

    def test_penn_treebank_word_tokenization(self):
        s = textwrap.dedent("""seoul also has instituted effective <unk> procedures to aid these teams she said
taiwan has improved""")
        actual = PennTreebankTokenization()(s)
        self.assertIsInstance(actual, collections.Iterable)
        expected = ["seoul", "also", "has", "instituted", "effective", "<unk>", "procedures", "to", "aid", "these",
                    "teams", "she", "said", "<eos>", "taiwan", "has", "improved", "<eos>"]
        self.assertEqual(list(actual), expected)


class TestIndexing(TestCase):
    def test_full_vocabulary(self):
        v = IndexedVocabulary("the quick brown fox jumped over the lazy dog".split())
        self.assertEqual(set(v.type_to_index.keys()), {"the", "quick", "brown", "fox", "jumped", "over", "lazy", "dog"})
        self.assertEqual(len(v), 9)

    def test_limited_vocabulary(self):
        v = IndexedVocabulary("to be or not to be".split(), max_vocabulary=2)
        self.assertEqual(set(v.type_to_index.keys()), {"to", "be"})
        self.assertEqual(len(v), 3)
        v = IndexedVocabulary("hamlet hamlet hamlet to be or not to be".split(), min_frequency=2)
        self.assertEqual(set(v.type_to_index.keys()), {"to", "be", "hamlet"})
        self.assertEqual(len(v), 4)
        v = IndexedVocabulary("hamlet hamlet hamlet to be or not to be".split(), max_vocabulary=2, min_frequency=2)
        self.assertEqual(set(v.type_to_index.keys()), {"be", "hamlet"})
        self.assertEqual(len(v), 3)

    def test_index_tokens(self):
        tokens = "the quick brown fox jumped over the lazy dog".split()
        v = IndexedVocabulary(tokens)
        indexes = v.index_tokens(tokens)
        self.assertIsInstance(indexes, collections.Iterable)
        self.assertEqual(list(indexes), [1, 8, 2, 4, 5, 7, 1, 6, 3])

    def test_out_of_vocabulary(self):
        tokens = "a a a b b OOV c".split()
        v = IndexedVocabulary(tokens, out_of_vocabulary="OOV")
        self.assertEqual(v.index("OOV"), 0)
        self.assertEqual(v.index("a"), 1)
        self.assertEqual(v.index("b"), 2)
        self.assertEqual(v.index("c"), 3)

    def test_vocabulary_from_documents(self):
        doc1 = "cat dog horse"
        doc2 = "cat cat mouse"
        doc3 = "whale dog zebra"
        v = vocabulary_from_documents([doc1, doc2, doc3], WhitespaceWordTokenization(), IndexedVocabulary.factory())
        self.assertEqual(v.index("cat"), 1)
        self.assertEqual(v.index("dog"), 2)
        self.assertEqual(v.index("horse"), 3)
        self.assertEqual(v.index("mouse"), 4)
        self.assertEqual(v.index("whale"), 5)
        self.assertEqual(v.index("zebra"), 6)


class TestBatches(TestCase):
    def test_batches(self):
        batches = language_model_batches(np.arange(20), time_steps=3, batch_size=4)
        self.assertIsInstance(batches, collections.Iterable)
        np.testing.assert_equal(list(batches),
                                [
                                    # Batch 0
                                    (np.array([[0, 1, 2],
                                               [3, 4, 5],
                                               [6, 7, 8],
                                               [9, 10, 11]]),
                                     np.array([[1, 2, 3],
                                               [4, 5, 6],
                                               [7, 8, 9],
                                               [10, 11, 12]])),
                                    # Batch 1
                                    (np.array([[12, 13, 14],
                                               [15, 16, 17],
                                               [18, 19, 0],
                                               [0, 0, 0]]),
                                     np.array([[13, 14, 15],
                                               [16, 17, 18],
                                               [19, 0, 0],
                                               [0, 0, 0]]))
                                ])
