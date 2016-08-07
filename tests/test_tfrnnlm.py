import collections
import textwrap
from unittest import TestCase

import numpy as np
from tfrnnlm.prepare_data import vocabulary_from_documents
from tfrnnlm.text import IndexedVocabulary, language_model_batches, WhitespaceWordTokenization, PennTreebankTokenization


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
        lm_batches = list(language_model_batches([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], time_steps=3, batch_size=4))
        self.assertEquals(len(lm_batches), 3)
        # Batch 0
        np.testing.assert_equal(lm_batches[0][0], np.array([[0, 1, 2],
                                                            [1, 2, 3],
                                                            [2, 3, 4],
                                                            [3, 4, 5]]
                                                           ))
        np.testing.assert_equal(lm_batches[0][1], np.array([[1, 2, 3],
                                                            [2, 3, 4],
                                                            [3, 4, 5],
                                                            [4, 5, 6]]
                                                           ))
        # Batch 1
        np.testing.assert_equal(lm_batches[1][0], np.array([[4, 5, 6],
                                                            [5, 6, 7],
                                                            [6, 7, 8],
                                                            [7, 8, 9]]
                                                           ))
        np.testing.assert_equal(lm_batches[1][1], np.array([[5, 6, 7],
                                                            [6, 7, 8],
                                                            [7, 8, 9],
                                                            [8, 9, 0]]))
        # Batch 2
        np.testing.assert_equal(lm_batches[2][0], np.array([[8, 9, 0],
                                                            [9, 0, 0],
                                                            [0, 0, 0],
                                                            [0, 0, 0]]
                                                           ))
        np.testing.assert_equal(lm_batches[2][1], np.array([[9, 0, 0],
                                                            [0, 0, 0],
                                                            [0, 0, 0],
                                                            [0, 0, 0]]))
