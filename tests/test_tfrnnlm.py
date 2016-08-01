from unittest import TestCase

import numpy as np
from tfrnnlm.text import IndexedVocabulary, whitespace_word_tokenization, language_model_batches_old, language_model_batches


class TestTokenization(TestCase):
    def test_whitespace_word_tokenization(self):
        tokens = whitespace_word_tokenization("\nCall me Ishmael. Some years ago--never mind how long precisely  ")
        self.assertEqual(tokens,
                         ["call", "me", "ishmael", "some", "years", "ago", "never", "mind", "how", "long", "precisely"])


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


class TestBatching(TestCase):
    def test_language_model_batches(self):
        batched_input, batched_targets = language_model_batches_old([79, 90, 136, 33, 5, 137], 3)
        np.testing.assert_equal(batched_input, np.array([
            [79, 90, 136],
            [90, 136, 33],
            [136, 33, 5],
            [33, 5, 137],
            [5, 137, 0],
            [137, 0, 0]
        ]))
        np.testing.assert_equal(batched_targets, np.array([
            [90, 136, 33],
            [136, 33, 5],
            [33, 5, 137],
            [5, 137, 0],
            [137, 0, 0],
            [0, 0, 0]
        ]))


class TestBatches(TestCase):
    def test_batches(self):
        lm_batches = list(language_model_batches([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 4))
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
