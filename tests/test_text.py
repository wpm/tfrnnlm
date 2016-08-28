import collections
import os
import pickle
import shutil
import tempfile
from unittest import TestCase

import numpy as np
from tests import create_partitioned_data, create_serialized_partitioned_data
from tfrnnlm.text import WordTokenizer, CharacterTokenizer, Vocabulary, PartitionedData, DocumentSet, \
    WhitespaceTokenizer, Tokenizer


class TestTokenization(TestCase):
    def test_base_tokenizer(self):
        tokenizer = Tokenizer(True)
        self.assertRaises(NotImplementedError, tokenizer, "one two three")

    def test_whitespace_tokenizer(self):
        tokenizer = WhitespaceTokenizer(True)
        tokens = tokenizer("\nCall me Ishmael. Some years ago--never mind how long precisely  ")
        self.assertEqual(
            ["call", "me", "ishmael.", "some", "years", "ago--never", "mind", "how", "long", "precisely"],
            tokens)
        tokenizer = WhitespaceTokenizer(False)
        tokens = tokenizer("\nCall me Ishmael. Some years ago--never mind how long precisely  ")
        self.assertEqual(
            ["Call", "me", "Ishmael.", "Some", "years", "ago--never", "mind", "how", "long", "precisely"],
            tokens)
        self.assertEqual("The man ran", tokenizer.join_tokens(["The", "man", "ran"]))

    def test_word_tokenizer(self):
        tokenizer = WordTokenizer(True)
        tokens = tokenizer("\nCall me Ishmael. Some years ago--never mind how long precisely  ")
        self.assertEqual(["call", "me", "ishmael", "some", "years", "ago", "never", "mind", "how", "long", "precisely"],
                         tokens)
        tokenizer = WordTokenizer(False)
        tokens = tokenizer("\nCall me Ishmael. Some years ago--never mind how long precisely  ")
        self.assertEqual(["Call", "me", "Ishmael", "Some", "years", "ago", "never", "mind", "how", "long", "precisely"],
                         tokens)
        self.assertEqual("The man ran", tokenizer.join_tokens(["The", "man", "ran"]))

    def test_character_tokenizer(self):
        tokenizer = CharacterTokenizer(True)
        tokens = tokenizer("One two\n Dog cat ")
        self.assertEqual(["o", "n", "e", " ", "t", "w", "o", "\n", " ", "d", "o", "g", " ", "c", "a", "t", " "],
                         tokens)
        tokenizer = CharacterTokenizer(False)
        tokens = tokenizer("one two\n dog cat ")
        self.assertEqual(["o", "n", "e", " ", "t", "w", "o", "\n", " ", "d", "o", "g", " ", "c", "a", "t", " "],
                         tokens)
        self.assertEqual("dog", tokenizer.join_tokens(["d", "o", "g"]))

    def test_equality(self):
        self.assertEqual(WordTokenizer(True), WordTokenizer(True))
        self.assertNotEqual(WordTokenizer(True), WordTokenizer(False))
        self.assertNotEqual(WordTokenizer(True), CharacterTokenizer(True))


class TestVocabulary(TestCase):
    def test_full_vocabulary(self):
        v = Vocabulary(["the quick brown fox jumped over the lazy dog"], WordTokenizer(True))
        self.assertEqual("Vocabulary, size 8: None:1 the:2 brown:3 dog:4 fox:5 ...", str(v))
        self.assertEqual({"the", "quick", "brown", "fox", "jumped", "over", "lazy", "dog"}, set(v.type_to_index.keys()))
        self.assertEqual(8, len(v))

    def test_limited_vocabulary(self):
        v = Vocabulary(["to be or not to be"], WordTokenizer(True), max_vocabulary=2)
        self.assertEqual({"to", "be"}, set(v.type_to_index.keys()))
        self.assertEqual(2, len(v))
        v = Vocabulary(["hamlet hamlet hamlet to be or not to be"], WordTokenizer(True), min_frequency=2)
        self.assertEqual({"to", "be", "hamlet"}, set(v.type_to_index.keys()))
        self.assertEqual(3, len(v))
        v = Vocabulary(["hamlet hamlet hamlet to be or not to be"], WordTokenizer(True),
                       max_vocabulary=2, min_frequency=2)
        self.assertEqual({"be", "hamlet"}, set(v.type_to_index.keys()))
        self.assertEqual(2, len(v))

    def test_vocabulary_factory(self):
        factory = Vocabulary.factory(WordTokenizer(True), max_vocabulary=2)
        self.assertEqual(Vocabulary(["to be or not to be"], WordTokenizer(True), max_vocabulary=2),
                         factory(["to be or not to be"]))

    def test_index_tokens(self):
        document = "the quick brown fox jumped over the lazy dog"
        vocabulary = Vocabulary([document], WordTokenizer(True))
        np.testing.assert_equal(np.array([2, 9, 3, 5, 6, 8, 2, 7, 4]), vocabulary.index_string(document))

    def test_out_of_vocabulary(self):
        vocabulary = Vocabulary(["a a a b b OOV c"], WordTokenizer(True), out_of_vocabulary="OOV")
        self.assertEqual(1, vocabulary.index("OOV"))
        self.assertEqual(1, vocabulary.index("z"))
        self.assertEqual(2, vocabulary.index("a"))
        self.assertEqual(3, vocabulary.index("b"))
        self.assertEqual(4, vocabulary.index("c"))

    def test_invalid_index(self):
        document = "the quick brown fox jumped over the lazy dog"
        vocabulary = Vocabulary([document], WordTokenizer(True))
        self.assertRaises(ValueError, vocabulary.type, 0)
        self.assertRaises(ValueError, vocabulary.type, -1)


class TestDocumentSet(TestCase):
    def setUp(self):
        self.document_set = DocumentSet([np.arange(20), np.arange(13)])

    def test_batches(self):
        batches = DocumentSet.language_model_batches(np.arange(20), time_steps=3, batch_size=4)
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

    def test_document_set_properties(self):
        self.assertEqual(len(self.document_set), 33)
        self.assertEqual(str(self.document_set), "2 documents, 33 tokens")

    def test_document_set_total_batches(self):
        self.assertEqual(self.document_set.total_batches(3, 4), 4)

    def test_document_set_epoch(self):
        batches = self.document_set.epoch(time_steps=3, batch_size=4)
        self.assertIsInstance(batches, collections.Iterable)
        np.testing.assert_equal(list(batches),
                                # Document 1, Batch 1
                                [(True,
                                  np.array([[0, 1, 2],
                                            [3, 4, 5],
                                            [6, 7, 8],
                                            [9, 10, 11]]),
                                  np.array([[1, 2, 3],
                                            [4, 5, 6],
                                            [7, 8, 9],
                                            [10, 11, 12]]),
                                  0.25),
                                 # Document 1, Batch 2
                                 (False,
                                  np.array([[12, 13, 14],
                                            [15, 16, 17],
                                            [18, 19, 0],
                                            [0, 0, 0]]),
                                  np.array([[13, 14, 15],
                                            [16, 17, 18],
                                            [19, 0, 0],
                                            [0, 0, 0]]),
                                  0.5),
                                 # Document 2, Batch 1
                                 (True,
                                  np.array([[0, 1, 2],
                                            [3, 4, 5],
                                            [6, 7, 8],
                                            [9, 10, 11]]),
                                  np.array([[1, 2, 3],
                                            [4, 5, 6],
                                            [7, 8, 9],
                                            [10, 11, 12]]),
                                  0.75),
                                 # Document 2, Batch 2
                                 (False,
                                  np.array([[12, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0]]),
                                  np.array([[0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0]]),
                                  1.0)]
                                )


class TestPartitionedData(TestCase):
    def setUp(self):
        self.partitioned_data = create_partitioned_data()

    def test_partitioned_data(self):
        self.assertIsInstance(self.partitioned_data, PartitionedData)
        self.assertEqual("Partitioned Data: train, test, validate", repr(self.partitioned_data))
        self.assertEqual("""Vocabulary, size 3: None:1 red:2
train: 2 documents, 6 tokens
blue blue green ...

test: 1 documents, 4 tokens
green green red ...

validate: 1 documents, 4 tokens
red blue blue ...
""", str(self.partitioned_data))

    def test_invalid_partition(self):
        self.assertRaises(ValueError, PartitionedData.from_text, {
            "train": ["red red red", "blue blue green"],
            "validate": ["red blue blue orange"],
            "test": ["green green red black"]
        }, ["bogus"], Vocabulary.factory(WordTokenizer(True)))

    def test_partition_name_iterator(self):
        self.assertEqual(["train", "test", "validate"], [p for p in self.partitioned_data])

    def test_index_by_partition_name(self):
        self.assertEqual(self.partitioned_data.document_partitions["train"], self.partitioned_data["train"])


class TestSerialization(TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def test_tokenizer_serialization(self):
        tokenizer_name = os.path.join(self.directory, "tokenizer.pkl")
        tokenizer = WordTokenizer(True)
        self._serialize(tokenizer_name, tokenizer)
        self.assertTrue(os.path.isfile(tokenizer_name))
        deserialized_tokenizer = self._deserialize(tokenizer_name)
        self.assertEqual(tokenizer, deserialized_tokenizer)
        self.assertEqual(tokenizer("The quick brown fox"), deserialized_tokenizer("The quick brown fox"))

    def test_vocabulary_serialization(self):
        vocabulary_name = os.path.join(self.directory, "vocabulary.pkl")
        vocabulary = Vocabulary(["the quick brown fox jumped over the lazy dog"], WordTokenizer(True))
        self._serialize(vocabulary_name, vocabulary)
        self.assertTrue(os.path.isfile(vocabulary_name))
        deserialized_vocabulary = self._deserialize(vocabulary_name)
        self.assertEqual(vocabulary, deserialized_vocabulary)
        s = "The quick black fox"
        np.testing.assert_equal(vocabulary.index_string(s), deserialized_vocabulary.index_string(s))

    def test_document_set_serialization(self):
        document_set = DocumentSet([np.arange(20), np.arange(13)])
        filename = os.path.join(self.directory, "document-set.npy")
        document_set.serialize(filename)
        self.assertTrue(os.path.isfile(filename))
        deserialized_document_set = document_set.deserialize(filename)
        self.assertEqual(document_set, deserialized_document_set)

    def test_partitioned_data_serialization(self):
        directory = os.path.join(self.directory, "partitioned-data")
        partitioned_data = create_serialized_partitioned_data(directory)
        self.assertTrue(os.path.isdir(directory))
        deserialized_partitioned_data = PartitionedData.deserialize(directory)
        self.assertEqual(partitioned_data, deserialized_partitioned_data)

    @staticmethod
    def _serialize(filename, obj):
        with open(filename, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def _deserialize(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def tearDown(self):
        shutil.rmtree(self.directory)
