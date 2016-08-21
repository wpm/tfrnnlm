import collections
import os.path
import shutil
import sys
import tempfile
import textwrap
from argparse import ArgumentParser, Namespace, ArgumentTypeError, ArgumentError
from unittest import TestCase

import numpy as np
from tfrnnlm.document_set import DocumentSet
from tfrnnlm.main import create_argument_parser
from tfrnnlm.prepare_data import vocabulary_from_documents, index_text_files
from tfrnnlm.rnn import ExitCriteria, Parameters, Validation
from tfrnnlm.text import IndexedVocabulary, WhitespaceWordTokenization, PennTreebankTokenization


class TestTokenization(TestCase):
    def test_whitespace_word_tokenization(self):
        tokenization = WhitespaceWordTokenization()
        self.assertEqual(str(tokenization), "Whitespace Word Tokenization")
        tokens = tokenization("\nCall me Ishmael. Some years ago--never mind how long precisely  ")
        self.assertEqual(tokens,
                         ["call", "me", "ishmael", "some", "years", "ago", "never", "mind", "how", "long", "precisely"])

    def test_penn_treebank_word_tokenization(self):
        s = textwrap.dedent("""seoul also has instituted effective <unk> procedures to aid these teams she said
taiwan has improved""")
        tokenization = PennTreebankTokenization()
        self.assertEqual(str(tokenization), "Penn Treebank Tokenization")
        actual = tokenization(s)
        self.assertIsInstance(actual, collections.Iterable)
        expected = ["seoul", "also", "has", "instituted", "effective", "<unk>", "procedures", "to", "aid", "these",
                    "teams", "she", "said", "<eos>", "taiwan", "has", "improved", "<eos>"]
        self.assertEqual(list(actual), expected)


class TestIndexing(TestCase):
    def test_full_vocabulary(self):
        v = IndexedVocabulary("the quick brown fox jumped over the lazy dog".split())
        self.assertEqual(str(v), "Indexed Vocabulary, size 9: None:0 the:1 brown:2 dog:3 fox:4 ...")
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


class TestRNN(TestCase):
    def test_configuration_groups(self):
        e = ExitCriteria(100, 50)
        self.assertEqual(e.max_iterations, 100)
        self.assertEqual(e.max_epochs, 50)
        p = Parameters(0.01, 0.5)
        self.assertEqual(p.learning_rate, 0.01)
        self.assertEqual(p.keep_probability, 0.5)
        v = Validation(None, None)
        self.assertEqual(v.interval, None)
        self.assertEqual(v.validation_set, None)


class TestCommandLine(TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.parser = create_argument_parser()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_parser_exists(self):
        self.assertIsInstance(self.parser, ArgumentParser)

    def test_index(self):
        output_directory = os.path.join(self.directory, "output")
        cmd = "index %s document1 document2" % output_directory
        actual = self.parser.parse_args(cmd.split())
        expected = Namespace(documents=["document1", "document2"], func=index_text_files,
                             indexed_data_directory=output_directory, log='INFO', max_vocabulary=None,
                             min_frequency=None, tokenization='word')
        self.assertEqual(actual, expected)

    def test_index_with_optional_arguments(self):
        output_directory = os.path.join(self.directory, "output")
        cmd = "index %s document1 document2 --max-vocabulary=50000 --min-frequency=100 --tokenization=penntb" \
              % output_directory
        actual = self.parser.parse_args(cmd.split())
        expected = Namespace(documents=["document1", "document2"], func=index_text_files,
                             indexed_data_directory=output_directory, log="INFO", max_vocabulary=50000,
                             min_frequency=100, tokenization="penntb")
        self.assertEqual(actual, expected)

    def test_invalid_integer(self):
        output_directory = os.path.join(self.directory, "output")
        cmd = "index %s document1 document2 --max-vocabulary=-50000" % output_directory
        self._command_line_error(ArgumentTypeError, lambda: self.parser.parse_args(cmd.split()))

    def test_directory_already_exists(self):
        cmd = "index %s document1 document2 --max-vocabulary=-50000" % self.directory
        self._command_line_error(ArgumentError, lambda: self.parser.parse_args(cmd.split()))

    def _command_line_error(self, error_type, action):
        with open(os.devnull, "w") as sys.stderr:  # Suppresses error message.
            try:
                with self.assertRaises(error_type):
                    action()
            except SystemExit:  # Stops argparse from ending the program
                pass
