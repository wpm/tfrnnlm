import json
import os
import shutil
import sys
import tempfile
from argparse import ArgumentParser, Namespace, ArgumentTypeError, ArgumentError
from io import StringIO
from unittest import TestCase

from tests import create_serialized_partitioned_data
from tfrnnlm.command import create_data_set, test_model
from tfrnnlm.main import create_argument_parser, main


class TestDatasetCommandLine(TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.output_directory = os.path.join(self.directory, "output")
        self.parser = create_argument_parser()
        self.partitions_json = os.path.join(self.directory, "partitions.json")
        self.partitions = {"train": ["train.txt"], "validate": ["valid.txt"], "test": ["test.txt"]}
        with open(self.partitions_json, "w") as f:
            json.dump(self.partitions, f)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_parser_exists(self):
        self.assertIsInstance(self.parser, ArgumentParser)

    def test_dataset(self):
        cmd = "dataset %s %s train" % (self.output_directory, self.partitions_json)
        actual = self.parser.parse_args(cmd.split())
        expected = Namespace(partitions=self.partitions, vocabulary_partitions=["train"], func=create_data_set,
                             directory=self.output_directory, log="INFO", max_vocabulary=None,
                             min_frequency=None, tokenizer="whitespace", case_normalized=False, out_of_vocabulary=None)
        self.assertEqual(expected, actual)

    def test_dataset_with_optional_arguments(self):
        cmd = "dataset %s %s train --max-vocabulary=50000 --min-frequency=100 --tokenizer=word" \
              % (self.output_directory, self.partitions_json)
        actual = self.parser.parse_args(cmd.split())
        expected = Namespace(partitions=self.partitions, vocabulary_partitions=["train"], func=create_data_set,
                             directory=self.output_directory, log="INFO", max_vocabulary=50000,
                             min_frequency=100, tokenizer="word", case_normalized=False, out_of_vocabulary=None)
        self.assertEqual(expected, actual)

    def test_invalid_max_vocabulary(self):
        cmd = "dataset %s %s train --max-vocabulary=-50000" % (self.output_directory, self.partitions_json)
        command_line_error(self, ArgumentTypeError, lambda: self.parser.parse_args(cmd.split()))
        cmd = "dataset %s %s train --max-vocabulary=0" % (self.output_directory, self.partitions_json)
        command_line_error(self, ArgumentTypeError, lambda: self.parser.parse_args(cmd.split()))

    def test_directory_already_exists(self):
        cmd = "dataset %s %s train" % (self.directory, self.partitions_json)
        command_line_error(self, ArgumentError, lambda: self.parser.parse_args(cmd.split()))


class TestTrainCommandLine(TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        data_set_directory = os.path.join(self.directory, "partitioned-data")
        self.data_set = create_serialized_partitioned_data(data_set_directory)
        self.cmd = "train %s" % data_set_directory
        self.parser = create_argument_parser()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_invalid_parameters(self):
        cmd = "%s --keep-probability=-0.5" % self.cmd
        command_line_error(self, ArgumentTypeError, lambda: self.parser.parse_args(cmd.split()))
        cmd = "%s --keep-probability=1.5" % self.cmd
        command_line_error(self, ArgumentTypeError, lambda: self.parser.parse_args(cmd.split()))
        cmd = "%s --learning-rate=-1.5" % self.cmd
        command_line_error(self, ArgumentTypeError, lambda: self.parser.parse_args(cmd.split()))

    def test_valid_parameters(self):
        cmd = "%s --keep-probability=0.5 --hidden-units=200 --max-gradient=4" % self.cmd
        actual = self.parser.parse_args(cmd.split())
        self.assertEqual(0.5, actual.keep_probability)
        self.assertEqual(200, actual.hidden_units)
        self.assertEqual(4, actual.max_gradient)
        self.assertEqual(self.data_set, actual.data_set)


class TestTestCommandLine(TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        data_set_directory = os.path.join(self.directory, "partitioned-data")
        self.data_set = create_serialized_partitioned_data(data_set_directory)
        self.cmd = "test model-directory %s" % data_set_directory
        self.parser = create_argument_parser()

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_test(self):
        actual = self.parser.parse_args(self.cmd.split())
        self.assertEqual(actual.model_directory, "model-directory")
        self.assertEqual(actual.data_set, self.data_set)
        self.assertEqual(actual.func, test_model)
        self.assertEqual(actual.log, "INFO")

    def test_invalid_sample(self):
        cmd = "%s --sample=1.5" % self.cmd
        command_line_error(self, ArgumentTypeError, lambda: self.parser.parse_args(cmd.split()))
        cmd = "%s --sample=-1.5" % self.cmd
        command_line_error(self, ArgumentTypeError, lambda: self.parser.parse_args(cmd.split()))


class TestMain(TestCase):
    def test_no_arguments(self):
        actual = main_function_output([])
        self.assertEqual("""usage: tfrnnlm [-h] [--version] [--log LOG]
               {dataset,dataset-info,train,test,sample} ...
""", actual)

    def test_help(self):
        actual = main_function_output(["--help"])
        expected = """usage: tfrnnlm [-h] [--version] [--log LOG]
               {dataset,dataset-info,train,test,sample} ...

tfrnnlm version 1.0.0

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --log LOG             logging level

TensorFlow RNN Language Model:
  {dataset,dataset-info,train,test,sample}
    dataset             create data set
    dataset-info        information about a data set
    train               train a language model
    test                Use a previously-trained model to get perplexity on a
                        test set
    sample              sample text from language model
"""
        self.assertEqual(expected, actual)


def main_function_output(args):
    sys.argv = ["tfrnnlm"] + args
    sys.stdout = s = StringIO()
    try:
        main()
    except SystemExit:
        pass
    sys.stderr = sys.__stdout__
    return s.getvalue()


def command_line_error(test_case, error_type, action):
    with open(os.devnull, "w") as sys.stderr:  # Suppresses error message.
        try:
            with test_case.assertRaises(error_type):
                action()
        except SystemExit:  # Stops argparse from ending the program
            pass
