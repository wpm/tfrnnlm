import argparse
import logging
import os
import pickle

import numpy as np
from tfrnnlm import __version__, logger
from tfrnnlm.command import train_model, test_model
from tfrnnlm.prepare_data import index_text_files


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


def create_argument_parser():
    parser = argparse.ArgumentParser(description="tfrnnlm version %s" % __version__, fromfile_prefix_chars='@')
    parser.add_argument('--version', action='version', version="%(prog)s " + __version__)
    parser.add_argument("--log", default="INFO", help="logging level")
    parser.set_defaults(func=usage(parser))

    subparsers = parser.add_subparsers(title="TensorFlow RNN Language Model")

    index = subparsers.add_parser("index", description="Index text files and create a vocabulary.",
                                  help="index text files")
    index.add_argument("indexed_data_directory", type=new_directory,
                       help="directory to put indexed files and vocabulary")
    index.add_argument("documents", nargs="+", help="text files")
    index.add_argument("--tokenization", choices=["word", "penntb"], default="word", help="tokenization method")
    index.add_argument("--min-frequency", type=positive_integer,
                       help="minimum type frequency for inclusion in the vocabulary")
    index.add_argument("--max-vocabulary", type=positive_integer, help="maximum vocabulary size")
    index.set_defaults(func=index_text_files)

    train = subparsers.add_parser("train", description="Train an RNN language model.", help="train a language model")
    train.add_argument("vocabulary", type=pickle_file, help="indexed vocabulary file")
    train.add_argument("training_set", nargs="+", type=np.load, help="files containing training data")
    train.add_argument("--validation-set", nargs="+", type=np.load, default=[], help="files containing validation data")
    train.add_argument("--validation-interval", type=positive_integer, default=1000,
                       help="how often to run the validation set")
    train.add_argument("--model-directory", type=new_directory, help="directory to which to write the model")
    train.add_argument("--summary-directory", type=new_directory,
                       help="directory to which to write a training summary")
    train.add_argument("--time-steps", type=positive_integer, default=20, help="training unrolled time steps")
    train.add_argument("--batch-size", type=positive_integer, default=20, help="training size batch")
    train.add_argument("--hidden-units", type=positive_integer, default=650, help="number of hidden units in the RNN")
    train.add_argument("--layers", type=positive_integer, default=2, help="number of RNN layers")
    train.add_argument("--keep-probability", type=real_zero_to_one, default=0.5,
                       help="probability to keep a cell in a dropout layer")
    train.add_argument("--max-gradient", type=positive_real, default=5, help="value to clip gradients to")
    train.add_argument("--max-iterations", type=positive_integer, help="number of training iterations to run")
    train.add_argument("--logging-interval", type=positive_integer, default=100,
                       help="log and write summary after this many iterations")
    train.add_argument("--max-epochs", type=positive_integer, default=6, help="number of training epochs to run")
    train.add_argument("--learning-rate", type=positive_real, default=1.0, help="training learning rate")
    train.add_argument("--init", type=positive_real, default=0.05, help="random initial absolute value range")
    train.add_argument("--sample", type=real_zero_to_one, help="only use this much of the data sets")
    train.set_defaults(func=train_model)

    test = subparsers.add_parser("test", description="Use an RNN model.",
                                 help="Use a previously-trained model to get perplexity on a test set")
    test.add_argument("model_directory", help="directory from which to read the model")
    test.add_argument("test_set", nargs="+", type=np.load, help="files containing test data")
    test.add_argument("--sample", type=real_zero_to_one, help="only use this much of the test set")
    test.set_defaults(func=test_model)

    sample = subparsers.add_parser("sample", description="Sample text from a RNN model.",
                                   help="sample text from language model")
    sample.add_argument("model", help="directory from which to read the model")
    sample.set_defaults(func=lambda a: print(a))
    return parser


def usage(parser):
    class UsageClosure(object):
        def __call__(self, _):
            parser.print_usage()
            parser.exit(0)

    return UsageClosure()


def configure_logger(level, format):
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(format))
    logger.addHandler(h)


# Various argparse type functions.

def positive_integer(i):
    i = int(i)
    if i <= 0:
        raise argparse.ArgumentTypeError("%d must be greater than zero" % i)
    return i


def positive_real(r):
    r = float(r)
    if r <= 0:
        raise argparse.ArgumentTypeError("%d must be greater than zero" % r)
    return r


def real_zero_to_one(r):
    r = float(r)
    if r < 0 or r > 1:
        raise argparse.ArgumentTypeError("%f must be between zero and one" % r)
    return r


def new_directory(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        raise argparse.ArgumentError(None, "The directory %s already exists." % directory)
    return directory


def pickle_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
