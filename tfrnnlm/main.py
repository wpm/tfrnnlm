import argparse
import os
import pickle

import numpy as np
from tfrnnlm import configure_logger, __version__
from tfrnnlm.prepare_data import index_text_files
from tfrnnlm.text import WhitespaceWordTokenization, PennTreebankTokenization
from tfrnnlm.train import train_model


def main():
    global parser
    parser = argparse.ArgumentParser(description="tfrnnlm version %s" % __version__, fromfile_prefix_chars='@')

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--log", default="INFO", help="logging level")

    subparsers = parser.add_subparsers(title="TensorFlow RNN Language Model")

    index = subparsers.add_parser("index", description="Index text files and create a vocabulary.", parents=[shared],
                                  help="index text files")
    index.add_argument("indexed_data_directory", type=create_new_directory,
                       help="directory to put indexed files and vocabulary")
    index.add_argument("documents", nargs="+", help="text files")
    index.add_argument("--tokenization", choices=["word", "penntb"], default="word", help="tokenization method")
    index.add_argument("--min-frequency", type=int, help="minimum type frequency for inclusion in the vocabulary")
    index.add_argument("--max-vocabulary", type=int, help="maximum vocabulary size")
    index.set_defaults(func=index_text_files)

    train = subparsers.add_parser("train", description="Train an RNN language model.", parents=[shared],
                                  help="train a language model")
    train.add_argument("vocabulary", type=vocabulary, help="indexed vocabulary file")
    train.add_argument("train", nargs="+", type=np.load, help="files containing training data")
    train.add_argument("--validate", nargs="+", type=np.load, default=[], help="files containing validation data")
    train.add_argument("--model", type=create_new_directory, help="directory to which to write the model")
    train.add_argument("--time-steps", type=int, default=20, help="training unrolled time steps")
    train.add_argument("--batch-size", type=int, default=20, help="training size batch")
    train.add_argument("--hidden-units", type=int, default=650, help="number of hidden units in the RNN")
    train.add_argument("--layers", type=int, default=2, help="number of RNN layers")
    train.add_argument("--keep", type=float, default=0.5, help="probability to keep a cell in a dropout layer")
    train.add_argument("--max-gradient", type=float, default=5, help="value to clip gradients to")
    train.add_argument("--max-iterations", type=int, help="number of training iterations to run")
    train.add_argument("--logging-interval", type=int, default=10,
                       help="log and write summary after this many iterations")
    train.add_argument("--max-epochs", type=int, default=6, help="number of training epochs to run")
    train.add_argument("--learning-rate", type=float, default=1.0, help="training learning rate")
    train.add_argument("--init", type=float, default=0.05, help="random initial absolute value range")
    train.set_defaults(func=train_model)

    sample = subparsers.add_parser("sample", description="Sample text from a RNN model.", parents=[shared],
                                   help="sample text from language model")
    sample.add_argument("model", help="directory from which to read the model")
    sample.set_defaults(func=lambda a: print(a))
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_usage()
        parser.exit(0)

    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")

    if hasattr(args, "tokenization"):
        args.tokenization = {"word": WhitespaceWordTokenization(),
                             "penntb": PennTreebankTokenization()}[args.tokenization]
    args.func(args)


def create_new_directory(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        parser.print_usage()
        parser.error("The directory %s already exists." % directory)
    return directory


def vocabulary(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
