import argparse
import os

from tfrnnlm import configure_logger, __version__
from tfrnnlm.train import train_model


def main():
    global parser
    parser = argparse.ArgumentParser(description="tfrnnlm version %s" % __version__, fromfile_prefix_chars='@')

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--log", default="INFO", help="logging level")

    subparsers = parser.add_subparsers(title="TensorFlow RNN Language Model")

    train = subparsers.add_parser("train", description="Train an RNN language model.", parents=[shared],
                                  help="train a language model")
    train.add_argument("text", type=argparse.FileType(), help="file containing training text")
    train.add_argument("model", type=model_directory, help="directory to which to write the model")
    train.add_argument("--max-vocabulary", type=int, help="maximum vocabulary size")
    train.add_argument("--time-steps", type=int, default=20, help="training unrolled time steps")
    train.add_argument("--batch-size", type=int, default=20, help="training size batch")
    train.add_argument("--hidden-units", type=int, default=650, help="number of hidden units in the RNN")
    train.add_argument("--layers", type=int, default=2, help="number of RNN layers")
    train.add_argument("--keep", type=float, default=0.5, help="probability to keep a cell in a dropout layer")
    train.add_argument("--max-gradient", type=float, default=5, help="value to clip gradients to")
    train.add_argument("--max-iterations", type=int, help="number of training iterations to run")
    train.add_argument("--max-epochs", type=int, default=6, help="number of training epochs to run")
    train.add_argument("--learning-rate", type=float, default=1.0, help="training learning rate")
    train.add_argument("--init", type=float, default=0.05, help="random initial absolute value range")
    train.set_defaults(func=train_model)

    sample = subparsers.add_parser("sample", description="Sample text from a RNN model.", parents=[shared],
                                   help="sample text from language model")
    sample.add_argument("model", help="directory from which to read the model")
    sample.set_defaults(func=lambda a: print(a))

    args = parser.parse_args()
    configure_logger(args.log.upper(), "%(asctime)-15s %(levelname)-8s %(message)s")
    args.func(args)


def model_directory(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        parser.print_usage()
        parser.error("The model directory %s already exists." % directory)
    return directory
