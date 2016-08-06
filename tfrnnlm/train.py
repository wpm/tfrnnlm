import os
import pickle

import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.rnn import RNN
from tfrnnlm.text import whitespace_word_tokenization, IndexedVocabulary


def train_model(args):
    tokens = whitespace_word_tokenization(args.text.read())
    vocabulary = IndexedVocabulary(tokens, max_vocabulary=args.max_vocabulary)
    logger.info(vocabulary)
    if args.model is not None:
        with open(os.path.join(args.model, "vocabulary"), "wb") as vocabulary_file:
            pickle.dump(vocabulary, vocabulary_file)
    else:
        logger.warn("Not saving a model.")
    data = [vocabulary.index(token) for token in tokens]
    with tf.Graph().as_default():
        model = RNN(args.batch_size, args.time_steps, len(vocabulary), args.hidden_units,
                    args.init, args.keep, args.layers,
                    args.max_gradient, args.learning_rate)
        model.train_model(data, args.time_steps, args.batch_size,
                          args.model, args.logging_interval,
                          args.max_epochs, args.max_iterations)
