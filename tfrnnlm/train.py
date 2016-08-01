import os
import pickle

import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.rnn import RNN
from tfrnnlm.text import whitespace_word_tokenization, IndexedVocabulary, language_model_batches_old


def train_model(args):
    tokens = whitespace_word_tokenization(args.text.read())
    vocabulary = IndexedVocabulary(tokens)
    logger.info(vocabulary)
    with open(os.path.join(args.model, "vocabulary"), "wb") as vocabulary_file:
        pickle.dump(vocabulary, vocabulary_file)
    data = [vocabulary.index(token) for token in tokens]
    batched_input, batched_targets = language_model_batches_old(data, args.batch_size)
    with tf.Graph().as_default():
        model = RNN(len(tokens), args.batch_size, len(vocabulary),
                    args.hidden_units, args.init, args.keep, args.layers,
                    args.max_gradient, args.learning_rate)
        with tf.Session() as session:
            model.train_model(session, batched_input, batched_targets,
                              args.training_epochs,
                              os.path.join(args.model, "summary"))
