import os
import pickle

import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.rnn import RNN
from tfrnnlm.text import IndexedVocabulary, vocabulary_from_documents


def train_model(args):
    # Build a vocabulary out of all the documents.
    document_names = args.train + args.validate
    vocabulary = vocabulary_from_documents((open(d).read() for d in document_names), args.tokenization,
                                           IndexedVocabulary.factory(max_vocabulary=args.max_vocabulary))
    logger.info(vocabulary)
    if args.model is not None:
        with open(os.path.join(args.model, "vocabulary"), "wb") as vocabulary_file:
            pickle.dump(vocabulary, vocabulary_file)
    else:
        logger.warn("Not saving a model.")
    # Convert documents into indexes of types.
    documents = []
    for train_filename in args.train:
        with open(train_filename) as f:
            documents.append(list(vocabulary.index_tokens(args.tokenization(f.read()))))
    # Train a model.
    with tf.Graph().as_default():
        model = RNN(args.batch_size, args.time_steps, vocabulary, args.hidden_units,
                    args.init, args.keep, args.layers,
                    args.max_gradient, args.learning_rate)
        model.train_model(documents, args.time_steps, args.batch_size,
                          args.model, args.logging_interval,
                          args.max_epochs, args.max_iterations)
