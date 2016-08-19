import time
from datetime import timedelta

import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.document_set import DocumentSet
from tfrnnlm.rnn import RNN, Parameters, ExitCriteria, Validation


def train_model(args):
    if args.model_directory is None:
        logger.warn("Not saving a model.")
    logger.info(args.vocabulary)
    # Optionally only retain a portion of the documents.
    if args.sample is not None:
        args.training_set = [document[:int(len(document) * args.sample)] for document in args.training_set]
        args.validation_set = [document[:int(len(document) * args.sample)] for document in args.validation_set]
    training_set = DocumentSet(args.training_set)
    logger.info("Training set: %s" % training_set)
    if args.validation_set is not None:
        validation_set = DocumentSet(args.validation_set)
        logger.info("Validation set: %s" % validation_set)
        validation = Validation(args.validation_interval, validation_set)
    else:
        validation = None
    # Run training.
    start_time = time.time()
    with tf.Graph().as_default():
        model = RNN(args.init, args.max_gradient,
                    args.batch_size, args.time_steps, len(args.vocabulary),
                    args.hidden_units, args.layers)
        with tf.Session() as session:
            model.train(session,
                        training_set,
                        Parameters(args.learning_rate, args.keep_probability),
                        ExitCriteria(args.max_iterations, args.max_epochs),
                        validation,
                        args.logging_interval,
                        args.summary_directory)
    logger.info("Total training time %s" % timedelta(seconds=(time.time() - start_time)))
