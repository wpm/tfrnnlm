import time
from datetime import timedelta

import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.document_set import DocumentSet
from tfrnnlm.rnn import RNN, Parameters, ExitCriteria, Validation, Directories


def train_model(args):
    if args.model_directory is None:
        logger.warn("Not saving a model.")
    logger.info(args.vocabulary)
    training_set = DocumentSet(args.training_set, args.sample)
    logger.info("Training set: %s" % training_set)
    if args.validation_set is not None:
        validation_set = DocumentSet(args.validation_set, args.sample)
        logger.info("Validation set: %s" % validation_set)
        validation = Validation(args.validation_interval, validation_set)
    else:
        validation = None
    # Run training.
    start_time = time.time()
    with tf.Graph().as_default():
        model = RNN(args.max_gradient,
                    args.batch_size, args.time_steps, len(args.vocabulary),
                    args.hidden_units, args.layers)
        with tf.Session() as session:
            model.train(session,
                        args.init,
                        training_set,
                        Parameters(args.learning_rate, args.keep_probability),
                        ExitCriteria(args.max_iterations, args.max_epochs),
                        validation,
                        args.logging_interval,
                        Directories(args.model_directory, args.summary_directory))
    logger.info("Total training time %s" % timedelta(seconds=(time.time() - start_time)))


def test_model(args):
    test_set = DocumentSet(args.test_set, args.sample)
    logger.info("Test set: %s" % test_set)
    with tf.Graph().as_default():
        with tf.Session() as session:
            model = RNN.restore(session, args.model_directory)
            perplexity = model.test(session, test_set)
    print("Perplexity %0.4f" % perplexity)
