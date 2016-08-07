import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.rnn import RNN


def train_model(args):
    if args.model is None:
        logger.warn("Not saving a model.")
    logger.info(args.vocabulary)
    with tf.Graph().as_default():
        model = RNN(args.batch_size, args.time_steps, args.vocabulary, args.hidden_units,
                    args.init, args.keep, args.layers,
                    args.max_gradient, args.learning_rate)
        model.train_model(args.train, args.time_steps, args.batch_size,
                          args.model, args.logging_interval,
                          args.max_epochs, args.max_iterations)
