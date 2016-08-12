import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.rnn import RNN


def train_model(args):
    if args.model is None:
        logger.warn("Not saving a model.")
    logger.info(args.vocabulary)
    with tf.Graph().as_default():
        model = RNN(args.init, 5, args.batch_size, args.time_steps, args.vocabulary,
                    args.hidden_units, args.layers)
        with tf.Session() as session:
            model.train(session, args.train, args.learning_rate, args.keep,
                        args.model, args.logging_interval, args.validation_interval,
                        args.max_epochs, args.max_iterations)
