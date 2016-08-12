import os

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
            if args.model is not None:
                summary_directory = os.path.join(args.model, "summary")
            else:
                summary_directory = None
            epoch = iteration = None
            train_summary = summary_writer(summary_directory, session.graph)
            try:
                for epoch, iteration, train_perplexity, summary in \
                        model.train(session, args.train, args.learning_rate, args.keep):
                    if iteration % args.logging_interval == 0:
                        logger.info(
                            "Epoch %d, Iteration %d, training perplexity %0.4f" % (epoch, iteration, train_perplexity))
                        train_summary.add_summary(summary, global_step=iteration)
                    if args.validation_interval is not None and iteration % args.validation_interval == 0:
                        pass
                    if args.max_iterations is not None and iteration > args.max_iterations:
                        break
                    if args.max_epochs is not None and epoch > args.max_epochs:
                        break
            except KeyboardInterrupt:
                pass
            logger.info("Stop training at epoch %d, iteration %d" % (epoch, iteration))
            train_summary.flush()


def summary_writer(summary_directory, graph):
    class NullSummaryWriter(object):
        def add_summary(self, *args, **kwargs):
            pass

        def flush(self):
            pass

    if summary_directory is not None:
        return tf.train.SummaryWriter(summary_directory, graph)
    else:
        return NullSummaryWriter()
