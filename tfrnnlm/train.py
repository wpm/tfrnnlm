import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.rnn import RNN


def train_model(args):
    if args.model_directory is None:
        logger.warn("Not saving a model.")
    logger.info(args.vocabulary)
    # Optionally downsample the size of the documents.
    if args.sample is not None:
        args.training_set = [document[:int(len(document) * args.sample)] for document in args.training_set]
        args.validation_set = [document[:int(len(document) * args.sample)] for document in args.validation_set]
    training_epoch_size = sum(len(document) for document in args.training_set)
    logger.info("Training epoch size %d tokens" % training_epoch_size)
    if args.validation_set is not None:
        logger.info("Validation epoch size %d tokens" % sum(len(document) for document in args.validation_set))
    # Run training.
    with tf.Graph().as_default():
        model = RNN(args.init, args.max_gradient,
                    args.batch_size, args.time_steps, len(args.vocabulary),
                    args.hidden_units, args.layers)
        with tf.Session() as session:
            epoch = iteration = None
            train_summary = summary_writer(args.summary_directory, session.graph)
            try:
                for epoch, complete, new_epoch, iteration, train_perplexity in \
                        model.train(session, args.training_set, args.learning_rate, args.keep_probability):
                    if iteration % args.logging_interval == 0:
                        logger.info(
                            "Epoch %d, Complete %0.3f, Batch %d, training perplexity %0.4f" %
                            (epoch, complete, iteration, train_perplexity))
                    if new_epoch and args.validation_set and iteration > 1:
                        validation_perplexity = model.test(session, args.validation_set)
                        summary = model.store_validation_perplexity(session, validation_perplexity)
                        logger.info("Epoch %d, Batch %d, validation perplexity %0.4f" %
                                    (epoch, iteration, validation_perplexity))
                        train_summary.add_summary(summary, global_step=iteration)
                    if args.max_iterations is not None and iteration > args.max_iterations:
                        break
                    if args.max_epochs is not None and epoch > args.max_epochs:
                        break
            except KeyboardInterrupt:
                pass
            logger.info("Stop training at epoch %d, iteration %d" % (epoch, iteration))
            train_summary.flush()
            train_summary.close()


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
