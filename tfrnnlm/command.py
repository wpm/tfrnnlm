import time
from datetime import timedelta

import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.rnn import RNN, Parameters, ExitCriteria, Validation, Directories
from tfrnnlm.text import PartitionedData, Vocabulary, WordTokenizer, CharacterTokenizer, WhitespaceTokenizer, \
    DocumentSet


def create_data_set(args):
    if args.vocabulary_partitions is None:
        args.vocabulary_partitions = args.partitions.keys()
    tokenizer = {"whitespace": WhitespaceTokenizer,
                 "word": WordTokenizer,
                 "character": CharacterTokenizer}[args.tokenizer](args.case_normalized)
    factory = Vocabulary.factory(tokenizer,
                                 min_frequency=args.min_frequency, max_vocabulary=args.max_vocabulary,
                                 out_of_vocabulary=args.out_of_vocabulary)
    partitions = {}
    for partition, filenames in args.partitions.items():
        partitions[partition] = [open(filename).read() for filename in filenames]
    partitioned_data = PartitionedData.from_text(partitions, args.vocabulary_partitions, factory)
    partitioned_data.serialize(args.directory)
    logger.info("Created %s in %s" % (partitioned_data, args.directory))


def get_data_set_info(args):
    print(args.data_set)
    if args.batches is not None:
        print()
        time_steps = args.batches[0]
        batch_size = args.batches[1]
        for partition_name in args.data_set:
            print("%s %d batches" %
                  (partition_name, args.data_set[partition_name].total_batches(time_steps, batch_size)))


def train_model(args):
    if args.model_directory is None:
        logger.warn("Not saving a model.")
    logger.info(args.data_set)
    if args.validation_partition:
        validation = Validation(args.validation_interval, args.data_set[args.validation_partition])
    else:
        validation = None
    # Run training.
    start_time = time.time()
    with tf.Graph().as_default():
        model = RNN(args.max_gradient,
                    args.batch_size, args.time_steps, len(args.data_set.vocabulary),
                    args.hidden_units, args.layers)
        with tf.Session() as session:
            model.train(session,
                        args.init,
                        args.data_set[args.training_partition],
                        Parameters(args.learning_rate, args.keep_probability),
                        ExitCriteria(args.max_iterations, args.max_epochs),
                        validation,
                        args.logging_interval,
                        Directories(args.model_directory, args.summary_directory))
    logger.info("Total training time %s" % timedelta(seconds=(time.time() - start_time)))


def test_model(args):
    test_set = DocumentSet(args.test_set)
    logger.info("Test set: %s" % test_set)
    with tf.Graph().as_default():
        with tf.Session() as session:
            model = RNN.restore(session, args.model_directory)
            perplexity = model.test(session, test_set)
    print("Perplexity %0.4f" % perplexity)
