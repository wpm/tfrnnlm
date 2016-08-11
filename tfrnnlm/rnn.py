import os

import numpy as np
import tensorflow as tf
from tfrnnlm import logger
from tfrnnlm.text import language_model_batches


class RNN(object):
    """Recursive Neural Network"""

    def __init__(self, init, max_gradient, batch_size, time_steps, vocabulary, hidden_units, layers):
        self.vocabulary = vocabulary
        vocabulary_size = len(self.vocabulary)
        with tf.name_scope("Parameters"):
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")

        with tf.name_scope("Input"):
            self.input = tf.placeholder(tf.int64, shape=(batch_size, time_steps), name="input")
            self.targets = tf.placeholder(tf.int64, shape=(batch_size, time_steps), name="targets")

        with tf.name_scope("Embedding"):
            self.embedding = tf.Variable(tf.random_uniform((vocabulary_size, hidden_units), -init, init),
                                         dtype=tf.float32,
                                         name="embedding")
            self.embedded_input = tf.nn.embedding_lookup(self.embedding, self.input, name="embedded_input")

        with tf.name_scope("RNN"):
            cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_probability)
            rnn_layers = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)
            self.reset_state = rnn_layers.zero_state(batch_size, dtype=tf.float32)
            self.state = tf.placeholder(tf.float32, self.reset_state.get_shape(), "state")
            self.outputs, self.next_state = tf.nn.dynamic_rnn(rnn_layers, self.embedded_input, time_major=True,
                                                              initial_state=self.state)

        with tf.name_scope("Cost"):
            # Concatenate all the batches into a single row.
            self.flattened_outputs = tf.reshape(tf.concat(1, self.outputs), (-1, hidden_units),
                                                name="flattened_outputs")
            # Project the outputs onto the vocabulary.
            self.w = tf.get_variable("w", (hidden_units, vocabulary_size))
            self.b = tf.get_variable("b", vocabulary_size)
            self.predicted = tf.matmul(self.flattened_outputs, self.w) + self.b
            # Compare predictions to labels.
            self.loss = tf.nn.seq2seq.sequence_loss_by_example([self.predicted], [tf.concat(-1, self.targets)],
                                                               [tf.ones(batch_size * time_steps)])
            self.cost = tf.div(tf.reduce_sum(self.loss), batch_size, name="cost")
            tf.scalar_summary(self.cost.op.name, self.cost)

        with tf.name_scope("Train"):
            self.iteration = tf.Variable(0, name="iteration", trainable=False)
            self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()),
                                                       max_gradient, name="clip_gradients")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train = optimizer.apply_gradients(zip(self.gradients, tf.trainable_variables()), name="train",
                                                   global_step=self.iteration)

        self.initialize = tf.initialize_all_variables()
        self.summary = tf.merge_all_summaries()

    @property
    def batch_size(self):
        return self.input.get_shape()[0].value

    @property
    def time_steps(self):
        return self.input.get_shape()[1].value

    def train_model(self, documents, learning_rate, keep_probability, model_directory, logging_interval,
                    max_epochs=None, max_iterations=None):
        if model_directory is not None:
            summary_directory = os.path.join(model_directory, "summary")
        else:
            summary_directory = None

        with tf.Session() as session:
            train_summary = summary_writer(summary_directory, session.graph)
            session.run(self.initialize)
            epoch = 1
            iteration = None
            try:
                while True:
                    epoch_cost = 0
                    epoch_iteration = 0
                    for document in documents:
                        state = session.run(self.reset_state)
                        for context, target in language_model_batches(document, self.time_steps, self.batch_size):
                            _, cost, state, summary, iteration = session.run(
                                [self.train, self.cost, self.next_state, self.summary, self.iteration],
                                feed_dict={
                                    self.input: context,
                                    self.targets: target,
                                    self.state: state,
                                    self.learning_rate: learning_rate,
                                    self.keep_probability: keep_probability
                                })
                            epoch_cost += cost
                            epoch_iteration += self.time_steps
                            if (iteration - 1) % logging_interval == 0:
                                logger.info("Epoch %d, Iteration %d, training perplexity %0.4f" % (
                                    epoch, iteration, np.exp(epoch_cost / epoch_iteration)))
                                train_summary.add_summary(summary, global_step=iteration)
                            if max_iterations is not None and iteration > max_iterations:
                                raise StopTrainingException()
                    epoch += 1
                    if max_epochs is not None and epoch > max_epochs:
                        raise StopTrainingException()
            except (StopTrainingException, KeyboardInterrupt):
                logger.info("Stop training at epoch %d, iteration %d" % (epoch, iteration))
                train_summary.flush()


class StopTrainingException(Exception):
    pass


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
