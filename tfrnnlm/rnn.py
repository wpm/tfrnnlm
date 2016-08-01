import tensorflow as tf
from tfrnnlm import logger


class RNN(object):
    """Recursive Neural Network"""

    def __init__(self, total_time, batch_size, vocabulary_size,
                 hidden_units, init, keep, layers,
                 max_gradient, learning_rate):
        with tf.name_scope("Input"):
            self.input = tf.placeholder(tf.int64, shape=(total_time, batch_size), name="input")
            self.targets = tf.placeholder(tf.int64, shape=(total_time, batch_size), name="targets")

        with tf.name_scope("Embedding"):
            self.embedding = tf.Variable(tf.random_uniform((vocabulary_size, hidden_units), -init, init),
                                         dtype=tf.float32,
                                         name="embedding")
            self.embedded_input = tf.nn.embedding_lookup(self.embedding, self.input, name="embedded_input")

        with tf.name_scope("RNN"):
            cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
            if keep < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep)
            rnn_layers = tf.nn.rnn_cell.MultiRNNCell([cell] * layers)
            self.outputs, _ = tf.nn.dynamic_rnn(rnn_layers, self.embedded_input, time_major=True,
                                                initial_state=rnn_layers.zero_state(batch_size, dtype=tf.float32))

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
                                                               [tf.ones(total_time * batch_size)])
            self.cost = tf.div(tf.reduce_sum(self.loss), batch_size * total_time, name="cost")
            tf.scalar_summary(self.cost.op.name, self.cost)

        with tf.name_scope("Train"):
            self.iteration = tf.Variable(0, name="iteration", trainable=False)
            self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), max_gradient,
                                                       name="clip_gradients")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train = optimizer.apply_gradients(zip(self.gradients, tf.trainable_variables()), name="train",
                                                   global_step=self.iteration)

        self.initialize = tf.initialize_all_variables()
        self.summary = tf.merge_all_summaries()

    def train_model(self, session, batched_input, batched_targets, training_epochs, summary_directory):
        train_summary = summary_writer(summary_directory, session.graph)
        session.run(self.initialize)
        for j in range(training_epochs):
            _, c, s, i = session.run([self.train, self.cost, self.summary, self.iteration],
                                     feed_dict={self.input: batched_input, self.targets: batched_targets})
            logger.info("Iteration %d, cost %0.4f" % (i, c))
            train_summary.add_summary(s, global_step=i)
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
