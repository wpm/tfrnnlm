from unittest import TestCase

from tfrnnlm.rnn import ExitCriteria, Parameters, Validation, Directories, Intervals


class TestRNN(TestCase):
    def test_configuration_groups(self):
        e = ExitCriteria(100, 50)
        self.assertEqual(100, e.max_iterations)
        self.assertEqual(50, e.max_epochs)
        p = Parameters(0.01, 0.5)
        self.assertEqual(0.01, p.learning_rate)
        self.assertEqual(0.5, p.keep_probability)
        v = Validation(None, None)
        self.assertEqual(None, v.interval)
        self.assertEqual(None, v.validation_set)
        d = Directories("model", "summary")
        self.assertEqual("model", d.model)
        self.assertEqual("summary", d.summary)
        i = Intervals(100, 200)
        self.assertEqual(100, i.logging_interval)
        self.assertEqual(200, i.model_interval)
