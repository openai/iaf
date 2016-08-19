import numpy as np
import tensorflow as tf
from distributions import logsumexp, compute_lowerbound, repeat


class DistributionsTestCase(tf.test.test_util.TensorFlowTestCase):
    def test_logsumexp(self):
        a = np.arange(10)
        res = np.log(np.sum(np.exp(a)))

        with self.test_session():
            res_tf = logsumexp(a.astype(np.float32).reshape([1, -1])).eval()
            self.assertEqual(res, res_tf)

    def test_lowerbound(self):
        a = np.log(np.array([0.3, 0.3, 0.3, 0.3], np.float32).reshape([1, -1]))
        b = np.log(np.array([0.1, 0.5, 0.9, 0.6], np.float32).reshape([1, -1]))

        res = - (- np.log(4) + np.log(np.sum(np.exp(a - b))))
        with self.test_session():
            res_tf = tf.reduce_sum(compute_lowerbound(a, b, 4)).eval()
            self.assertAlmostEqual(res, res_tf, places=4)

    def test_lowerbound2(self):
        a = np.log(np.array([0.3, 0.3, 0.3, 0.3], np.float32).reshape([-1, 1]))
        b = np.log(np.array([0.1, 0.5, 0.9, 0.6], np.float32).reshape([-1, 1]))

        res = (b - a).sum()
        with self.test_session():
            res_tf = tf.reduce_sum(compute_lowerbound(a, b, 1)).eval()
            self.assertAlmostEqual(res, res_tf, places=4)

    def test_repeat(self):
        a = np.random.randn(10, 5, 2)
        repeated_a = np.repeat(a, 2, axis=0)
        with self.test_session():
            repeated_a_tf = repeat(a, 2).eval()
            self.assertAllClose(repeated_a, repeated_a_tf)
