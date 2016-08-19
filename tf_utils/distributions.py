import numpy as np
import tensorflow as tf


def gaussian_diag_logps(mean, logvar, sample=None):
    if sample is None:
        noise = tf.random_normal(tf.shape(mean))
        sample = mean + tf.exp(0.5 * logvar) * noise

    return -0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) / tf.exp(logvar))


class DiagonalGaussian(object):

    def __init__(self, mean, logvar, sample=None):
        self.mean = mean
        self.logvar = logvar

        if sample is None:
            noise = tf.random_normal(tf.shape(mean))
            sample = mean + tf.exp(0.5 * logvar) * noise
        self.sample = sample

    def logps(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)


def discretized_logistic(mean, logscale, binsize=1 / 256.0, sample=None):
    scale = tf.exp(logscale)
    sample = (tf.floor(sample / binsize) * binsize - mean) / scale
    logp = tf.log(tf.sigmoid(sample + binsize / scale) - tf.sigmoid(sample) + 1e-7)
    return tf.reduce_sum(logp, [1, 2, 3])


def logsumexp(x):
    x_max = tf.reduce_max(x, [1], keep_dims=True)
    return tf.reshape(x_max, [-1]) + tf.log(tf.reduce_sum(tf.exp(x - x_max), [1]))


def repeat(x, n):
    if n == 1:
        return x

    shape = map(int, x.get_shape().as_list())
    shape[0] *= n
    idx = tf.range(tf.shape(x)[0])
    idx = tf.reshape(idx, [-1, 1])
    idx = tf.tile(idx, [1, n])
    idx = tf.reshape(idx, [-1])
    x = tf.gather(x, idx)
    x.set_shape(shape)
    return x


def compute_lowerbound(log_pxz, sum_kl_costs, k=1):
    if k == 1:
        return sum_kl_costs - log_pxz

    # log 1/k \sum p(x | z) * p(z) / q(z | x) = -log(k) + logsumexp(log p(x|z) + log p(z) - log q(z|x))
    log_pxz = tf.reshape(log_pxz, [-1, k])
    sum_kl_costs = tf.reshape(sum_kl_costs, [-1, k])
    return - (- tf.log(float(k)) + logsumexp(log_pxz - sum_kl_costs))
