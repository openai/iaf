import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope


@add_arg_scope
def linear(name, x, num_units, init_scale=1., init=False):
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            v = tf.get_variable("V", [int(x.get_shape()[1]), num_units], tf.float32,
                                tf.random_normal_initializer(0, 0.05))
            v_norm = tf.nn.l2_normalize(v.initialized_value(), [0])
            x_init = tf.matmul(x, v_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            _ = tf.get_variable("g", initializer=scale_init)
            _ = tf.get_variable("b", initializer=-m_init * scale_init)
            return tf.reshape(scale_init, [1, num_units]) * (x_init - tf.reshape(m_init, [1, num_units]))
        else:
            v = tf.get_variable("V", [int(x.get_shape()[1]), num_units])
            g = tf.get_variable("g", [num_units])
            b = tf.get_variable("b", [num_units])

            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x, v)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(v), [0]))
            return tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])


@add_arg_scope
def conv2d(name, x, num_filters, filter_size=(3, 3), stride=(1, 1), pad="SAME", init_scale=0.1, init=False,
           mask=None, dtype=tf.float32, **_):
    stride_shape = [1, 1, stride[0], stride[1]]
    filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[1]), num_filters]

    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            v = tf.get_variable("V", filter_shape, dtype, tf.random_normal_initializer(0, 0.05, dtype=dtype))
            v = v.initialized_value()
            if mask is not None:  # Used for auto-regressive convolutions.
                v = mask * v

            v_norm = tf.nn.l2_normalize(v, [0, 1, 2])
            x_init = tf.nn.conv2d(x, v_norm, stride_shape, pad, data_format="NCHW")
            m_init, v_init = tf.nn.moments(x_init, [0, 2, 3])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            _ = tf.get_variable("g", initializer=tf.log(scale_init) / 3.0)
            _ = tf.get_variable("b", initializer=-m_init * scale_init)
            return tf.reshape(scale_init, [1, -1, 1, 1]) * (x_init - tf.reshape(m_init, [1, -1, 1, 1]))
        else:
            v = tf.get_variable("V", filter_shape)
            g = tf.get_variable("g", [num_filters])
            b = tf.get_variable("b", [num_filters])
            if mask is not None:
                v = mask * v

            # use weight normalization (Salimans & Kingma, 2016)
            w = tf.reshape(tf.exp(g), [1, 1, 1, num_filters]) * tf.nn.l2_normalize(v, [0, 1, 2])

            # calculate convolutional layer output
            b = tf.reshape(b, [1, -1, 1, 1])
            return tf.nn.conv2d(x, w, stride_shape, pad, data_format="NCHW") + b


def my_deconv2d(x, filters, strides):
    input_shape = x.get_shape()
    output_shape = [int(input_shape[0]), int(filters.get_shape()[2]),
                    int(input_shape[2] * strides[2]), int(input_shape[2] * strides[3])]
    x = tf.transpose(x, (0, 2, 3, 1))  # go to NHWC data layout
    output_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
    strides = [strides[0], strides[2], strides[3], strides[1]]
    x = tf.nn.conv2d_transpose(x, filters, output_shape=output_shape, strides=strides, padding="SAME")
    x = tf.transpose(x, (0, 3, 1, 2))  # back to NCHW
    return x


@add_arg_scope
def deconv2d(name, x, num_filters, filter_size=(3, 3), stride=(2, 2), pad="SAME", init_scale=0.1, init=False,
             mask=None, dtype=tf.float32, **_):
    stride_shape = [1, 1, stride[0], stride[1]]
    filter_shape = [filter_size[0], filter_size[1], num_filters, int(x.get_shape()[1])]

    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            v = tf.get_variable("V", filter_shape, dtype, tf.random_normal_initializer(0, 0.05, dtype=dtype))
            v = v.initialized_value()
            if mask is not None:  # Used for auto-regressive convolutions.
                v = mask * v

            v_norm = tf.nn.l2_normalize(v, [0, 1, 2])
            x_init = my_deconv2d(x, v_norm, stride_shape)
            m_init, v_init = tf.nn.moments(x_init, [0, 2, 3])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            _ = tf.get_variable("g", initializer=tf.log(scale_init) / 3.0)
            _ = tf.get_variable("b", initializer=-m_init*scale_init)
            return tf.reshape(scale_init, [1, -1, 1, 1]) * (x_init - tf.reshape(m_init, [1, -1, 1, 1]))
        else:
            v = tf.get_variable("V", filter_shape)
            g = tf.get_variable("g", [num_filters])
            b = tf.get_variable("b", [num_filters])
            if mask is not None:
                v = mask * v

            # use weight normalization (Salimans & Kingma, 2016)
            w = tf.reshape(tf.exp(g), [1, 1, num_filters, 1]) * tf.nn.l2_normalize(v, [0, 1, 2])

            # calculate convolutional layer output
            b = tf.reshape(b, [1, -1, 1, 1])
            return my_deconv2d(x, w, stride_shape) + b


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = n_out / n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in / n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask


def get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal=False):
    l = (h - 1) / 2
    m = (w - 1) / 2
    mask = np.ones([h, w, n_in, n_out], dtype=np.float32)
    mask[:l, :, :, :] = 0
    mask[l, :m, :, :] = 0
    mask[l, m, :, :] = get_linear_ar_mask(n_in, n_out, zerodiagonal)
    return mask


@add_arg_scope
def ar_conv2d(name, x, num_filters, filter_size=(3, 3), stride=(1, 1), pad="SAME", init_scale=1.,
              zerodiagonal=True, **_):
    h = filter_size[0]
    w = filter_size[1]
    n_in = int(x.get_shape()[1])
    n_out = num_filters

    mask = tf.constant(get_conv_ar_mask(h, w, n_in, n_out, zerodiagonal))
    with arg_scope([conv2d]):
        return conv2d(name, x, num_filters, filter_size, stride, pad, init_scale, mask=mask)


# Auto-Regressive convnet with l2 normalization
@add_arg_scope
def ar_multiconv2d(name, x, context, n_h, n_out, nl=tf.nn.elu, **_):
    with tf.variable_scope(name), arg_scope([ar_conv2d]):
        for i, size in enumerate(n_h):
            x = ar_conv2d("layer_%d" % i, x, size, zerodiagonal=False)
            if i == 0:
                x += context
            x = nl(x)
        return [ar_conv2d("layer_out_%d" % i, x, size, zerodiagonal=True) for i, size in enumerate(n_out)]


def resize_nearest_neighbor(x, scale):
    input_shape = map(int, x.get_shape().as_list())
    size = [int(input_shape[2] * scale), int(input_shape[3] * scale)]
    x = tf.transpose(x, (0, 2, 3, 1))  # go to NHWC data layout
    x = tf.image.resize_nearest_neighbor(x, size)
    x = tf.transpose(x, (0, 3, 1, 2))  # back to NCHW
    return x
