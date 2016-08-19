import cPickle
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import tensorflow as tf


def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)


def unpickle(file):
    fo = open(file, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return {'x': np.cast[np.uint8](d['data'].reshape((10000, 3, 32, 32))),
            'y': np.array(d['labels']).astype(np.uint8)}


def load(data_dir, subset='train'):
    maybe_download_and_extract(data_dir)
    if subset == 'train':
        train_data = [unpickle(os.path.join(data_dir,'cifar-10-batches-py/data_batch_' + str(i))) for i in range(1,6)]
        trainx = np.concatenate([d['x'] for d in train_data],axis=0)
        trainy = np.concatenate([d['y'] for d in train_data],axis=0)
        return trainx, trainy
    elif subset=='test':
        test_data = unpickle(os.path.join(data_dir,'cifar-10-batches-py/test_batch'))
        testx = test_data['x']
        testy = test_data['y']
        return testx, testy
    else:
        raise NotImplementedError('subset should be either train or test')


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.
      Recommendation: if you want N-way read parallelism, call this function
      N times.  This will give you N independent Readers reading different
      files & positions within those files, which will give better mixing of
      examples.
      Args:
        filename_queue: A queue of strings with the filenames to read from.
      Returns:
        An object representing a single example, with the following fields:
          height: number of rows in the result (32)
          width: number of columns in the result (32)
          depth: number of color channels in the result (3)
          key: a scalar string Tensor describing the filename & record number
            for this example.
          label: an int32 Tensor with the label in the range 0..9.
          uint8image: a [height, width, depth] uint8 Tensor with the image data
      """

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    result.uint8image = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                   [result.depth, result.height, result.width])
    return result


def cifar_preloaded(images, batch_size):
    with tf.device("/cpu:0"):
        input_images = tf.constant(images)
        image = tf.train.slice_input_producer([input_images])
        return tf.train.shuffle_batch(image, batch_size, 20000 + 3 * batch_size, 20000, 5)


def cifar_inputs(data_dir, batch_size):
    with tf.device("/cpu:0"):
        filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
        filename_queue = tf.train.string_input_producer(filenames)
        image_list = [read_cifar10(filename_queue).uint8image for _ in range(5)]
        images = tf.train.shuffle_batch_join(
            [image_list],
            batch_size=batch_size,
            capacity=20000 + 3 * batch_size,
            min_after_dequeue=20000)[0]
        return images
