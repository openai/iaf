import os
from cifar10_data import load, cifar_preloaded
import numpy as np


def get_inputs(dataset, mode, batch_size, image_size):
    if dataset == "cifar10":
        assert image_size == 32
        images = get_images(dataset, mode, image_size).x
        return cifar_preloaded(images, batch_size)
        # return cifar_inputs("/home/rafal/data/cifar-10-batches-bin", batch_size)


class Dataset(object):
    def __init__(self, x, deterministic=False):
        self.x = x
        self.n = x.shape[0]
        self._deterministic = deterministic
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if not self._deterministic:
            perm = np.arange(self.n)
            np.random.shuffle(perm)
            self.x = self.x[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size > self.n:
            self.shuffle()

        cur_id = self._next_id
        self._next_id += batch_size

        return self.x[cur_id:cur_id+batch_size]


class CIFAR(object):
    def __init__(self, data_dir, deterministic=False):
        self.train = Dataset(load(data_dir, "train")[0], deterministic)
        self.test = Dataset(load(data_dir, "test")[0], deterministic)


def get_images(dataset, mode, image_size, deterministic=False):
    if dataset == "cifar10":
        path = os.environ['CIFAR10_PATH']
        cifar = CIFAR(path, deterministic)
        if mode == "train":
            return cifar.train
        if mode == "test":
            return cifar.test
