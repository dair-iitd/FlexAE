import os
import numpy as np
import tensorflow as tf
from copy import deepcopy


SEED = 42


def load_test_dataset(name, root_folder):
    if name.lower() == 'mnist':
        (_, _), (x, _) = tf.keras.datasets.mnist.load_data()
        x = x.reshape(-1, 28, 28, 1)
        side_length = 28
        channels = 1
    elif name.lower() == 'fashion':
        (_, _), (temp, _) = tf.keras.datasets.fashion_mnist.load_data()
        x = deepcopy(temp)
        x = x.reshape(-1, 28, 28, 1)
        side_length = 28
        channels = 1
    elif name.lower() == 'cifar10':
        (_, _), (x, _) = tf.keras.datasets.cifar10.load_data()
        side_length = 32
        channels = 3
        np.random.shuffle(x)
    elif name.lower() == 'celeba140':
        data_folder = os.path.join(root_folder, 'data', name)
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    elif name.lower() == 'celeba':
        data_folder = os.path.join(root_folder, 'data', name)
        x = np.load(os.path.join(data_folder, 'test.npy'))
        side_length = 64
        channels = 3
    else:
        raise Exception('No such dataset called {}.'.format(name))
    np.random.seed(SEED)
    np.random.shuffle(x)
    return x, side_length, channels
