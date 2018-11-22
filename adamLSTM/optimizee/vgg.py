import tensorflow as tf
import numpy as np
import optimizee
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.python.framework import dtypes
import os
import collections
import sys
import tarfile
import pickle
from six.moves import urllib

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
get_path = os.getcwd()
parent_directory = os.path.split(get_path)[0]
datasets = os.path.join(parent_directory, "datasets")
Datasets = collections.namedtuple('Datasets', ['train', 'test'])


class VGGModel(optimizee.Optimizee):
    '''VGG-like CNNs on dataset MNIST or CIFAR10.'''
    mnist_dataset = None
    cifar_dataset = None

    def __init__(self, input_data='mnist', n_batches=128, fc_num=1, conv_num=2, pool_num=1,
                 add_dropout=False, use_batch_normalization=False,
                 initial_param_scale=0.1):
        assert conv_num % pool_num == 0
        optimizee.Optimizee.__init__(self)
        self.n_batches = n_batches
        self.input_data = input_data
        self.add_dropout = add_dropout
        self.use_batch_normalization = use_batch_normalization
        self.fc_num = fc_num
        self.conv_num = conv_num
        self.pool_num = pool_num
        self.initial_param_scale = initial_param_scale
        if self.input_data == 'cifar10':
            self.n_classes = 10
            self.input_size = 32
            self.input_channel = 3
        if self.input_data == 'mnist':
            self.n_classes = 10
            self.input_size = 28
            self.input_channel = 1
        assert self.input_size % (2 ** self.pool_num) == 0
        self.x_dim = 0

        for j in range(self.pool_num):
            self.x_dim += self.get_n([3, 3, self.input_channel if j == 0 else 2 ** (j + 3), 2 ** (j + 4)])
            self.x_dim += self.get_n([2 ** (j + 4)])
            for k in range(self.conv_num / self.pool_num - 1):
                self.x_dim += self.get_n([3, 3, 2 ** (j + 4), 2 ** (j + 4)])
                self.x_dim += self.get_n([2 ** (j + 4)])
        for j in range(self.fc_num):
            self.x_dim += self.get_n([(
                                          self.input_size * self.input_size * 2 ** 3 / 2 ** self.pool_num) if j == 0 else 2 ** (
                self.pool_num + 4), self.n_classes if j == (self.fc_num - 1) else 2 ** (self.pool_num + 4)])
            self.x_dim += self.get_n([self.n_classes if j == (self.fc_num - 1) else 2 ** (self.pool_num + 4)])

    def _build_dataset(self):
        if self.input_data == 'mnist':
            if VGGModel.mnist_dataset == None:
                data_path = os.path.join(datasets, "MNIST_data/")
                VGGModel.mnist_dataset = input_data.read_data_sets(data_path, one_hot=True, reshape=False)
            self.dataset = VGGModel.mnist_dataset
        if self.input_data == 'cifar10':
            if VGGModel.cifar_dataset == None:
                data_path = os.path.join(datasets, "Cifar10_data/")
                VGGModel.cifar_dataset = cifar_datasets(data_path, one_hot=True)
            self.dataset = VGGModel.cifar_dataset

    def build(self):
        self._build_dataset()
        self.x = tf.placeholder(tf.float32,
                                [None, self.n_batches, self.input_size, self.input_size, self.input_channel])
        self.y_ = tf.placeholder(tf.float32, [None, self.n_batches, self.n_classes])

    def build_test(self):
        self._build_dataset()
        self.x_test = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, self.input_channel])
        self.y_test = tf.placeholder(tf.float32, [None, self.n_classes])

    def get_x_dim(self):
        return self.x_dim

    def get_initial_x(self):
        return np.random.normal(size=[self.x_dim], scale=self.initial_param_scale)

    def next_internal_feed_dict(self):
        return {}

    def next_feed_dict(self, n_iterations):
        x_data = np.zeros([n_iterations, self.n_batches, self.input_size, self.input_size, self.input_channel])
        y_data = np.zeros([n_iterations, self.n_batches, self.n_classes])
        for i in range(n_iterations):
            x_data[i], y_data[i] = self.dataset.train.next_batch(self.n_batches)
        return {self.x: x_data, self.y_: y_data}

    def feed_dict_test(self):
        x_test_data = self.dataset.test.images
        y_test_data = self.dataset.test.labels
        return {self.x_test: x_test_data, self.y_test: y_test_data}

    def inference(self, x, inputs):
        self.start_get_weights(x)

        conv_f = []
        conv_b = []
        for j in range(self.pool_num):
            f = [self.get_weights([3, 3, self.input_channel if j == 0 else 2 ** (j + 3), 2 ** (j + 4)])]
            b = [self.get_weights([2 ** (j + 4)])]
            for k in range(self.conv_num / self.pool_num - 1):
                f.append(self.get_weights([3, 3, 2 ** (j + 4), 2 ** (j + 4)]))
                b.append(self.get_weights([2 ** (j + 4)]))
            conv_f.append(f)
            conv_b.append(b)

        fc_w = []
        fc_b = []
        for j in range(self.fc_num):
            fc_w.append(self.get_weights([(
                                              self.input_size * self.input_size * 2 ** 3 / 2 ** self.pool_num) if j == 0 else 2 ** (
                self.pool_num + 4), self.n_classes if j == (self.fc_num - 1) else 2 ** (self.pool_num + 4)]))
            fc_b.append(self.get_weights([self.n_classes if j == (self.fc_num - 1) else 2 ** (self.pool_num + 4)]))

        last = inputs
        # print last.get_shape
        for j in range(self.pool_num):
            for k in range(self.conv_num / self.pool_num):
                last = self.conv(last, conv_f[j][k], conv_b[j][k])
                if self.use_batch_normalization:
                    last = self.bn(last)
            last = self.max_pool(last)
        # print last.get_shape

        for j in range(self.fc_num - 1):
            last = self.fc(last, fc_w[j], fc_b[j])
            last = tf.nn.relu(last)
            if self.add_dropout:
                last = tf.nn.dropout(last, 0.5)
        pred = self.fc(last, fc_w[self.fc_num - 1], fc_b[self.fc_num - 1])
        return pred
    def loss(self, i, x):

        pred = self.inference(x, self.x[i])
        # TODO 100
        loss = 1.00 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y_[i]))
        return loss

    def test_optimizee(self, x):
        pred_activation = self.inference(x, self.x_test)
        predction = tf.argmax(pred_activation, axis=1)
        label = tf.argmax(self.y_test, axis=1)
        correct_prediction = tf.equal(predction, label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        correct_prediction_2 = tf.nn.in_top_k(pred_activation, label, 2)
        accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_activation, labels=self.y_test))
        return [loss, accuracy, accuracy_2]

    def conv(self, bottom, f, b):
        last = tf.nn.conv2d(bottom, f, [1, 1, 1, 1], padding='SAME')
        last = tf.nn.bias_add(last, b)
        last = tf.nn.relu(last)
        return last

    def fc(self, bottom, w, b):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])
        last = tf.nn.bias_add(tf.matmul(x, w), b)
        return last

    def bn(self, activation):
        ac_mean, ac_var = tf.nn.moments(activation, axes=[0], keep_dims=True)
        offset = None
        scale = None
        last = tf.nn.batch_normalization(activation, ac_mean, ac_var, offset, scale, 1e-5)
        return last

    def max_pool(self, bottom):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cifar_datasets(dirname, one_hot=True,
                   dtype=dtypes.float32,
                   reshape=False,
                   seed=None):
    maybe_download_and_extract(dirname)
    dirname = os.path.join(dirname, 'cifar-10-batches-py/')
    train_images = []
    train_labels = []
    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        image, label = load_batch(fpath)
        if i == 1:
            train_images = np.array(image)
            train_labels = np.array(label)
        else:
            train_images = np.concatenate([train_images, image], axis=0)
            train_labels = np.concatenate([train_labels, label], axis=0)
    train_images = np.dstack((train_images[:, :1024], train_images[:, 1024:2048], train_images[:, 2048:]))
    train_images = np.reshape(train_images, [-1, 32, 32, 3])
    if one_hot:
        train_labels = dense_to_one_hot(train_labels, 10)
    print 'Cifar train_images size:', train_images.shape
    print 'Cifar train_labels size:', train_labels.shape
    train_images = train_images / 255.0 - 0.5

    fpath = os.path.join(dirname, "test_batch")
    image, label = load_batch(fpath)
    test_images = np.array(image)
    test_labels = np.array(label)
    test_images = np.dstack((test_images[:, :1024], test_images[:, 1024:2048], test_images[:, 2048:]))
    test_images = np.reshape(test_images, [-1, 32, 32, 3])
    if one_hot:
        test_labels = dense_to_one_hot(test_labels, 10)
    print "Cifar test_images size:", test_images.shape
    print "Cifar test_lables size:", test_labels.shape
    test_images = test_images / 255.0 - 0.5

    options = dict(dtype=dtype, reshape=reshape, seed=seed)
    train = DataSet(train_images, train_labels, options)
    test = DataSet(test_images, test_labels, options)
    return Datasets(train=train, test=test)


def maybe_download_and_extract(data_dir):
    """Download and extract the tarball from Alex's website."""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def load_batch(fpath):
    with open(fpath, 'rb') as f:
        if sys.version_info > (3, 0):
            # Python3
            d = pickle.load(f, encoding='latin1')
        else:
            # Python2
            d = pickle.load(f)
    data = d["data"]
    labels = d["labels"]
    return data, labels
