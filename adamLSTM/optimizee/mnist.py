import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import optimizee
import os

get_path = os.getcwd()
parent_directory = os.path.split(get_path)[0]
datasets = os.path.join(parent_directory, "datasets")


class MnistLinearModel(optimizee.Optimizee):
    '''A MLP on dataset MNIST.'''

    mnist = None

    def __init__(self, activation='sigmoid', n_batches=128, n_h=20, n_l=1, initial_param_scale=0.1,
                 add_dropout=False):
        optimizee.Optimizee.__init__(self)

        self.activation = activation
        self.n_batches = n_batches
        self.n_l = n_l
        self.n_h = n_h
        self.initial_param_scale = initial_param_scale
        self.add_dropout = add_dropout

        if n_l == 0:
            self.x_dim = 784 * 10 + 10
        else:
            self.x_dim = 784 * n_h + n_h + (n_h * n_h + n_h) * (n_l - 1) + n_h * 10 + 10

    def _build_dataset(self):
        if MnistLinearModel.mnist == None:
            data_path = os.path.join(datasets, "MNIST_data/")
            MnistLinearModel.mnist = input_data.read_data_sets(data_path, one_hot=True)
        self.mnist = MnistLinearModel.mnist
    def build(self):
        self._build_dataset()
        self.x = tf.placeholder(tf.float32, [None, None, 784])
        self.y_ = tf.placeholder(tf.float32, [None, None, 10])

    def build_test(self):
        self._build_dataset()
        self.x_test = tf.placeholder(tf.float32, [None, 784])
        self.y_test = tf.placeholder(tf.float32, [None, 10])

    def get_x_dim(self):
        return self.x_dim

    def get_initial_x(self, seed=None):
        # np.random.seed(seed)   # TODO seed
        para = np.random.normal(size=[self.x_dim], scale=self.initial_param_scale)
        return para  # initial weights of optimizee

    def next_internal_feed_dict(self):
        return {}

    def next_feed_dict(self, n_iterations):
        x_data = np.zeros([n_iterations, self.n_batches, 784])
        y_data = np.zeros([n_iterations, self.n_batches, 10])
        for i in range(n_iterations):
            x_data[i], y_data[i] = self.mnist.train.next_batch(self.n_batches, shuffle=True)  # TODO shuffle = True
        return {self.x: x_data, self.y_: y_data}

    def feed_dict_test(self):
        # x_test_data, y_test_data = self.mnist.test.next_batch(n_batches, shuffle=True)
        x_test_data = self.mnist.test.images
        y_test_data = self.mnist.test.labels
        return {self.x_test: x_test_data, self.y_test: y_test_data}

    def inference(self, x, inputs):
        self.start_get_weights(x)

        if self.n_l > 0:
            w1 = self.get_weights([784, self.n_h])
            b1 = self.get_weights([self.n_h])
            w2 = self.get_weights([self.n_h, 10])
            b2 = self.get_weights([10])

            wl = [self.get_weights([self.n_h, self.n_h]) for k in range(self.n_l - 1)]
            bl = [self.get_weights([self.n_h]) for k in range(self.n_l - 1)]

            def act(x):
                if self.activation == 'sigmoid':
                    return tf.sigmoid(x)
                elif self.activation == 'relu':
                    return tf.nn.relu(x)
                elif self.activation == 'elu':
                    return tf.nn.elu(x)
                elif self.activation == 'tanh':
                    return tf.tanh(x)

            last = tf.matmul(inputs, w1) + b1
            last = act(last)

            for k in range(self.n_l - 1):
                last = tf.matmul(last, wl[k]) + bl[k]
                if self.add_dropout:
                    last = tf.nn.dropout(last, 0.5)
                last = act(last)

            pred = tf.matmul(last, w2) + b2
        else:
            w = self.get_weights([784, 10])
            b = self.get_weights([10])
            pred = tf.matmul(input, w) + b
        return pred

    def loss(self, i, x):
        last = self.inference(x, self.x[i])
        return 1.00 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=last, labels=self.y_[i]))

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


def forward():
    forward_net = MnistLinearModel()
    forward_net.build_test()
    test = forward_net.feed_dict_test()
    x = tf.placeholder(dtype=tf.float32, shape=[None])
    para = forward_net.get_initial_x()
    fx, accu = forward_net.test_optimizee(x)
    feed_dict = forward_net.next_internal_feed_dict()
    feed_dict.update(test)
    feed_dict.update({x: para})
    sess = tf.Session()
    f, a = sess.run([fx, accu], feed_dict=feed_dict)
    print (f,a)
    return f, a

def main(_):
    graph = tf.Graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    gpu_options = tf.GPUOptions(allow_growth=True)

    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as session:

             f,a = forward()

if __name__ == '__main__':
    tf.app.run()