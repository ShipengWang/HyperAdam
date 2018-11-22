import tensorflow as tf
# from tensorflow.python.ops import gradients
from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell
# import numpy as np
import nn_opt
import numpy as np
import matplotlib.pyplot as plt

class RNNpropModel(nn_opt.BasicNNOptModel):
    def _build_pre(self, size):
        self.dimA = size
        self.num_of_layers = 2
        self.cellA = MultiRNNCell([LSTMCell(num_units=self.dimA) for _ in range(self.num_of_layers)])
        self.b1 = 0.95
        self.b2 = 0.95
        self.lr = 0.1
        self.eps = 1e-8

    def _build_input(self):
        self.x = self.ph([None])
        self.m = self.ph([None])
        self.v = self.ph([None])
        self.b1t = self.ph([])
        self.b2t = self.ph([])
        self.sid = self.ph([])
        self.cellA_state = tuple((self.ph([None, size.c]), self.ph([None, size.h])) for size in self.cellA.state_size)
        self.input_state = [self.x, self.sid, self.b1t, self.b2t, self.m, self.v, self.cellA_state]

    def _build_initial(self):
        x = self.x
        m = tf.zeros(shape=tf.shape(x))
        v = tf.zeros(shape=tf.shape(x))
        b1t = tf.ones([])
        b2t = tf.ones([])
        cellA_state = self.cellA.zero_state(tf.size(x), tf.float32)
        self.initial_state = [x, tf.zeros([]), b1t, b2t, m, v, cellA_state]

    # return state, fx
    def iter(self, f, i, state):
        x, sid, b1t, b2t, m, v, cellA_state = state

        fx, grad = self._get_fx(f, i, x)
        # self.optimizee_grad.append(grad)
        grad = tf.stop_gradient(grad)

        m = self.b1 * m + (1 - self.b1) * grad
        v = self.b2 * v + (1 - self.b2) * (grad ** 2)

        b1t *= self.b1
        b2t *= self.b2

        sv = tf.sqrt(v / (1 - b2t)) + self.eps
        # TODO
        last = tf.stack([grad / sv, (m / (1 - b1t)) / sv], 1)
        # last = tf.stack([grad], 1)
        # last = self._deepmind_log_encode(grad, p=10)
        # last = tf.stack([grad/tf.norm(grad, ord=2)], 1)
        last = tf.nn.elu(self.fc(last, 20))

        with tf.variable_scope("cellA"):
            lastA, cellA_state = self.cellA(last, cellA_state)

        with tf.variable_scope("fc_A"):
            a = self.fc(lastA, 1, use_bias=True)[:, 0]

        a = tf.tanh(a) * self.lr

        x -= a

        return [x, sid + 1, b1t, b2t, m, v, cellA_state], fx

