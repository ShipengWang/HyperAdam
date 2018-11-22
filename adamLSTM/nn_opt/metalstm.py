import nn_opt
from MetaLSTMCell import *
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops import gradients
import os

flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_var", 5, " #'rank' of the denominator "
                                          "in AdamShuffleLSTMCell to approximate the outer product ")
tf.app.flags.DEFINE_float("lr", 0.005, "learning rate")


class MetaLSTMOptModel(nn_opt.BasicNNOptModel):
    # def lstm_cell(self, ):
    #     return LSTMCell(num_units=self.dimH)

    def my_lstm_cell(self,):
        return MyLSTMCell(num_units=self.dimH, split=self.split)

    def adam_lstm_cell(self, varepsilon=1e-24):
        return AdamLSTMCell(num_units=self.dimA, split=self.split, varepsilon=varepsilon)  #

    def adam_shuffle_lstm_cell(self):
        return AdamLSTMCell(num_units=self.dimH, num_var=flags.num_var)

    def _build_pre(self, size):
        self.dimH = size
        self.dimA = size
        self.split = 1  # where to split the input of rnn into true gradient and preprogressed gradient
        self.cellH = MyMultiRNNCell([self.my_lstm_cell(), self.adam_lstm_cell()])
        self.lr = flags.lr  # training phase: 1e-3

    def _build_input(self):
        self.x = self.ph([None])
        # self.x_forward = self.ph([None])
        self.cell_h_state = tuple((self.ph([None, size.c]), self.ph([None, size.h]))
                                  for size in self.cellH.state_size)
        # self.input_state = [self.x, self.x_forward, self.cell_h_state]
        self.input_state = [self.x, self.cell_h_state]

    def _build_initial(self):
        x = self.x
        cell_h_state = self.cellH.zero_state(batch_size=tf.size(x), dtype=tf.float32)
        self.initial_state = [x, cell_h_state]

    def _explore_surface_feature_point(self, f, i, x_now, grad_now, stepsize=0.0):
        shape = tf.shape(x_now)
        x = x_now + stepsize * grad_now + tf.random_normal(shape, mean=0,
                                                           stddev=0.001, dtype=tf.float32)
        fx, grad = self._get_fx(f, i, x)
        loss = fx * tf.ones_like(x, dtype=tf.float32)
        return x, grad, loss

    def _propress_deepmind(self, xs, *args):
        for k, x in enumerate(xs):
            xs[k] = tf.stack([x], 1)
        last = tf.concat([x for _, x in enumerate(xs)], 1)

        for arg in args:
            arg_after_pre = self._deepmind_log_encode(arg)
            last = tf.concat([last, arg_after_pre], 1)
        last = tf.stop_gradient(last)
        return last

    def _propress_elu(self, x, *args):
        last = tf.stack([x], 1)
        if self.is_training:
            last = tf.stop_gradient(last)
        last = tf.nn.elu(self.fc(last, self.dimH, seed=None))  # TODO seed
        return last

    def _normalization(self, x, ord=2):
        norm = tf.norm(x, ord=ord)
        x_normalization = x / norm
        return x_normalization

    def _get_fx_stochastic(self, f, i, x):
        if isinstance(f, list):
            return f[0], f[1]
        fx = f(i, x)
        grad = gradients.gradients(fx, x)[0]
        return fx, grad

    # return state, fx

    def iter(self, f, i, state):
        x, cell_h_state = state
        fx, grad = self._get_fx(f, i, x)
        if self.is_training:
            grad = tf.stop_gradient(grad)

        grad_stack = tf.stack([grad], 1)
        grad_0_norm = self._normalization(grad, ord=2)
        # test
        last = tf.nn.elu(self.fc(tf.stack([grad_0_norm], 1), self.dimH, seed=None))
        # training
        last = tf.concat([grad_stack, last], 1)
        with tf.variable_scope("cellH"):
            out, cell_h_state = self.cellH(last, cell_h_state)

        with tf.variable_scope("fc"):
            last = self.fc(out, 1, seed=None, use_bias=False)  # TODO seed
            delta_x = last[:, 0] * self.lr
        x -= delta_x
        return [x, cell_h_state, [out, delta_x]], fx  # [out, delta_x]


class AdamLstmOptModel(MetaLSTMOptModel):
    def _build_pre(self, size):
        self.dimH = size
        self.dimA = size
        self.split = 1
        self.cellH = MyMultiRNNCell([self.my_lstm_cell(), self.adam_lstm_cell()])
        self.lr = flags.lr  # training phase: 1e-3


class AdamLstmAdaWeightOptModel(MetaLSTMOptModel):
    # moment field & weight field
    def _build_pre(self, size):
        self.dimH = size
        self.dimA = size
        self.split = 1
        self.cellH = MyMultiRNNCell([self.my_lstm_cell(), self.adam_lstm_cell(varepsilon=1e-24)])
        self.lr = flags.lr  # training phase: 1e-3

    # return state, fx
    def iter(self, f, i, state):
        x, cell_h_state = state
        fx, grad = self._get_fx(f, i, x)
        if self.is_training:
            grad = tf.stop_gradient(grad)

        grad_0_norm = self._normalization(grad, ord=2)
        # test
        last = tf.nn.elu(self.fc(tf.stack([grad_0_norm], 1), self.dimH, seed=None))
        # training

        last = tf.concat([tf.stack([grad], 1), last], 1)
        with tf.variable_scope("cellH"):
            adam, cell_h_state = self.cellH(last, cell_h_state)
        #
        # # gate, adam = out
        with tf.variable_scope("fc"):
            weight = tf.nn.elu(self.fc(cell_h_state[0][1], self.dimA, seed=None, use_bias=True))
            direc_field = adam * weight
            delta_x = tf.reduce_sum(direc_field, axis=-1) * self.lr

        x -= delta_x
        return [x, cell_h_state], fx  # , [out, weight, delta_x]


class AdamLstmOptModelH(AdamLstmAdaWeightOptModel):
    # H is the size of the field
    def _build_pre(self, size):
        self.dimH = 21
        self.dimA = 21
        self.split = 1
        self.cellH = MyMultiRNNCell([self.my_lstm_cell(), self.adam_lstm_cell(varepsilon=1e-24)])
        self.lr = flags.lr  # training phase: 1e-3


class AdamLstmAdaWeightOptModelLSTM(MetaLSTMOptModel):
    # moment field & weight field
    def _build_pre(self, size):
        self.dimH = size
        self.dimA = size
        self.split = 1
        self.cellH = MyMultiRNNCell([self.my_lstm_cell(), self.adam_lstm_cell()])
        self.lr = flags.lr  # training phase: 1e-3

    # return state, fx
    def iter(self, f, i, state):
        x, cell_h_state = state
        fx, grad = self._get_fx(f, i, x)
        self.optimizee_grad.append(grad)
        grad = tf.stop_gradient(grad)

        grad_stack = tf.stack([grad], 1)
        grad_0_norm = self._normalization(grad, ord=2)

        last = tf.stack([grad_0_norm], 1)

        last = tf.concat([grad_stack, last], 1)
        with tf.variable_scope("cellH"):
            out, cell_h_state = self.cellH(last, cell_h_state)

        gate, adam = out
        with tf.variable_scope("fc"):
            weight = tf.nn.elu(self.fc(cell_h_state[0][1], self.dimA, seed=None, use_bias=True))
            direc_field = adam * weight
            delta_x = tf.reduce_sum(direc_field, axis=-1) * self.lr

        x -= delta_x
        return [x, cell_h_state], fx  # , [out, weight, delta_x]


class AdamLstmAdaWeightOptModelPre(MetaLSTMOptModel):
    # moment field & weight field
    def _build_pre(self, size):
        self.dimH = size
        self.dimA = size
        self.split = 1
        self.cellH = MyMultiRNNCell([self.adam_lstm_cell(varepsilon=1e-16)])
        self.lr = flags.lr  # training phase: 1e-3

    # return state, fx
    def iter(self, f, i, state):
        x, cell_h_state = state
        fx, grad = self._get_fx(f, i, x)
        self.optimizee_grad.append(grad)
        grad = tf.stop_gradient(grad)

        grad_stack = tf.stack([grad], 1)
        grad_0_norm = self._normalization(grad, ord=2)

        last = self._propress_elu(grad_0_norm)

        cell_input = tf.concat([grad_stack, last], 1)
        with tf.variable_scope("cellH"):
            out, cell_h_state = self.cellH(cell_input, cell_h_state)

        gate, adam = out
        with tf.variable_scope("fc"):
            weight = tf.nn.elu(self.fc(last, self.dimA, seed=None, use_bias=True))
            direc_field = adam * weight
            delta_x = tf.reduce_sum(direc_field, axis=-1) * self.lr

        x -= delta_x
        return [x, cell_h_state], fx  # , [out, weight, delta_x]


class AdamLstmAdaWeightOptModelNeither(MetaLSTMOptModel):
    # moment field & weight field
    def _build_pre(self, size):
        self.dimH = size
        self.dimA = size
        self.split = 1
        self.cellH = MyMultiRNNCell([self.adam_lstm_cell(varepsilon=1e-16)])
        self.lr = flags.lr  # training phase: 1e-3

    # return state, fx
    def iter(self, f, i, state):
        x, cell_h_state = state
        fx, grad = self._get_fx(f, i, x)
        self.optimizee_grad.append(grad)
        grad = tf.stop_gradient(grad)

        grad_stack = tf.stack([grad], 1)
        grad_0_norm = self._normalization(grad, ord=2)

        last = tf.stack([grad_0_norm], 1)
        cell_input = tf.concat([grad_stack, last], 1)
        with tf.variable_scope("cellH"):
            out, cell_h_state = self.cellH(cell_input, cell_h_state)

        gate, adam = out
        with tf.variable_scope("fc"):
            weight = tf.nn.elu(self.fc(last, self.dimA, seed=None, use_bias=True))
            direc_field = adam * weight
            delta_x = tf.reduce_sum(direc_field, axis=-1) * self.lr

        x -= delta_x
        return [x, cell_h_state], fx 
