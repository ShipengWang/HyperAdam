from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple, MultiRNNCell, _LayerRNNCell
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_EPS_VARIABLE_NAME = "eps"


class MyLSTMCell(_LayerRNNCell):
    def __init__(self, num_units, split=1,
                 use_peepholes=False, cell_clip=None,
                 initializer=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None):
        super(_LayerRNNCell, self).__init__()
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh
        self._state_size = (
            LSTMStateTuple(num_units, num_units)
            if state_is_tuple else 2 * num_units)
        self._output_size = num_units
        self._split_lo = split

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value - 1  # the first column is the gradient, so minus 1
        h_depth = self._num_units

        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units],
            initializer=self._initializer, )
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, ginputs, state):
        """ Run one step of LSTM.
        """
        sigmoid = tf.sigmoid
        grad = tf.slice(ginputs, [0, 0], [-1, self._split_lo])
        inputs = tf.slice(ginputs, [0, self._split_lo], [-1, -1])
        (c_prev, m_prev) = state
        # input_size = inputs.get_shape().with_rank(2)[1]
        # if input_size.value is None:
        #     raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, initializer=self._initializer) as unit_scope:
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            inputs_norm = batch_normalization(inputs, name_scope="lstm_inputs")
            m_prev_norm = batch_normalization(m_prev, name_scope="lstm_hidden")
            # lstm_matrix = _linear([inputs_norm, m_prev_norm], 4 * self._num_units, bias=True)
            lstm_matrix = math_ops.matmul(
                tf.concat([inputs, m_prev], 1), self._kernel) # inputs_norm, m_prev_norm
            lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)
            i, j, f, o = tf.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)

            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))
            m = sigmoid(o) * self._activation(c)

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     tf.concat([c, m], 1))

        output = tf.concat([grad, m], 1)

        return output, new_state


class AdamLSTMCell(MyLSTMCell):
    def __init__(self, num_units, num_var=1, split=1, varepsilon=1e-24,
                 use_peepholes=False, cell_clip=None,
                 initializer=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None):

        super(AdamLSTMCell, self).__init__(num_units,
                                           use_peepholes=use_peepholes, cell_clip=cell_clip,
                                           initializer=initializer,
                                           forget_bias=forget_bias, state_is_tuple=state_is_tuple,
                                           activation=activation)
        self.rank = num_var
        self._split_lo = split
        # 1 + self.rank : momentum + variance
        self._state_size = LSTMStateTuple(num_units * (1 + self.rank), num_units * (1 + self.rank))
        self.eps = varepsilon

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value - 1  # the first column is the gradient, so minus 1
        h_depth = self._num_units

        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 2 * self._num_units],
            initializer=self._initializer, )
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, ginputs, state):
        """ Run one step of LSTM.
        """
        sigmoid = tf.sigmoid
        (momentum_all_prev, accum_forgetgate_all_prev) = state
        momentum_prev = tf.split(momentum_all_prev, num_or_size_splits=self.rank + 1, axis=1)
        grad = tf.slice(ginputs, [0, 0], [-1, self._split_lo])
        inputs = tf.slice(ginputs, [0, self._split_lo], [-1, -1])
        scope = tf.get_variable_scope()

        with tf.variable_scope(scope, initializer=self._initializer):
            inputs_norm = batch_normalization(inputs, name_scope="adam_lstm_inputs")
            momentum_prev_norm = batch_normalization(momentum_prev[0], name_scope="adam_momentum")

            lstm_matrix = math_ops.matmul(
                tf.concat([inputs_norm, momentum_prev_norm], 1), self._kernel)
            lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

            this_forget = sigmoid(lstm_matrix)
            this_input = 1 - this_forget

            input_gate = tf.split(this_input, num_or_size_splits=self.rank + 1, axis=1)

            delta_momentum_all = tf.concat([grad * input_gate[0], grad ** 2 * input_gate[1]], axis=1)
            momentum_all = momentum_all_prev * this_forget + delta_momentum_all
            accum_forgetgate_all = accum_forgetgate_all_prev * this_forget + this_input
            new_state = LSTMStateTuple(momentum_all, accum_forgetgate_all)

            correct_bias = momentum_all / (accum_forgetgate_all + self.eps)
            correct_bias_split = tf.split(correct_bias, num_or_size_splits=self.rank + 1, axis=1)
            v_correct_bias_eps = correct_bias_split[1] + self.eps
            adaptive_moment = correct_bias_split[0] / (tf.sqrt(v_correct_bias_eps))
            return adaptive_moment, new_state



class MyMultiRNNCell(MultiRNNCell):
    def __init__(self, cells, state_is_tuple=True):
        """Create a RNN cell composed sequentially of a number of RNNCells.

            Args:
              cells: list of RNNCells that will be composed in this order.
              state_is_tuple: If True, accepted and returned states are n-tuples, where
                `n = len(cells)`.  If False, the states are all
                concatenated along the column axis.  This latter behavior will soon be
                deprecated.

            Raises:
              ValueError: if cells is empty (not allowed), or at least one of the cells
                returns a state tuple but the flag `state_is_tuple` is `False`.
            """
        super(MyMultiRNNCell, self).__init__(cells=cells, state_is_tuple=state_is_tuple)


def batch_normalization(x, name_scope, epsilon=1e-16):
    with tf.variable_scope(name_scope):
        norm = tf.norm(x, ord=2, axis=0, keep_dims=True)
        inv = tf.rsqrt(tf.square(norm) + epsilon)
        x *= inv
    return x


def _batch_normalization(x, norm, scale=None, norm_epsilon=1e-16, name=None):
    """
    Normalizes a tensor by norm, and applies (optionally) a 'scale' \\(\gamma\\) to it, as well as
    an 'offset' \\(\beta\\):
             \\(\frac{\gamma(x)}{norm} + \beta\\)

    'norm', 'scale' are all expected to be of shape:
    * they can have the same number of dimensions as the input 'x', with identical sizes as 'x' for
      dimensions that are not normalized over, and dimension 1 for the others which are being normalized
      over
    :param x: 'Tensor'
    :param norm:
    :param scale:
    :param norm_epsilon:
    :param name:
    :return:
        the normalized, scaled, offset tensor
    """

    with tf.name_scope(name, "batchnorm", [x, norm, scale]):
        inv = tf.rsqrt(tf.square(norm) + norm_epsilon)
        if scale is not None:
            inv *= scale

        # def _debug_print_func(f):
        #     print("inv={}".format(f))
        #     return False
        # debug_print_op = tf.py_func(_debug_print_func,[inv],[tf.bool])
        # with tf.control_dependencies(debug_print_op):
        #     inv = tf.identity(inv, name="scale")

        x *= inv

        # def _debug_print_func(f):
        #     print("x_bn[0]={}".format(f[0, :]))
        #     # print("x_bn: mean,max,min={},{},{}".format(f.mean(),f.max(),f.min()))
        #     return False
        #
        # debug_print_op = tf.py_func(_debug_print_func, [x], [tf.bool])
        # with tf.control_dependencies(debug_print_op):
        #     x = tf.identity(x, name="x_norm")
        return x


def _lrelu_eps(x, eps=1e-8, weight=0.1):
    """
    :param x: a tensor, feature
    :param eps: thresholding
    :param weight: the weight in LReLU, should not set too large
    :return: (1-weight)*eps + weight*x if x < eps else x
    a soft thresholding operator constructed by LReLU
    moreover, weight can be generated randomly
    """

    negetive_part = tf.nn.relu(eps - x)
    parative_part = tf.nn.relu(x - eps)
    x_out = parative_part - weight * negetive_part
    x_out = x_out + eps
    return x_out
