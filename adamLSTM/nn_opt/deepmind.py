import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell
import nn_opt


# batch size = 1


class LSTMOptModel(nn_opt.BasicNNOptModel):
    def lstm_cell(self):
        return LSTMCell(num_units=self.dimH)

    def _build_pre(self, size):
        self.dimH = size
        self.num_of_layers = 2
        self.cellH = MultiRNNCell([self.lstm_cell() for _ in range(self.num_of_layers)])
        self.lr = 0.1

    def _build_input(self):
        self.x = self.ph([None])
        self.cellH_state = tuple((self.ph([None, size.c]), self.ph([None, size.h])) for size in self.cellH.state_size)
        self.input_state = [self.x, self.cellH_state]

    def _build_initial(self):
        x = self.x  # weights of optimizee
        cellH_state = self.cellH.zero_state(tf.size(x), tf.float32)
        self.initial_state = [x, cellH_state]

    # return state, fx
    def iter(self, f, i, state):
        x, cellH_state = state

        fx, grad = self._get_fx(f, i, x)
        self.optimizee_grad.append(grad)
        grad = tf.stop_gradient(grad)

        last = self._deepmind_log_encode(grad)


        with tf.variable_scope("cellH"):
            last, cellH_state = self.cellH(last, cellH_state)

        with tf.variable_scope("fc"):
            last = self.fc(last, 1)

        delta_x = last[:, 0] * self.lr

        x += delta_x
        return [x, cellH_state], fx
