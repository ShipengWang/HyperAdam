import tensorflow as tf
from tensorflow.python.ops import gradients
import numpy as np
import math
import os
from main import log
import matplotlib.pyplot as plt
import timeit
# batch size = 1

curr_path = os.getcwd()
parent_path = os.path.split(curr_path)[0]
graph_path = os.path.join(parent_path, "graph")
if not os.path.exists(graph_path):
    os.makedirs(graph_path)


class BasicNNOptModel:
    '''Base class for all RNN optimizers.

    Provide the functions needed to train an RNN optimizer.
    '''

    def weight_initializer(self, seed=None):
        return tf.truncated_normal_initializer(stddev=0.1, seed=seed)

    def bias_initializer(self):
        return tf.constant_initializer(0.0)

    def _get_variable(self, getter, name, *args, **kwargs):
        kwargs['trainable'] = self.is_training
        if not self.is_training:
            kwargs['collections'] = [tf.GraphKeys.MODEL_VARIABLES]
        return getter(name, *args, **kwargs)

    # placeholder

    def ph(self, shape, dtype=tf.float32):
        return tf.placeholder(dtype, shape=shape)

    def fc(self, x, c, seed=None, use_bias=True):
        n = x.get_shape()[1]
        w = tf.get_variable("w", [n, c], initializer=self.weight_initializer(seed))
        if use_bias:
            b = tf.get_variable("b", [c], initializer=self.bias_initializer())
            return tf.matmul(x, w) + b
        else:
            return tf.matmul(x, w)

    def __init__(self, name, optimizee=None, n_bptt_steps=None, n_steps=100,
                 decay_step=20, decay_rate=0.96, stair_case=False, lr_decay_name=None,
                 lr=1e-4, use_avg_loss=False, is_training=True, optimizer_name='adam',
                 **kwargs):
        self.name = name
        self.is_training = is_training
        self.kwargs = kwargs
        if self.is_training:
            self.optimizee = optimizee
            self.optimizer_name = optimizer_name
            self.x_dim = optimizee.get_x_dim()
            self.f = optimizee.loss
            self.n_bptt_steps = n_bptt_steps
            self.n_steps = n_steps
            self.lr_decay_name = lr_decay_name
            self.decay_steps = decay_step
            self.decay_rate = decay_rate
            self.stair_case = stair_case
            self.train_lr = lr  # TODO lr
            self.use_avg_loss = use_avg_loss
        else:
            self.x_dim = 233

        self.session = tf.get_default_session()

        self._build()

        self.bid = 0

    def _get_fx(self, f, i, x):
        if isinstance(f, list):
            return f[0], f[1]
        fx = f(i, x)
        grad = gradients.gradients(fx, x)[0]
        return fx, grad

    def _deepmind_log_encode(self, x, p=10.0):
        # gradient preprocessing
        xa = tf.log(tf.maximum(tf.abs(x), math.exp(-p))) / p
        xb = tf.clip_by_value(x * math.exp(p), -1, 1)
        return tf.stack([xa, xb], 1)

    def _build_pre(self, size):
        pass

    def _build_input(self):
        self.x = tf.placeholder(tf.float32, shape=[self.x_dim])
        self.input_state = [self.x]

    def _build_initial(self):
        self.initial_state = [self.x]

    def _build_loop(self):
        with tf.variable_scope("loop") as self.loop_scope:
            self.states = []

            state = self.input_state

            self.all_internal_loss = []
            self.optimizee_grad = []

            for i in range(self.n_bptt_steps):
                state, fx = self.iter(self.f, i, state)
                self.states.append(state)
                self.all_internal_loss.append(fx)
                if i == 0:
                    self.loop_scope.reuse_variables()

    def _build_loss(self):
        if self.use_avg_loss:
            self.loss = tf.reduce_mean(self.all_internal_loss)  # TODO tf.log
        else:
            self.loss = self.all_internal_loss[-1]

    def _build_optimizer(self):
        if self.optimizer_name == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.train_lr)
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.train_lr)
        elif self.optimizer_name == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.train_lr, self.kwargs['beta'])
        self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.all_vars)
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients)

    def _build(self):
        with tf.variable_scope("nn_opt", custom_getter=self._get_variable) as scope:
            graph_file_path = os.path.join(graph_path, self.name + "_data")
            self.summary_writer_train = tf.summary.FileWriter(graph_file_path + "/train",
                                                              self.session.graph)
            self.summaries = []
            self._build_pre(size=20)
            self._build_input()
            self._build_initial()
            if self.is_training:
                self._build_loop()
                self._build_loss()
                self.all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                self.against_apply_gradients = None
                self._build_optimizer()
                self.summaries.append(tf.summary.scalar('train_loss', self.loss))
            else:
                with tf.variable_scope("loop") as self.loop_scope:
                    self.optimizee_grad = []
                    state = self.input_state
                    self.next_state, self.next_fx = self.iter([self.ph([]), self.ph([None])], 0, state)
                self.all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope=scope.name)

            self.saver = tf.train.Saver(max_to_keep=None, var_list=self.all_vars, allow_empty=True)

    # return state, fx
    def iter(self, f, i, state):
        x, = state
        fx, grad = self._get_fx(f, i, x)

        input_x = tf.expand_dims(tf.concat(0, [x, grad, [fx]]), 0)

        out_x = self.fc(input_x, self.x_dim)

        x += out_x[0]
        return [x], fx

    def train_one_iteration(self, n_steps):
        self.bid += 1
        session = tf.get_default_session()

        ret = {}

        internal_feed_dict = self.optimizee.next_internal_feed_dict()
        val_x = self.optimizee.get_initial_x()  # initial weights of optimizee
        val_initial_state = session.run(self.initial_state, feed_dict={self.x: val_x})
        val_state = val_initial_state

        losses = []
        for i in range(n_steps / self.n_bptt_steps):
            feed_dict = internal_feed_dict
            feed_dict.update(dict(zip(self.input_state, val_state)))
            feed_dict.update(self.optimizee.next_feed_dict(self.n_bptt_steps))
            apply_grad, grad, val_state_all, val_loss, summaries_str, \
            optimizee_loss, optimizee_grad \
                = session.run([self.apply_gradients, self.gradients, self.states, self.loss, self.summaries,
                               self.all_internal_loss, self.optimizee_grad[0:-1]],
                              feed_dict=feed_dict)
            val_state = val_state_all[-1]  
            losses.append(val_loss)
        ret['loss'] = np.mean(losses)

        for summary_str in summaries_str:
            self.summary_writer_train.add_summary(summary_str, self.bid)
        self.summary_writer_train.flush()

        return ret

    def prepare_train_optimizee(self, tests):
        with tf.variable_scope("nn_opt") as scope:
            self.tests = {}
            for name, test in tests.items():
                if test['frequency'] == 0:
                    continue
                optimizee = test['optimizee']
                optimizee.build()
                with tf.variable_scope("loop") as loop_scope:
                    loop_scope.reuse_variables()
                    state, fx = self.iter(optimizee.loss, 0, self.input_state)

                gd_x = tf.Variable(tf.zeros([optimizee.x_dim]))  # weights of optimizee
                gd_fx = optimizee.loss(0, gd_x)  # loss of optimizee
                old = set(tf.global_variables())
                gd = test['gd']()
                grad_var = gd.compute_gradients(gd_fx, gd_x)
                gd_step = gd.apply_gradients(grad_var)
                # gd_step = gd.minimize(gd_fx)  # gradient descent , train optimizee by gradient descent
                gd_vars = list(set(tf.global_variables()) - old)  # only contains variables in gradient opt

                self.tests[name] = {
                    'frequency': test['frequency'],
                    'state': state,
                    'fx': fx,
                    'optimizee': optimizee,
                    'gd_x': gd_x,  # gradient descent
                    'gd_fx': gd_fx,
                    'grad_var': grad_var,
                    'gd': gd,
                    'gd_step': gd_step,
                    'gd_vars': gd_vars,
                    'n_steps': test['n_steps']
                }

    def prepare_train_optimizee_time(self, tests):
        with tf.variable_scope("nn_opt") as scope:
            self.tests = {}
            for name, test in tests.items():
                if test['frequency'] == 0:
                    continue
                optimizee = test['optimizee']
                optimizee.build()
                with tf.variable_scope("loop") as loop_scope:
                    loop_scope.reuse_variables()
                    state, fx = self.iter(optimizee.loss, 0, self.input_state)

                self.tests[name] = {
                    'frequency': test['frequency'],
                    'state': state,
                    'fx': fx,
                    'optimizee': optimizee,
                    'n_steps': test['n_steps']
                }

    def test(self, eid, gd=1):
        session = tf.get_default_session()

        for name, test in self.tests.items():
            if test['frequency'] == 0:
                continue
            if eid % test['frequency'] != 0:
                continue
            internal_feed_dict = test['optimizee'].next_internal_feed_dict()
            n_steps = test['n_steps']
            # n_steps: how many times of step descent every epoch

            val_x = test['optimizee'].get_initial_x()
            # initial weights of optimizee,
            # return different parameters of optimizee when invoked every time
            val_gd_x = np.copy(val_x)

            # opt
            val_state = session.run(self.initial_state, feed_dict={self.x: val_x})
            # gd
            session.run(tf.variables_initializer(test['gd_vars']))
            gd_x = session.run(test['gd_x'].assign(val_gd_x))  # pass parameter value of optimizee, same with val_x.

            losses = []
            gd_losses = []
            val_final_loss = 0.0
            val_gd_final_loss = 0.0
            s = 1
            for j in range(s):
                data_dicts = []
                for i in range(n_steps / s):
                    data_dicts.append(test['optimizee'].next_feed_dict(1))

                num_G = []
                num_D = []
                num_A = []
                num_AD = []
                for i in range(n_steps / s):
                    feed_dict = internal_feed_dict
                    feed_dict.update(dict(zip(self.input_state, val_state)))
                    feed_dict.update(data_dicts[i])
                    x_p = val_state[0]
                    #  , optimizee_grad self.optimizee_grad[-1]
                    val_state, loss \
                        = session.run([test['state'], test['fx']], feed_dict=feed_dict)
                    losses.append(loss)

                # loss on the whole test datasets (may be a batch, when flag.n_tests !=1)
                for i in range(n_steps / s):
                    feed_dict = internal_feed_dict
                    feed_dict.update(dict(zip(self.input_state, val_state)))
                    feed_dict.update(data_dicts[i])
                    val_loss = session.run(test['fx'], feed_dict=feed_dict)
                    val_final_loss += val_loss

                if gd == 1:
                    for i in range(n_steps / s):
                        gd_x_prev = gd_x
                        feed_dict_gd = internal_feed_dict
                        feed_dict_gd.update(data_dicts[i])
                        _, gd_loss, grad = session.run([test['gd_step'], test['gd_fx'], test['grad_var'][0][0]],
                                                       feed_dict=feed_dict_gd)
                        gd_x = session.run(test['grad_var'][0][1], feed_dict=feed_dict_gd)
                        gd_losses.append(gd_loss)

                        # training phase with gradient descent, get weights

                    # loss on the whole test datasets(may be a batch, when flag.n_tests !=1)
                     for i in range(n_steps / s):
                         feed_dict = internal_feed_dict
                         feed_dict.update(data_dicts[i])
                         val_gd_final_loss += session.run(test['gd_fx'], feed_dict=feed_dict)
            val_final_loss /= n_steps
            val_gd_final_loss /= n_steps

            yield name, val_final_loss, val_gd_final_loss, losses, gd_losses

    def test_time(self, eid):
        session = tf.get_default_session()

        for name, test in self.tests.items():
            if test['frequency'] == 0:
                continue
            if eid % test['frequency'] != 0:
                continue
            internal_feed_dict = test['optimizee'].next_internal_feed_dict()
            n_steps = test['n_steps']
            # n_steps: how many times of step descent every epoch

            val_x = test['optimizee'].get_initial_x()
            # initial weights of optimizee,
            # return different parameters of optimizee when invoked every time

            # opt
            val_state = session.run(self.initial_state, feed_dict={self.x: val_x})

            losses = []
            duration_steps = []
            total_time = timeit.default_timer()
            for i in range(n_steps):
                feed_dict = internal_feed_dict
                feed_dict.update(dict(zip(self.input_state, val_state)))
                data_dict = test['optimizee'].next_feed_dict(1)
                feed_dict.update(data_dict)
                step_time_start = timeit.default_timer()
                val_state, loss \
                    = session.run([test['state'], test['fx']], feed_dict=feed_dict)
                duration_steps.append(timeit.default_timer() - step_time_start)
                losses.append(loss)
            duration = timeit.default_timer() - total_time
            yield name, losses, duration_steps, duration

    def restore(self, eid):
        data_path = os.path.join(parent_path, "training_data")
        filename = os.path.join(data_path, self.name + "_data")
        self.saver.restore(self.session, filename + "/epoch-%d" % eid)
        print self.name, "restored."

    def save(self, eid):
        data_path = os.path.join(parent_path, "training_data")
        folder = os.path.join(data_path, self.name + "_data")
        filename = "%s/epoch" % folder
        sfilename = "%s/epoch-last" % folder

        self.saver.save(self.session, filename, global_step=eid)
        os.unlink("%s-%d.meta" % (filename, eid))
        if os.path.lexists(sfilename):
            os.unlink(sfilename)
        os.symlink("epoch-%d" % eid, sfilename)
        print self.name, "saved."

    def save_best(self, eid):
        data_path = os.path.join(parent_path, "training_data")
        folder = os.path.join(data_path, self.name + "_data")
        filename = "{}/epoch".format(folder)
        self.saver.save(self.session, filename, global_step=eid)
        os.unlink("%s-%d.meta" % (filename, eid))
        file_name = os.path.join(parent_path, self.name + "-best-evaluation.txt")
        log("best meta-optimizer up to now: eid = {}".format(eid), file_name)
        print self.name, "saved."


import deepmind
import rnn
import metalstm

__all__ = ['deepmind', 'rnn', 'metalstm']
