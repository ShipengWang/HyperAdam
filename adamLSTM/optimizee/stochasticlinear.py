import tensorflow as tf
import numpy as np
import optimizee
import matplotlib.pyplot as plt
from amsgrad import AmsGradOptimizer
import os

class StochLinear(optimizee.Optimizee):
    def __init__(self, x_dim):
        optimizee.Optimizee.__init__(self)
        self.x_dim = x_dim
        self.t = np.zeros(shape=[self.x_dim])

    def build(self):
        self.a = tf.placeholder(dtype=tf.float32, shape=[self.x_dim])

    def get_x_dim(self):
        return self.x_dim

    def get_initial_x(self):
        return np.zeros(shape=[self.x_dim])

    def next_internal_feed_dict(self):
        return {}

    def next_feed_dict(self, n_iterations):
        # self.internal_feed_dict[self.a] = np.random.uniform(0.0, 3.0, size=[self.x_dim])
        self.t = self.t + 1.0
        return {self.a: self.t}

    def loss(self, i, x):
        return tf.cond(tf.equal(tf.mod(self.a, 101.0),
                                tf.constant(1.0, dtype=tf.float32, shape=[self.x_dim]))[0],
                       lambda: 1010 * x, lambda: -10.0 * x)


def main(_):
    session = tf.Session()
    step = 50000
    # build problem
    pro = StochLinear(1)
    pro.build()
    # input and loss
    x = tf.Variable(tf.zeros([pro.x_dim]))
    clip_x = tf.clip_by_value(x, -1.0, 1.0)
    fx = pro.loss(0, clip_x)

    # optimizer = tf.train.AdamOptimizer(1e-2, beta1=0.99, beta2=0.99)
    optimizer = AmsGradOptimizer(1e-3, beta1=0.9, beta2=0.99)
    grad = optimizer.compute_gradients(fx, var_list=x)
    apply_grad, new_state = optimizer.apply_gradients(grad)

    internal_feed_dict = pro.next_internal_feed_dict()

    session.run(tf.global_variables_initializer())

    coff_dict = []
    losses = 0
    reget = []
    xs = []
    states = []
    # for i in range(step):
    #     coff_dict.append(pro.next_feed_dict(1))
    for i in range(1, step+1, 1):
        # change the true gradient value in amsgrad, since clip_by_value make gradient zero for some extreme x
        beta1 = 0.1 if np.mod(i, 101) == 1 else 0.9
        beta2 = 0.4 if np.mod(i, 101) == 1 else 0.99
        feed_dict = internal_feed_dict
        # feed_dict.update(coff_dict[i])
        feed_dict.update(pro.next_feed_dict(1))
        feed_dict.update({optimizer._beta1: beta1})
        feed_dict.update({optimizer._beta2: beta2})
        # loss = session.run([fx], feed_dict=feed_dict)
        _, g, x_, loss, state = session.run([apply_grad, grad, clip_x, fx, new_state], feed_dict=feed_dict)
        xs.append(x_)
        states.append(state)

    ms = [states[i][0]['m'] for i in range(step)]
    vs = [states[i][0]['v'] for i in range(step)]
    vhats = [states[i][0]['vhat'] for i in range(step)]
    plt.figure(1)
    plt.plot(np.arange(1, step + 1, 1), xs, '-')
    plt.savefig(os.path.join(os.getcwd(), "xsams.png"))
    # plt.figure(2)
    # plt.plot(np.arange(1, step + 1, 1), reget, '-')
    # plt.savefig(os.path.join(os.getcwd(), "regetams.png"))
    plt.show()
    print("done")


if __name__ == '__main__':
    tf.app.run()
