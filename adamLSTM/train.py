import tensorflow as tf
import numpy as np
import sys
import os
import task_list
import matplotlib.pyplot as plt
import cPickle as pickle
from main import log, randid, task_id

flags = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("task", "adam-lstm-adacom-avg-tricky",
                           "the name of the task")  # deepmind-lstm-avg adam-lstm-avg-tricky
tf.app.flags.DEFINE_string("id", randid(), "id")  # training phase randid()
tf.app.flags.DEFINE_integer("gpu", 0, "gpu id")
tf.app.flags.DEFINE_integer("eid", 0, "the epoch id to continue training")
tf.app.flags.DEFINE_integer("n_steps", 100, "the number of iterations in training RNN optimizer")
tf.app.flags.DEFINE_integer("n_bptt_steps", 20, "the number of iterations in training RNN optimize")
tf.app.flags.DEFINE_integer("n_batches", 100, "#batches per epoch")
tf.app.flags.DEFINE_integer("n_tests", 10, "#batches of  per epoch at the test stage")
tf.app.flags.DEFINE_integer("n_epochs", 700, "#epochs")


def train_optimizer(task):
    '''Train an RNN optimizer.

    Args:
        task: A dictionary in task_list.py which specifies the model of the optimizer and
            the tricks and the optimizee to use when training the optimizer.
    '''
    flags = tf.app.flags.FLAGS
    session = tf.get_default_session()

    curr_path = os.getcwd()
    parent_path = os.path.split(curr_path)[0]

    task['optimizee']['train'].build()

    n_steps = task['n_steps'] if 'n_steps' in task else flags.n_steps
    n_bptt_steps = task['n_bptt_steps'] if 'n_bptt_steps' in task else flags.n_bptt_steps
    use_avg_loss = task['use_avg_loss'] if 'use_avg_loss' in task else False
    options = task['options'] if 'options' in task else {}
    lr_decay_name = task['lr_decay_name'] if 'lr_decay_name' in task else None
    assert n_steps % n_bptt_steps == 0

    model = task['model'](
        name=task_id(),
        optimizee=task['optimizee']['train'],
        n_bptt_steps=n_bptt_steps,
        n_steps=n_steps, decay_step=20, lr_decay_name=lr_decay_name,
        decay_rate=0.96, stair_case=False,
        lr=task['training lr'],
        use_avg_loss=use_avg_loss,
        **options)
    model.prepare_train_optimizee(task['optimizee']['tests'])

    log_path = os.path.join(parent_path, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_filename = os.path.join(log_path, model.name + "_training.txt")

    fig_path = os.path.join(parent_path, "training_data", model.name + "_data")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_filename = os.path.join(fig_path, model.name + ".png")

    session.run(tf.global_variables_initializer())

    eid = flags.eid
    if eid != 0:
        model.restore(flags.eid)
        model.bid = flags.eid * flags.n_batches

        file_name = os.path.join(fig_path, "loss_curve.pkl")
        pickle_file = open(file_name, "rb")
        loss_curve = pickle.load(pickle_file)
        pickle_file.close()
        train_loss = loss_curve[:eid]
        val_loss = loss_curve[eid:2 * eid]
        val_gd_loss = loss_curve[2 * eid:3 * eid]
        val_loss1 = loss_curve[3 * eid:4 * eid]
        val_gd_loss1 = loss_curve[4 * eid:5*eid]
    elif eid == 0:
        train_loss = []
        val_loss = []
        val_gd_loss = []
        val_loss1 = []
        val_gd_loss1 = []
    else:
        print("check eid")

    best_evaluation = float("inf")

    while flags.n_epochs == 0 or eid < flags.n_epochs:
        eid += 1

        loss_values = []
        for i in range(flags.n_batches):
            ret = model.train_one_iteration(n_steps)

            loss_value = ret['loss']

            loss_values.append(loss_value)

            # sys.stdout.write("\r\033[K")
            msg = "\riteration #%d" % i
            msg += ": loss = %.5f avg loss = %.5f" % (loss_value, np.mean(loss_values))  # mean loss of a batch
            sys.stdout.write(msg)
            sys.stdout.flush()

        # sys.stdout.write("\r\033[K")
        msg = "\repoch #%d" % eid
        msg += ": loss = %.5f" % np.mean(loss_values)
        log(msg, log_filename)
        log(str(loss_values), log_filename)
        train_loss.append(np.mean(loss_values))

        if eid % 10 == 0:
            model.save(eid)

        test_loss_values = {}
        for i in range(flags.n_tests):  # flag.n_tests: batch size in test
            for name, avg_loss_value, gd_avg_loss_value, test_loss, gd_test_loss in model.test(eid):
                if name not in test_loss_values:
                    test_loss_values[name] = {'nn': [], 'gd': [], 'nn1': [], 'gd1': []}
                test_loss_values[name]['nn'].append(avg_loss_value)
                test_loss_values[name]['gd'].append(gd_avg_loss_value)
                test_loss_values[name]['nn1'].append(np.mean(test_loss))
                test_loss_values[name]['gd1'].append(np.mean(gd_test_loss))
        for name in test_loss_values:
            log("epoch #%d test %s: loss = %.5f gd_loss = %.5f" % (eid, name,
                                                                   np.mean(test_loss_values[name]['nn']),
                                                                   np.mean(test_loss_values[name]['gd'])), log_filename)
            val_loss.append(np.mean(test_loss_values[name]['nn']))
            val_gd_loss.append(np.mean(test_loss_values[name]['gd']))
            val_loss1.append(np.mean(test_loss_values[name]['nn1']))
            val_gd_loss1.append(np.mean(test_loss_values[name]['gd1']))
            fig_legend = "val_gd_" + task['optimizee']['tests'][name]['gdAlgo']

        if val_loss[-1] < best_evaluation:
            model.save_best(eid)
            best_evaluation = val_loss[-1]

        if eid % 10 == 0:
            file_name = os.path.join(fig_path, "loss_curve.pkl")
            pickle_file = open(file_name, "wb")
            pickle.dump(train_loss + val_loss + val_gd_loss + val_loss1 + val_gd_loss1, pickle_file)
            pickle_file.close()

        plt.figure(1)
        plt.plot(np.arange(1, eid + 1, 1), train_loss, "r-")
        plt.plot(np.arange(1, eid + 1, 1), val_loss, "b-")
        plt.plot(np.arange(1, eid + 1, 1), val_gd_loss, "k-")

        plt.plot(np.arange(1, eid + 1, 1), val_loss1, "b--")
        plt.plot(np.arange(1, eid + 1, 1), val_gd_loss1, "k--")

        plt.legend(["train", "val", fig_legend, 'val_nn', 'val_gd'])
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.draw()
        plt.pause(0.0001)
        plt.savefig(fig_filename)


def main(_):
    graph = tf.Graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu)
    gpu_options = tf.GPUOptions(allow_growth=True)

    tasks = task_list.tasks

    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as session:
            train_optimizer(tasks[flags.task])
            session.close()


if __name__ == '__main__':
    tf.app.run()
