import tensorflow as tf
import numpy as np
import os
import task_list
import cPickle as pickle
from main import log, task_id
import time

flags = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("task", "adam-lstm-adacom-avg-tricky", "the name of the task")  # adam-lstm-adacom-avg-tricky deepmind-lstm-avg # adam-lstm-avg-tricky
tf.app.flags.DEFINE_string("id", "umyzst", "id") 
tf.app.flags.DEFINE_integer("gpu", 0, "gpu id")
tf.app.flags.DEFINE_integer("eid", 700, "the epoch id to continue training")
tf.app.flags.DEFINE_integer("n_tests", 100, "#batches of  per epoch at the test stage")
tf.app.flags.DEFINE_integer("n_epochs", 1, "#epochs")
tf.app.flags.DEFINE_integer("exception", 0, "#whether delete some bad result")
tf.app.flags.DEFINE_integer("gd", 0, "#whether run gradient descent")


def optimizer_train_optimizee(task):
    '''Test a trained RNN optimizer on different optimizees.

    Args:
        task: A dictionary in task_list.py which specifies the RNN optimizer.
    '''

    test_names = [
        # 'mnist-nn-sigmoid-100',  
        # 'mnist-nn-relu-100',   
        # 'mnist-nn-sigmoid-2000',   
        # 'mnist-nn-sigmoid-10000',  
        # 'mnist-nn-sigmoid-50000',   
        # 'vgg-mnist-fc2-conv4-pool2-10000',    
        # 'vgg-mnist-fc2-conv4-pool2-2000',
        # 'mnist-nn-l9-sigmoid-10000',
        # 'vgg-cifar-fc2-conv4-pool2-2000',
        # 'vgg-cifar-fc2-conv4-pool2-10000',
        # 'vgg-cifar-fc1-conv2-pool1-10000',    
        # 'vgg-mnist-fc2-conv4-pool2-10000',     
        # 'vgg-cifar-fc2-conv4-pool2-10000',   
        # 'vgg-cifar-fc1-conv2-pool1-100',
        # 'vgg-mnist-fc1-conv2-pool1-100-dropout',  
        # 'vgg-cifar-fc1-conv2-pool1-100-dropout',  
        # 'vgg-mnist-fc2-conv4-pool2-100-dropout',   
        # 'vgg-cifar-fc2-conv4-pool2-100-dropout',   
        # 'vgg-mnist-fc1-conv2-pool1-100-bn',   
        # 'vgg-cifar-fc1-conv2-pool1-100-bn',   
        # 'vgg-mnist-fc2-conv4-pool2-100-bn',   
        # 'vgg-cifar-fc2-conv4-pool2-100-bn',   
        # 'mnist-nn-elu-100',  
        # 'mnist-nn-tanh-100',  
        # 'mnist-nn-l2-sigmoid-100',  
        # 'mnist-nn-l3-sigmoid-100',  
        # 'mnist-nn-l3-sigmoid-2000',
        'mnist-nn-l4-sigmoid-100',  
        # 'mnist-nn-l5-sigmoid-100',  
        # 'mnist-nn-l9-sigmoid-2000',
        # 'mnist-nn-l6-sigmoid-100',  
        # 'mnist-nn-l7-sigmoid-100',  
        # 'mnist-nn-l8-sigmoid-100',  
        # 'mnist-nn-l9-sigmoid-100',  
        # 'mnist-nn-l10-sigmoid-100', 
        # 'sin_lstm', 
        #  'sin_lstm-x2',  
        # 'sin_lstm-no001', 
        # 'stochasticlinear',
        # 'sin_lstm-x2-10000',
    ]

    parent_path = os.path.split(os.getcwd())[0]
    data_path = os.path.join(parent_path, "test_data")

    tests = {}
    curr_path = {}
    assert flags.eid != 0
    for name in test_names:
        tests[name] = task['optimizee']['tests'][name]
        tests[name]['frequency'] = 1
        if tests[name]['frequency'] == 1:
            curr_path[name] = os.path.join(data_path, '%s' % name, 'all')
            if not os.path.exists(curr_path[name]):
                os.makedirs(curr_path[name])

    options = task['options'] if 'options' in task else {}

    model = task['model'](
        name=task_id(),
        is_training=False,
        **options)
    model.prepare_train_optimizee(tests)

    log_path = os.path.join(parent_path, "log")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_filename = os.path.join(log_path, model.name + "_test.txt" )

    model.restore(flags.eid)
    log("model %s restored at epoch #%d" % (task_id(), flags.eid), log_filename)

    eid = 0
    test_loss_values = {}
    test_loss = {}

    while flags.n_epochs == 0 or eid < flags.n_epochs:
        eid += 1
        for i in range(flags.n_tests):
            for name, avg_loss_value, gd_avg_loss_value, losses, gd_losses in model.test(eid, flags.gd):
                if np.max(losses) > 100 and flags.exception == 1:
                    print("exception")
                    i = i - 1
                    continue
                if name not in test_loss_values:
                    test_loss_values[name] = {'nn': [], 'gd': []}
                    test_loss[name] = {'nn': np.zeros(np.array(losses).shape), 'gd': np.zeros(np.array(losses).shape)}
                test_loss_values[name]['nn'].append(avg_loss_value)
                test_loss[name]['nn'] += np.array(losses)
                if flags.gd == 1:
                    test_loss_values[name]['gd'].append(gd_avg_loss_value)
                    test_loss[name]['gd'] += np.array(gd_losses)
                print("test #%d:" % i + name + " val_final_loss %f"
                      % avg_loss_value + " val_gd_final_loss %f" % gd_avg_loss_value)
                # log("batch={batch}:loss_curve={loss_curve}" .format(batch=str(i), loss_curve=losses),
                #     os.path.join(log_path, model.name + "_" + name + "_test.txt"))
        for name in sorted(test_loss_values):
            log("epoch #%d: test %s: loss = %.5f gd_loss = %.5f" % (eid, name,
                                                                    np.mean(test_loss_values[name]['nn']),
                                                                    np.mean(test_loss_values[name]['gd'])),
                log_filename)
    for name in sorted(test_loss):
        file_name = "%s_100.pkl" % (task_id()[:-7])  # TODO 100
        nn_file_name = os.path.join(curr_path[name], file_name)
        pickle_file = open(nn_file_name, 'wb')  # wb
        pickle.dump(test_loss[name]['nn']/flags.n_tests, pickle_file)
        pickle_file.close()

        if flags.gd == 1:
            file_name = task['optimizee']['tests'][name]['gdAlgo'] + "_100.pkl"
            nn_file_name = os.path.join(curr_path[name], file_name)
            pickle_file = open(nn_file_name, 'wb')  # wb
            pickle.dump(test_loss[name]['gd'] / flags.n_tests, pickle_file)
            pickle_file.close()


def main(_):
    graph = tf.Graph()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags.gpu)
    gpu_options = tf.GPUOptions(allow_growth=True)

    tasks = task_list.tasks

    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=graph) as session:

            optimizer_train_optimizee(tasks[flags.task])


if __name__ == '__main__':
    tf.app.run()
