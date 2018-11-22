'''The model, tricks and optimizee used when training an RNN optimizer.'''

import nn_opt
import optimizee
import test_list

tasks = {
    'rnnprop': {
        'model': nn_opt.rnn.RNNpropModel,
        'optimizee': {
            'train': optimizee.tricks.Mixed({
                optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.mnist.MnistLinearModel()): 1.0,
                optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.trivial.Square(x_dim=20), 1.0): 1.0
            }),
            'tests': test_list.tests
        },
        'use_avg_loss': False,
        'training lr': 1e-4,
    },
    'deepmind-lstm-avg': {
        'model': nn_opt.deepmind.LSTMOptModel,
        'optimizee': {
            'train': optimizee.mnist.MnistLinearModel(),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr' : 1e-4,
    },
    'adam-lstm-adacom-avg-tricky': {
        'model': nn_opt.metalstm.AdamLstmAdaWeightOptModel,
        'optimizee': {
            'train': optimizee.tricks.Mixed({
                optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.mnist.MnistLinearModel()): 1.0, # TODO n_l
                optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.trivial.Square(x_dim=20), 1.0): 1.0
            }),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr': 1e-4,
        # 'lr_decay_name': 'exp'
    },
    'adam-lstm-adacom-avg-RS': {
        'model': nn_opt.metalstm.AdamLstmAdaWeightOptModel,
        'optimizee': {
            'train': optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.mnist.MnistLinearModel()),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr': 1e-4,
        # 'lr_decay_name': 'exp'
    },
    'adam-lstm-adacom-avg-CC': {
        'model': nn_opt.metalstm.AdamLstmAdaWeightOptModel,
        'optimizee': {
            'train': optimizee.tricks.Mixed({
                optimizee.mnist.MnistLinearModel(): 1.0, # TODO n_l
                optimizee.trivial.Square(x_dim=20): 1.0
            }),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr': 5e-5,
        # 'lr_decay_name': 'exp'
    },
    'adam-lstm-adacom-avg': {
        'model': nn_opt.metalstm.AdamLstmAdaWeightOptModel,
        'optimizee': {
            'train': optimizee.mnist.MnistLinearModel(),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr': 1e-4,
        # 'lr_decay_name': 'exp'
    },
    'adam-lstm-adacom-21-avg-tricky': {
        'model': nn_opt.metalstm.AdamLstmOptModelH,
        'optimizee': {
            'train': optimizee.tricks.Mixed({
                optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.mnist.MnistLinearModel()): 1.0, # TODO n_l
                optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.trivial.Square(x_dim=20), 1.0): 1.0
            }),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr': 1e-4,
        # 'lr_decay_name': 'exp'
    },
    'adam-lstm-adacom-avg-tricky-lstm': {
        'model': nn_opt.metalstm.AdamLstmAdaWeightOptModelLSTM,
        'optimizee': {
            'train': optimizee.mnist.MnistLinearModel(),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr': 1e-4,
        # 'lr_decay_name': 'exp'
    },
    'adam-lstm-adacom-avg-tricky-pre': {
        'model': nn_opt.metalstm.AdamLstmAdaWeightOptModelPre,
        'optimizee': {
            'train': optimizee.mnist.MnistLinearModel(),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr': 1e-4,
        # 'lr_decay_name': 'exp'
    },
    'adam-lstm-adacom-avg-tricky-neither': {
        'model': nn_opt.metalstm.AdamLstmAdaWeightOptModelNeither,
        'optimizee': {
            'train': optimizee.mnist.MnistLinearModel(),
            'tests': test_list.tests
        },
        'use_avg_loss': True,
        'training lr': 1e-4,
        # 'lr_decay_name': 'exp'
    },

}
