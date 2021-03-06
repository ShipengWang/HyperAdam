HyperAdam: A Learnable Task-Adaptive Adam for Network Training
=
Introduction
-
The current project page provides tensorflow code that implements our AAAI2019 paper:

**Title**:  "HyperAdam: A Learnable Task-Adaptive Adam for Network Training"

**Authors**: Shipeng Wang, Jian Sun*, Zongben Xu

**Email:** wangshipeng8128@stu.xjtu.edu.cn

**Institution**: School of Mathematics and Statistics, Xi'an Jiaotong University

**Code**:  https://github.com/ShipengWang/HyperAdam-Tensorflow, https://github.com/ShipengWang/Variational-HyperAdam (PyTorch)

**Link**: https://ojs.aaai.org//index.php/AAAI/article/view/4466

**Arxiv**: https://arxiv.org/abs/1811.08996

**Abstract**:

Deep neural networks are traditionally trained using human-designed stochastic optimization algorithms, such as SGD and Adam. Recently, the approach of learning to optimize network parameters has emerged as a promising research topic. However, these learned black-box optimizers sometimes do not fully utilize  the experience in human-designed optimizers, therefore have limitation in generalization ability. In this paper, a new optimizer, dubbed as \textit{HyperAdam}, is proposed that combines the idea of ``learning to optimize'' and traditional Adam optimizer. Given a network for training, its parameter update in each iteration generated by HyperAdam is an adaptive combination of multiple updates generated by Adam with varying decay rates. The combination weights and decay rates in HyperAdam are adaptively learned depending on the task.  HyperAdam is  modeled as a recurrent neural network with AdamCell, WeightCell and StateCell. It is justified to be state-of-the-art for various network training, such as multilayer perceptron, CNN and LSTM.

Citing our paper
-
If the code is useful in your research, please cite our AAAI2019 paper:

@inproceedings{wang2019hyperadam,

title={HyperAdam: A Learnable Task-Adaptive Adam for Network Training},

author={Wang, Shipeng and Sun, Jian and Xu, Zongben},

booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},

year={2019},

pages={5297--5304},

volume={33},

}

Requirements
-
+ python 2.7
+ TensorFlow 1.5 or PyTorch 1.0



