#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import common.layers as layers
from collections import OrderedDict
from common.datasets import MNIST, MNISTCSV
from common.util import get_one_batch
from common.visualize import show_imgs, show_accuracy_loss, show_filter
import common.optimizer as optimizer
import pandas as pd


class LeNet(object):
    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv_param_1=None,
                 conv_param_2=None,
                 pool_param_1=None,
                 pool_param_2=None,
                 hidden_size_1=120,
                 hidden_size_2=84,
                 output_size=10,
                 weight_init_std=0.01):

        conv_1_output_h = (input_dim[1] - conv_param_1['filter_size'] + 2*conv_param_1['pad']) / conv_param_1['stride'] + 1
        pool_1_output_h = int(conv_1_output_h/pool_param_1['pool_h'])
        conv_2_output_h = (pool_1_output_h - conv_param_2['filter_size'] + 2*conv_param_2['pad']) / conv_param_2['stride'] + 1
        pool_2_output_size = int(conv_param_2['filter_num'] * (conv_2_output_h/pool_param_2['pool_h']) * (conv_2_output_h/pool_param_2['pool_h']))

        # init parameters
        self.params = {}
        # conv 1
        self.params['W1'] = weight_init_std * np.random.randn(conv_param_1['filter_num'], input_dim[0], conv_param_1['filter_size'], conv_param_1['filter_size'])  # 6,1,5,5
        self.params['b1'] = np.zeros(conv_param_1['filter_num'])  # 6,
        # conv 2
        self.params['W2'] = weight_init_std * np.random.randn(conv_param_2['filter_num'], conv_param_1['filter_num'], conv_param_2['filter_size'], conv_param_2['filter_size'])  # 16,1,5,5
        self.params['b2'] = np.zeros(conv_param_2['filter_num'])  # 16,
        # affine 1
        self.params['W3'] = weight_init_std * np.random.randn(pool_2_output_size, hidden_size_1)  # (N*4*4),100 => 4320,100
        self.params['b3'] = np.zeros(hidden_size_1)  # 100,
        # affine 2
        self.params['W4'] = weight_init_std * np.random.randn(hidden_size_1, hidden_size_2)  # 100,10
        self.params['b4'] = np.zeros(hidden_size_2)  # 10,
        # affine 3 --- out
        self.params['W5'] = weight_init_std * np.random.randn(hidden_size_2, output_size)  # 100,10
        self.params['b5'] = np.zeros(output_size)  # 10,

        # create layers
        self.layers = OrderedDict()
        # conv 1
        self.layers['Conv1'] = layers.Convolution(self.params['W1'], self.params['b1'], conv_param_1['stride'], conv_param_1['pad'])
        # relu 1
        self.layers['Relu1'] = layers.Relu()
        # pool 1
        self.layers['Pool1'] = layers.Pooling(pool_h=pool_param_1['pool_h'], pool_w=pool_param_1['pool_h'], stride=pool_param_1['pool_stride'])
        # conv 2
        self.layers['Conv2'] = layers.Convolution(self.params['W2'], self.params['b2'], conv_param_1['stride'], conv_param_1['pad'])
        # relu 2
        self.layers['Relu2'] = layers.Relu()
        # pool 2
        self.layers['Pool2'] = layers.Pooling(pool_h=pool_param_1['pool_h'], pool_w=pool_param_1['pool_h'], stride=pool_param_1['pool_stride'])
        # affine 1
        self.layers['Affine1'] = layers.Affine(self.params['W3'], self.params['b3'])
        # relu 3
        self.layers['Relu2'] = layers.Relu()
        # affine 2
        self.layers['Affine2'] = layers.Affine(self.params['W4'], self.params['b4'])
        # relu 4
        self.layers['Relu2'] = layers.Relu()
        # affine 3
        self.layers['Affine3'] = layers.Affine(self.params['W5'], self.params['b5'])
        # loss
        self.lossLayer = layers.SoftmaxCrossEntropy(class_num=10)

        self.loss_list = []

    def predict(self, x_batch):
        tmp = x_batch.copy()
        for layer in self.layers.values():
            tmp = layer.forward(tmp)
            # print(layer)
        return tmp

    def loss(self, x_batch, t_batch):
        y = self.predict(x_batch)
        ret = self.lossLayer.forward(y, t_batch)
        # print(self.lossLayer)
        return ret

    def accuracy(self, x_batch, t_batch):
        y = self.predict(x_batch)
        y = np.argmax(y, axis=1)
        # if t_batch.ndim != 1:
        #     tmp = t_batch.copy()
        #     tmp = np.argmax(tmp, axis=1)
        accuracy = np.sum(y == t_batch) / float(x_batch.shape[0])
        return accuracy

    def gradient(self, x_batch, t_batch):
        # forward
        loss = self.loss(x_batch, t_batch)
        self.loss_list.append(loss)
        # backward
        d_y = 1
        d_y = self.lossLayer.backward(d_y)
        layers_list = list(self.layers.values())
        layers_list.reverse()
        for layer in layers_list:
            d_y = layer.backward(d_y)
        # calculate gradients
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].d_W, self.layers["Conv1"].d_b
        grads["W2"], grads["b2"] = self.layers["Conv2"].d_W, self.layers["Conv2"].d_b
        grads["W3"], grads["b3"] = self.layers["Affine1"].d_W, self.layers["Affine1"].d_b
        grads["W4"], grads["b4"] = self.layers["Affine2"].d_W, self.layers["Affine2"].d_b
        grads["W5"], grads["b5"] = self.layers["Affine3"].d_W, self.layers["Affine3"].d_b

        return grads


def show_structure(net, x_batch, y_batch):
    ret = net.loss(x_batch, y_batch)
    for layer in network.layers.values():
        print(layer)
    print(net.lossLayer)
    print('****** Print structure with values: OK ******')
    return ret

    
def generate_submission_csv(id_column, predict_y, filename, write_index=False):
    df = pd.DataFrame({'ImageId': id_column, 'Label': predict_y.flatten()})
    df.to_csv(filename, index=write_index)

    
if __name__ == '__main__':
    mnist = MNISTCSV('data\\MNIST')
    # mnist = MNIST('data\\mnist')
    train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=False, label_one_hot=False)
    # # show sample images
    # train_x_batch, train_y_batch = get_one_batch(train_x, train_y, batch_size=5)
    # show_imgs(train_x_batch.reshape(-1, 28, 28), train_y_batch)

    learning_rate = 0.01
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    network = LeNet(input_dim=(1, 28, 28),
                    conv_param_1={'filter_num': 6, 'filter_size': 5, 'pad': 0, 'stride': 1},
                    conv_param_2={'filter_num': 16, 'filter_size': 5, 'pad': 0, 'stride': 1},
                    pool_param_1={'pool_h': 2, 'pool_stride': 2},
                    pool_param_2={'pool_h': 2, 'pool_stride': 2},
                    hidden_size_1=120,
                    hidden_size_2=84,
                    output_size=10,
                    weight_init_std=0.01)
    # for layer in network.layers.values():
    #     print(layer)
    # print(network.lossLayer)
    # print('****** Print structure without values: OK ******')

    train_x_batch, train_y_batch = get_one_batch(train_x, train_y, batch_size=10)
    show_structure(network, train_x_batch, train_y_batch)

    op = optimizer.Adam(lr=0.001)
    epoch = 100
    for i in range(5000):
        train_x_batch, train_y_batch = get_one_batch(train_x, train_y, batch_size=30)
        grads = network.gradient(train_x_batch, train_y_batch)
        try:
            op.update(network.params, grads)
        except ZeroDivisionError as e:
            print('Handling run-time error:', e)

        if i % epoch == 0:
            # calculate accuracy
            train_acc = network.accuracy(train_x_batch, train_y_batch)
            train_acc_list.append(train_acc)
            # test_acc = network.accuracy(test_x, test_y)
            # test_acc_list.append(test_acc)
            # print("train accuracy: {:.3f}".format(train_acc), "test accuracy: {:.3f}".format(test_acc))
            print("train accuracy: {:.3f}".format(train_acc))
            # calculate loss
            train_loss = network.loss(train_x_batch, train_y_batch)
            train_loss_list.append(train_loss)
            # test_loss = network.loss(test_x, test_y)
            # test_loss_list.append(test_loss)
            # print("train loss: {:.3f}".format(train_loss), "test loss: {:.3f}".format(test_loss))
            print("train loss: {:.3f}".format(train_loss))

    # show_accuracy_loss(train_acc_list, test_acc_list, train_loss_list, test_loss_list)
    # show_filter(network.params['W1'])
    pre_list = []
    for x in test_x:
        pre = network.predict(x.reshape(1, 1 , 28, 28))
        pre = np.argmax(pre, axis=1)
        pre_list.append(pre)
        
        # print(pre.shape)
        # print(pre[0])
        
    generate_submission_csv(range(1, len(pre_list)+1), np.array(pre_list), 'submission.csv')
