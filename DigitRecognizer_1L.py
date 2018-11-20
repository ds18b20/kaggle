#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
import common.layers as layers
from collections import OrderedDict
from common.datasets import MNISTCSV, MNIST
from common.util import get_one_batch
from common.visualize import show_imgs, show_accuracy_loss, show_filter
import common.optimizer as optimizer
import pandas as pd


class SimpleConvNet(object):
    def __init__(self,
                 input_dim=(1, 28, 28),
                 conv_param=None,
                 pool_param=None,
                 hidden_size=100,
                 output_size=10,
                 weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        pool_size = pool_param['pool_size']
        pool_stride = pool_param['pool_stride']
        channel, input_size, _ = input_dim
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/pool_size) * (conv_output_size/pool_size))

        # init para
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, channel, filter_size, filter_size)  # 30,1,5,5
        self.params['b1'] = np.zeros(filter_num)  # 30,
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)  # (30*12*12),100 => 4320,100
        self.params['b2'] = np.zeros(hidden_size)  # 100,
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)  # 100,10
        self.params['b3'] = np.zeros(output_size)  # 10,

        # create layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = layers.Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = layers.Relu()
        self.layers['Pool1'] = layers.Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = layers.Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = layers.Relu()
        self.layers['Affine2'] = layers.Affine(self.params['W3'], self.params['b3'])
        self.lossLayer = layers.SoftmaxCrossEntropy(class_num=10)

        # self.loss_list = []

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
        # self.loss_list.append(loss)  # remove saving loss to run separated routine
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
        grads["W2"], grads["b2"] = self.layers["Affine1"].d_W, self.layers["Affine1"].d_b
        grads["W3"], grads["b3"] = self.layers["Affine2"].d_W, self.layers["Affine2"].d_b

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
    # mnist = MNIST('data\\MNIST')
    train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=False, label_one_hot=False)
    print(train_x.shape)
    print(test_x.shape)
    # # show sample images
    # train_x_sample, train_y_sample = get_one_batch(train_x, train_y, batch_size=5)
    # show_imgs(train_x_batch.reshape(-1, 28, 28), train_y_batch)

    learning_rate = 0.01
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []
    network = SimpleConvNet(input_dim=(1, 28, 28),
                            conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            pool_param={'pool_size': 2, 'pool_stride': 2},
                            hidden_size=100, output_size=10, weight_init_std=0.01)
    # for layer in network.layers.values():
    #     print(layer)
    # print(network.lossLayer)
    # print('****** Print structure without values: OK ******')

    # show network structure
    train_x_batch, train_y_batch = get_one_batch(train_x, train_y, batch_size=10)
    show_structure(network, train_x_batch, train_y_batch)

    op = optimizer.Adam(lr=0.001)
    epoch = 100
    for i in range(1000):
        train_x_batch, train_y_batch = get_one_batch(train_x, train_y, batch_size=10)
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
        
    generate_submission_csv(range(1, len(pre_list) + 1), np.array(pre_list), 'submission.csv')
