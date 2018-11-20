#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from common import layers
from common.util import numerical_gradient, get_one_batch
from common.visualize import show_accuracy_loss, show_activation
from common.optimizer import SGD, Adam
from common.datasets import MushroomClass

import functools
import time


class MultiLayerRegression(object):
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu',
                 weight_decay_lambda=0.0,
                 use_dropout=False, dropout_ratio=0.5,
                 use_batchnorm=False):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.params = {}

        # weights initialization
        self.__init_weight(weight_init_std)

        # generate layers
        activation_layer = {'sigmoid': layers.Sigmoid, 'relu': layers.Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)],
                                                             self.params['b' + str(idx)])
            if self.use_batchnorm:
                self.layers['BatchNorm' + str(idx)] = layers.BatchNormalization(self.params['gamma' + str(idx)],
                                                                                self.params['beta' + str(idx)])
            self.layers['Activation' + str(idx)] = activation_layer[activation]()
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = layers.Dropout(dropout_ratio)
        # last Affine layer need no Activation & Batch Norm
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = layers.Affine(self.params['W' + str(idx)],
                                                         self.params['b' + str(idx)])

        self.last_layer = layers.SoftmaxCrossEntropy(class_num=2)
        # self.last_layer = layers.MSE()
        # dict to save activation layer output
        self.activation_dict = OrderedDict()

    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            # print('weight_init_std:', weight_init_std)
            if isinstance(weight_init_std, str):
                if weight_init_std.lower() in ['sigmoid', 'xavier']:
                    scale = np.sqrt(1.0 / all_size_list[idx - 1])
                elif weight_init_std.lower() in ['relu', 'he']:
                    scale = np.sqrt(2.0 / all_size_list[idx - 1])
                else:
                    print('str ')
            else:
                print('not str')
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.random.randn(all_size_list[idx])
        # last Affine layer need no Activation & Batch Norm
        if self.use_batchnorm:
            for idx in range(1, self.hidden_layer_num + 1):
                self.params['gamma' + str(idx)] = np.ones(all_size_list[idx])
                self.params['beta' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x_batch, train_flag=False):
        tmp = x_batch.copy()  # .copy() is not necessary!
        for layer_name, layer in self.layers.items():
            if 'Dropout' in layer_name or 'BatchNorm' in layer_name:
                tmp = layer.forward(tmp, train_flag=train_flag)
            else:
                tmp = layer.forward(tmp)
            if 'Activation' in layer_name:
                self.activation_dict[layer_name] = tmp
        return tmp

    def loss(self, x_batch, t_batch, train_flag=False):
        y_batch = self.predict(x_batch, train_flag=train_flag)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y_batch, t_batch) + weight_decay

    def gradient(self, x_batch, t_batch):
        # forward
        self.loss(x_batch, t_batch, train_flag=True)
        # backward
        dout = 1
        dout = self.last_layer.backward(d_y=dout)
        layers = list(self.layers.values())  # self.layers is a dict not a list!!!
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        for idx in range(1, len(self.hidden_size_list) + 2):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].d_W + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].d_b
        # calculate gradients of gamma & beta
        # last Affine layer need no BN
        if self.use_batchnorm:
            for idx in range(1, self.hidden_layer_num + 1):
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta
        return grads

    def numerical_gradient(self, x, t):
        self.x = x
        self.t = t
        loss_W = lambda W: self.loss(x, t, train_flag=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
        # calculate gradients of gamma & beta
        # last Affine layer need no BN
        if self.use_batchnorm:
            for idx in range(1, self.hidden_layer_num + 1):
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])
        return grads

    def accuracy(self, x_batch, t_batch):
        y = self.predict(x_batch, train_flag=False)
        y = np.argmax(y, axis=1)

        accuracy = np.mean(y == t_batch.flatten())
        return accuracy


if __name__ == '__main__':
    hp_data = MushroomClass('./data/Mushrooms')
    x, t = hp_data.load(non_nan_ratio=0.8)
    print('x.shape:', x.shape)
    print('t.shape:', t.shape)
    
    feature_count = x.shape[-1]
    
    train_num = 8000
    train_x, train_y, test_x, test_y = x[:train_num, :], t[:train_num, :], x[train_num:, :], t[train_num:, :]
    print('train_x.shape:', train_x.shape)
    print('train_y.shape:', train_y.shape)
    print('test_x.shape:', test_x.shape)
    print('test_y.shape:', test_y.shape)

    max_iterations = 2000
    batch_size = 128
    # initialize network optimizer
    weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
    # optimizer = SGD(lr=1e-4)
    optimizer = Adam(lr=1e-4)
    
    network = MultiLayerRegression(input_size=feature_count, hidden_size_list=[100, 10], output_size=2,
                                   weight_init_std='relu', activation='relu',
                                   weight_decay_lambda=1e-6,
                                   use_dropout=True, dropout_ratio=0.05,
                                   use_batchnorm=True)

    print('network layers:', network.layers.keys())
    x_batch, t_batch = get_one_batch(train_x, train_y, batch_size=3)
    g = network.gradient(x_batch, t_batch)
    g_n = network.numerical_gradient(x_batch, t_batch)

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    # Start training
    for i in range(max_iterations):
        x_batch, t_batch = get_one_batch(train_x, train_y, batch_size=batch_size)
        
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        # train_loss = network.loss(x_batch, t_batch, train_flag=False)
        # train_loss_list.append(train_loss)
        # test_loss = network.loss(test_x, test_y, train_flag=False)
        # test_loss_list.append(test_loss)
        if i % 100 == 0:
            print("===========" + "iteration:" + str(i) + "===========")
            
            # calculate accuracy
            train_acc = network.accuracy(x_batch, t_batch)
            train_acc_list.append(train_acc)
            test_acc = network.accuracy(test_x, test_y)
            test_acc_list.append(test_acc)
            print("train accuracy: {:.3f}".format(train_acc), "test accuracy: {:.3f}".format(test_acc))
            # calculate loss
            train_loss = network.loss(x_batch, t_batch, train_flag=False)
            train_loss_list.append(train_loss)
            test_loss = network.loss(test_x, test_y, train_flag=False)
            test_loss_list.append(test_loss)
            print("train loss: {:.6f}".format(train_loss), "test loss: {:.6f}".format(test_loss))
    show_accuracy_loss(train_acc_list, test_acc_list, train_loss_list, test_loss_list, 5)
    show_activation(network.activation_dict, bins_range=30)
