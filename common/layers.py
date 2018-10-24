#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import numpy as np
from common import functions
from common.util import one_hot, im2col, col2im
import sys


class Affine(object):
    def __init__(self, weights, bias):
        self.W = weights
        self.b = bias
        self.d_W = None
        self.d_b = None

        self.x = None
        self.y = None
        self.original_x_shape = None

        self.d_x = None
        logging.info('M@{}, C@{}, F@{}, W shape: {}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, self.W.shape))
        logging.info('M@{}, C@{}, F@{}, b shape: {}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, self.b.shape))

    def __str__(self):
        if hasattr(self.x, 'shape'):
            batch_size = self.x.shape[0]
        else:
            batch_size = '?'
        (x_feature_count, y_feature_count) = self.W.shape
        x_shape = (batch_size, x_feature_count)
        y_shape = (batch_size, y_feature_count)
        ret_str = "Affine layer: {} dot {} + {} => {}".format(x_shape, self.W.shape, self.b.shape, y_shape)
        return ret_str
        
    @property
    def grad(self):
        return self.d_W, self.d_b

    @grad.setter
    def grad(self, value):
        pass
        # if not isinstance(value, np.ndarray):
            # raise ValueError('grad must be an numpy.ndarray!')
        # if value < 0 or value > 100:
            # raise ValueError('grad must between 0 ~ 100!')
        # self._score = value

    def forward(self, x_batch):
        # テンソル対応
        self.original_x_shape = x_batch.shape
        x = x_batch.reshape(x_batch.shape[0], -1)
        # self.x = x_batch.copy()
        self.x = x
        self.y = np.dot(self.x, self.W) + self.b

        return self.y

    def backward(self, d_y):
        self.d_x = np.dot(d_y, self.W.T)
        self.d_W = np.dot(self.x.T, d_y)
        self.d_b = np.sum(d_y, axis=0)
        self.d_x = self.d_x.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）

        return self.d_x


class Sigmoid(object):
    def __init__(self):
        self.x = None
        self.y = None

        self.d_x = None

    def __str__(self):
        if hasattr(self.x, 'shape'):
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if hasattr(self.y, 'shape'):
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')

        return "Sigmoid layer: {} => {}".format(x_shape, y_shape)

    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-self.x))
        return self.y

    def backward(self, d_y):
        d_x = d_y * (1.0 - self.y) * self.y
        return d_x


class Tanh(object):
    def __init__(self):
        self.x = None
        self.y = None

        self.d_x = None

    def __str__(self):
        if hasattr(self.x, 'shape'):
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if hasattr(self.y, 'shape'):
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')

        return "Tanh layer: {} => {}".format(x_shape, y_shape)

    def forward(self, x_batch):
        self.x = x_batch
        self.y = np.tanh(self.x)
        return self.y

    def backward(self, d_y):
        self.d_x = d_y * (1. - self.y ** 2)
        return self.d_x


class Relu(object):
    def __init__(self):
        self.x = None
        self.y = None

        self.d_x = None

    def __str__(self):
        if hasattr(self.x, 'shape'):
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if hasattr(self.y, 'shape'):
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')

        return "Relu layer: {} => {}".format(x_shape, y_shape)

    def forward(self, x_batch):
        self.x = x_batch
        self.y = np.maximum(self.x, 0)
        return self.y

    def backward(self, d_y):
        idx = (self.x <= 0)
        # tmp = d_y.copy()  # keep d_y not modified, even modification is OK
        # tmp[idx] = 0
        # self.d_x = tmp
        d_y[idx] = 0
        self.d_x = d_y
        return self.d_x


# Mean Squared Error
class MSE(object):
    def __init__(self):
        self.x = None
        self.t = None

        self.y = None
        self.loss = None

        self.d_x = None

    def __str__(self):
        if hasattr(self.x, 'shape'):
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if hasattr(self.y, 'shape'):
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')
        if hasattr(self.t, 'shape'):
            t_shape = self.t.shape
        else:
            t_shape = ('?', '?')
        if hasattr(self.loss, 'shape'):
            loss_shape = self.loss.shape
        else:
            loss_shape = ('?', '?')
        return "Mean Squared Error layer: x:{} => y:{} & t:{} => loss:{}".format(x_shape, y_shape, t_shape, loss_shape)

    def forward(self, x_batch, t_batch):
        self.x = x_batch
        self.t = t_batch
        self.y = self.x
        self.loss = functions.mean_squared_error(self.y, self.t)

        return self.loss

    def backward(self, d_y=1):
        assert self.t.shape[-1] == 1
        batch_size = self.y.shape[0]
        self.d_x = (self.y - self.t) / batch_size
        return d_y * self.d_x


class SoftmaxCrossEntropy(object):
    def __init__(self):
        self.x = None
        self.t = None

        self.y = None
        self.loss = None

        self.d_x = None

    def __str__(self):
        if hasattr(self.x, 'shape'):
            x_shape = self.x.shape
        else:
            x_shape = ('?', '?')
        if hasattr(self.x, 'shape'):
            y_shape = self.y.shape
        else:
            y_shape = ('?', '?')
        if hasattr(self.x, 'shape'):
            loss_shape = self.loss.shape
        else:
            loss_shape = ('?', '?')
        return "Softmax Cross Entropy layer: {} => {} => {}".format(x_shape, y_shape, loss_shape)

    def forward(self, x_batch, t_batch):
        # self.x = x_batch.copy()
        self.x = x_batch
        self.t = t_batch
        self.y = functions.softmax(self.x)
        self.loss = functions.cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, d_y=1):
        assert self.t.ndim == 1
        batch_size = self.y.shape[0]
        # 此处错误导致梯度无法正常下降
        self.d_x = (self.y - one_hot(self.t)) / batch_size  # fix here: (y - t) / batch
        return d_y * self.d_x


class Dropout(object):
    """
    http://arxiv.org/abs/1207.0580
    http://zh.gluon.ai/chapter_deep-learning-basics/dropout.html
    """
    def __init__(self, drop_ratio=0.5):
        self.drop_ratio = drop_ratio
        self.keep_ratio = 1.0 - drop_ratio
        self.mask = None
        assert 0.0 <= self.drop_ratio < 1.0

    def forward(self, x, train_flag=False):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.drop_ratio
            return x * self.mask / self.keep_ratio
        else:
            return x

    def backward(self, d_y):
        return d_y * self.mask


class BatchNormalization(object):
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flag=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flag)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flag):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flag:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Convolution(object):
    def __init__(self, weights, bias, stride=1, pad=0):
        self.W = weights
        self.b = bias
        self.pad = pad
        self.stride = stride
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.d_W = None
        self.d_b = None

        logging.info('M@{}, C@{}, F@{}, W shape: {}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, self.W.shape))
        logging.info('M@{}, C@{}, F@{}, b shape: {}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, self.b.shape))
    
    @property
    def grad(self):
        return self.d_W, self.d_b

    @grad.setter
    def grad(self, value):
        pass

    def __str__(self):
        # self.x changeds at each time
        if hasattr(self.x, 'shape'):
            N, C, H, W = self.x.shape
        else:
            N, H, W= '?', '?', '?'
        FN, C, FH, FW = self.W.shape
        x_shape = (N, C, H, W)
        y_shape = (N, FN, 'new_H', 'new_W')
        ret_str = "Convolution layer: {} conv {} + {} => {}".format(x_shape, self.W.shape, self.b.shape, y_shape)
        return ret_str
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, d_y):
        FN, C, FH, FW = self.W.shape
        d_y = d_y.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.d_b = np.sum(d_y, axis=0)
        self.d_W = np.dot(self.col.T, d_y)
        self.d_W = self.d_W.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(d_y, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling(object):
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def __str__(self):
        # self.x changeds at each time
        if hasattr(self.x, 'shape'):
            x_shape = self.x.shape
            N, C, H, W = self.x.shape
            H = int((H + 2 * self.pad - self.pool_h) / self.stride) + 1
            W = int((W + 2 * self.pad - self.pool_w) / self.stride) + 1
        else:
            x_shape = '?', '?', '?', '?'
            N, C, H, W = '?', '?', '?', '?'
        
        ret_str = "Pooling layer: {} => {}".format(x_shape, (N, C, H, W))
        return ret_str
        
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, d_y):
        d_y = d_y.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((d_y.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = d_y.flatten()
        dmax = dmax.reshape(d_y.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


if __name__ == '__main__':
    # data = (np.arange(9)+1).reshape(3, 3)
    # drop = Dropout()
    # print(drop.forward(data))
    # print(drop.backward(1.0))
    mse = MSE()
    print(mse)
    x_batch = np.array([[1], [1], [0]])
    t_batch = np.array([[1], [1], [1]])
    print(mse.forward(x_batch, t_batch))
    print(mse)
    print(mse.backward(d_y=1))