# coding: utf-8
# Reference: https://github.com/oreilly-japan/deep-learning-from-scratch
# import logging; logging.basicConfig(level=logging.INFO)
# import logging; logging.basicConfig(level=logging.DEBUG)
import numpy as np
import platform
from six.moves import cPickle as pickle


def accuracy(y_hat: np.array, y: np.array):
    tmp = y_hat.argmax(axis=1) == y  # type: np.ndarray
    return np.mean(tmp)


def check_is_ndarray(input_array):
    try:
        return isinstance(input_array, np.ndarray)  # or use: hasattr(input_array, shape)
    except TypeError:
        print("Input data type should be ndarray.")
        
        
def one_hot(input_array, class_num=10):
    """
    (matrix, )-->(matrix, length)
    """
    if input_array.shape[-1] == 1:
        dim = input_array.ndim
        input_array = np.squeeze(input_array, axis=dim - 1)
    array_size = input_array.size
    array_shape = input_array.shape

    # vec = input_array.reshape(array_size, )  # flatten input_array
    vec = input_array.flatten()  # flatten input_array
    ret = np.zeros((array_size, class_num))  # temp zero matrix
    ret[range(array_size), vec] = 1  # modify last dimension by vec values

    return ret.reshape(*array_shape, class_num)


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        # print('fxh1:', fxh1)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        # print('fxh2:', fxh2)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す
        it.iternext()

    return grad


def get_one_batch(image_set, label_set, batch_size=100):
    """
    get batch data
    """
    set_size = len(image_set)
    index = np.random.choice(set_size, batch_size)
    return image_set[index], label_set[index]


def label2name(index_array, label_array):
    try:
        return label_array[index_array]
    except TypeError:
        print("Please check index type.")


def load_pickle(f, encoding='latin1'):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding=encoding)
    raise ValueError("invalid python version: {}".format(version))

    
def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


if __name__ == '__main__':
    step_num = 2
    batch_size = 3
    class_num = 5
    input_array = np.random.choice(class_num, size=(step_num, batch_size))
    output_array = one_hot(input_array, class_num=class_num)
    print('input_array:\n', input_array)
    print('output_array:\n', output_array)