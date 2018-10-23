#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging; logging.basicConfig(level=logging.INFO)
import struct
import numpy as np
import pandas as pd
import os
import sys
from .util import show_img, show_imgs, load_pickle, one_hot, label2name, get_one_batch

mnist_fashion_name_list = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                           'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


class Loader(object):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    --- *** ---
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    """
    def __init__(self):
        self.raw_data_type = None
        self.raw_dims = None
        self.raw_shape = None

    def load_raw(self, path):
        """ A function that can read MNIST's idx file format into numpy arrays.
        The MNIST data files can be downloaded from here:
        http://yann.lecun.com/exdb/mnist/

        This relies on the fact that the MNIST dataset consistently uses
        unsigned char types with their data segments.
        https://gist.github.com/tylerneylon/ce60e8a06e7506ac45788443f7269e40
        """
        with open(path, 'rb') as f:
            zero, self.raw_data_type, self.raw_dims = struct.unpack('>HBB', f.read(4))
            self.raw_shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(self.raw_dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(self.raw_shape)  # count, H, W


class MNIST(Loader):
    def __init__(self, root):
        super(MNIST, self).__init__()
        self.root = root
        self.train_image_path = os.path.join(self.root, 'train-images-idx3-ubyte')
        self.train_label_path = os.path.join(self.root, 'train-labels-idx1-ubyte')
        self.test_image_path = os.path.join(self.root, 't10k-images-idx3-ubyte')
        self.test_label_path = os.path.join(self.root, 't10k-labels-idx1-ubyte')
        
    def load(self, normalize=True, image_flat=False, label_one_hot=False):
        log_info = 'M@{}, C@{}, F@{}, MNIST load: normalize={}, image_flat={}, label_one_hot={}'.format(__name__, self.__class__.__name__, sys._getframe().f_code.co_name, normalize, image_flat, label_one_hot)
        logging.info(log_info)
        
        train_image = self.load_raw(self.train_image_path).reshape(-1, 1, 28, 28)  # count, channel=1, H, W
        train_label = self.load_raw(self.train_label_path)
        test_image = self.load_raw(self.test_image_path).reshape(-1, 1, 28, 28)  # count, channel=1, H, W
        test_label = self.load_raw(self.test_label_path)
        
        if normalize:
            train_image = train_image / 255.0
            test_image = test_image / 255.0

        if image_flat:
            train_image = train_image.reshape(train_image.shape[0], -1)
            test_image = test_image.reshape(test_image.shape[0], -1)

        if label_one_hot:
            train_label = one_hot(train_label)
            test_label = one_hot(test_label)
            
        return train_image, train_label, test_image, test_label


class CIFAR10(object):
    """
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
    There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with 10000 images.
    The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
    """

    def __init__(self, root):
        """
        initialize CIFAR Loader with file path.
        root: file path
        """
        self.root = root
        self.data_mata = 'batches.meta'
        self.data_batch_file_list = [('data_batch_%d' % idx) for idx in range(1, 6)]
        self.test_batch_file = 'test_batch'
        self.category_list = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',
                                       'dog', 'frog', 'horse', 'ship', 'truck'])

    def _load_data(self, filename):
        """
        load single batch of cifar10
        :param filename: file name to load
        :return:
        """
        with open(filename, 'rb') as f:
            datadict = load_pickle(f)
            logging.info('dict keys: {}'.format(datadict.keys()))

            data = datadict['data']  # --> numpy.ndarray
            data = data.reshape(-1, 3, 32, 32).astype("float32")  # float32(4 bytes)

            labels = np.array(datadict['labels'])  # convert list to numpy.ndarray

            return data, labels

    def load_cifar10(self, normalize=True):
        """
        load all of cifar10 dataset
        :param normalize: [0, 255] to [0, 1]
        :return:
        """
        logging.info('cifar10 load all batches: normalize={}'.format(normalize))

        # load training data
        xs = []
        ys = []
        x = None
        y = None
        file_path_list = [os.path.join(self.root, fn) for fn in self.data_batch_file_list]
        for file_path in file_path_list:
            x, y = self._load_data(file_path)
            xs.append(x)
            ys.append(y)
        train_image = np.concatenate(xs)
        train_label = np.concatenate(ys)
        del x, y

        # load test data
        test_image, test_label = self._load_data(os.path.join(self.root, self.test_batch_file))

        # data processing
        # 由于没有归一化到[0, 1]导致梯度无法正常下降
        if normalize:
            train_image = train_image / 255.0
            test_image = test_image / 255.0

        logging.info('train_image shape: {}'.format(train_image.shape))
        logging.info('train_label shape: {}'.format(train_image.shape))
        logging.info('test_image shape: {}'.format(test_image.shape))
        logging.info('test_label shape: {}'.format(test_label.shape))
        return train_image, train_label, test_image, test_label

    def load_cifar10_batch_one(self, normalize=True):
        """
        load cifar train_batch_1 & test_batch
        :param normalize: [0, 255] to [0, 1]
        :return:
        """
        logging.info('cifar10 load batch_one: normalize={}'.format(normalize))

        # load training data
        f = os.path.join(self.root, 'data_batch_%d' % (1,))
        train_image, train_label = self._load_data(f)

        # load test data
        test_image, test_label = self._load_data(os.path.join(self.root, 'test_batch'))

        # data processing
        if normalize:
            train_image = train_image / 255.0
            test_image = test_image / 255.0
        return train_image, train_label, test_image, test_label


class CIFAR100(object):
    """
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
    There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with 10000 images.
    The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
    """

    def __init__(self, root):
        """
        initialize CIFAR Loader with file path.
        root: file path
        """
        self.root = root
        self.data_mata = 'meta'
        self.train_batch_file = 'train'
        self.test_batch_file = 'test'
        self.fine_label_names = None
        self.coarse_label_names = None
        self.load_meta()

    def _extract_meta(self, filename):
        """
        extract meta data: name list
        :param filename:
        :return:
        """
        with open(filename, 'rb') as f:
            meta_dict = load_pickle(f)
            logging.info('dict keys: {}'.format(meta_dict.keys()))
            self.fine_label_names = np.array(meta_dict['fine_label_names'])  # convert list to numpy.ndarray
            self.coarse_label_names = np.array(meta_dict['coarse_label_names'])  # convert list to numpy.ndarray

    def load_meta(self,):
        self._extract_meta(os.path.join(self.root, self.data_mata))

    def _load_data(self, filename):
        """
        load single batch of cifar10
        :param filename: file name to load
        :return:
        """
        with open(filename, 'rb') as f:
            datadict = load_pickle(f)
            logging.info('dict keys: {}'.format(datadict.keys()))

            data = datadict['data']  # --> numpy.ndarray
            data = data.reshape(-1, 3, 32, 32).astype("float32")  # float32(4 bytes)

            fine_labels = np.array(datadict['fine_labels'])  # convert list to numpy.ndarray
            coarse_labels = np.array(datadict['coarse_labels'])  # convert list to numpy.ndarray

            return data, fine_labels, coarse_labels

    def load_cifar100(self, normalize=True):
        """
        load all of cifar100 dataset
        :param normalize: [0, 255] to [0, 1]
        :return:
        """
        logging.info('cifar100 load all batches: normalize={}'.format(normalize))

        # load training data
        train_image, train_label, _ = self._load_data(os.path.join(self.root, self.train_batch_file))

        # load test data
        test_image, test_label, _ = self._load_data(os.path.join(self.root, self.test_batch_file))

        # data processing
        if normalize:
            train_image = train_image / 255.0
            test_image = test_image / 255.0

        logging.info('train_image shape: {}'.format(train_image.shape))
        logging.info('train_label shape: {}'.format(train_image.shape))
        logging.info('test_image shape: {}'.format(test_image.shape))
        logging.info('test_label shape: {}'.format(test_label.shape))
        return train_image, train_label, test_image, test_label


class TEXT(object):
    def __init__(self, root):
        """
        initialize TEXT Loader with file path.
        :param root: file path
        """
        self.root = root
        self.filename='jaychou_lyrics.txt'
        self.corpus_chars = None
        self.char_to_idx_dict = None
        self.idx_to_char_dict = None

    def load(self, convert=True):
        """
        open a text file and return data as an array
        :param convert: convert \n \r \u3000 to space
        :return: ndarray data
        """
        with open(os.path.join(self.root, self.filename), 'r', encoding='utf-8') as f:
            data = f.read()  # --> str
        if convert:
            data = data.replace('\n', ' ').replace('\r', ' ').replace('\u3000', ' ')
        # print(data.find('\u3000'))
        # print(data[11044-2: 11044+2])
        data = np.array(list(data))
        self.corpus_chars = list(set(data))
        logging.info('data size: {}, vocab size: {}'.format(len(data), len(self.corpus_chars)))

        self.char_to_idx_dict = {ch: i for i, ch in enumerate(self.corpus_chars)}
        self.idx_to_char_dict = {i: ch for i, ch in enumerate(self.corpus_chars)}
        return data

    def char_to_idx(self, batch_x_char):
        batch_x_idx = np.zeros_like(batch_x_char, dtype=np.int)
        it = np.nditer(batch_x_char, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            batch_x_idx[idx] = self.char_to_idx_dict[batch_x_char[idx]]
            it.iternext()
        return batch_x_idx

    def idx_to_char(self, batch_x_idx):
        batch_x_char = np.zeros_like(batch_x_idx)
        it = np.nditer(batch_x_idx, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            batch_x_char[idx] = self.idx_to_char_dict[batch_x_idx[idx]]
            it.iternext()
        return batch_x_char

    def get_one_batch_random(self, data, batch_size, steps_num):
        x = np.array([])
        y = np.array([])
        valid_len = len(data) - steps_num - 1
        start_pts = np.random.choice(valid_len, batch_size)
        for idx, start_pt in enumerate(start_pts):
            tmp_x = data[start_pt: start_pt + steps_num]
            tmp_y = data[start_pt + 1: start_pt + 1 + steps_num]
            if idx == 0:
                x = tmp_x
                y = tmp_y
            else:
                x = np.concatenate((x, tmp_x))
                y = np.concatenate((y, tmp_y))
        return x.reshape(batch_size, steps_num), y.reshape(batch_size, steps_num)

    def data_iter_random(self, data, batch_size, steps_num):
        num_examples = (len(data) - 1) // steps_num
        epoch_size = num_examples // batch_size
        example_indices = list(range(num_examples))
        np.random.shuffle(example_indices)
        # 返回从 pos 开始的长为 num_steps 的序列
        _data = lambda pos: data[pos: pos + steps_num]
        for i in range(epoch_size):
            # 每次读取 batch_size 个随机样本。
            idx = i * batch_size
            batch_indices = example_indices[idx: idx + batch_size]
            x = [_data(j * steps_num) for j in batch_indices]
            y = [_data(j * steps_num + 1) for j in batch_indices]
            yield x, y

    def data_iter_consecutive(self, data, batch_size, steps_num):
        batch_len = len(data) // batch_size
        indices = data[0: batch_size * batch_len].reshape((batch_size, batch_len))
        epoch_size = (batch_len - 1) // steps_num
        for i in range(epoch_size):
            idx = i * steps_num
            x = indices[:, idx: idx + steps_num]
            y = indices[:, idx + 1: idx + steps_num + 1]
            yield x, y

            
class HousePrices(object):
    def __init__(self, root):
        """
        initialize HousePrices Loader with file path.
        :param root: file path
        """
        self.root = root
        self.train_filename='train.csv'
        self.test_filename='test.csv'
        self.test_id = None

    def load(self, scale=True, label_log10=True, non_nan_ratio=0.75):
        """
        load House Prices dataset
        :param scale:
        :param label_log10:
        :param non_nan_ratio:
        :return:
        """
        train_data = pd.read_csv(os.path.join(self.root, self.train_filename))
        # keep columns which Non-NaN count > non_nan_ratio * ALL
        # delete columns which Non-NaN count < non_nan_ratio * ALL
        # delete columns which NaN count > non_nan_ratio * ALL 
        train_data = train_data.dropna(axis=1, how='any', thresh=train_data.shape[0] * non_nan_ratio)
        
        test_data = pd.read_csv(os.path.join(self.root, self.test_filename))
        # keep columns which Non-NaN count > non_nan_ratio * ALL
        # delete columns which Non-NaN count < non_nan_ratio * ALL
        # delete columns which NaN count > non_nan_ratio * ALL 
        test_data = test_data.dropna(axis=1, how='any', thresh=test_data.shape[0] * non_nan_ratio)

        self.test_id = test_data['Id'].values
        
        all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # remove ID & SalePrice(only in train) columns
        numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
        all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
        all_features = all_features.fillna(all_features.mean())
        all_features = pd.get_dummies(all_features, dummy_na=True)  # dummy_na=True take NaN as a legal feature label
        
        if scale:  # scale data to [0, 1] by (features - features.min) / (features.max - features.min)
            slices = (all_features.max() != 0).values
            all_features -= all_features.min()
            all_features.iloc[:, slices] /= all_features.iloc[:, slices].max()  # to avoid the case where /* max == 0 */
        
        n_train = train_data.shape[0]
        train_features = all_features[:n_train].values
        test_features = all_features[n_train:].values
        train_labels = train_data.SalePrice.values.reshape((-1, 1))
        
        if label_log10:
            assert (train_labels>0).all()
            train_labels = np.log10(train_labels)

        return train_features, train_labels, test_features
    
    def get_id(self, type='test'):
        if type == 'test':
            return self.test_id
        else:
            pass

            
if __name__ == '__main__':
    """ test MNIST """
    # mnist = MNIST('datasets/mnist')
    # train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=False, label_one_hot=False)
    # train_x_batch, train_y_batch = get_one_batch(train_x, train_y, batch_size=10)
    # logging.info('batch train shape: {}'.format(train_x_batch.shape))
    # logging.info('batch test shape: {}'.format(train_y_batch.shape))
    # show_imgs(train_x_batch.transpose(0, 2, 3, 1), train_y_batch)

    """ test CIFAR10 """
    # cifar10 = CIFAR10('datasets/cifar10')
    # train_x, train_y, test_x, test_y = cifar10.load_cifar10_batch_one(normalize=True)
    #
    # n = 10
    # images = train_x[0:n].transpose(0, 2, 3, 1)
    # labels = train_y[0:n]
    # sm = show_img()  # coroutine by generator
    # sm.__next__()
    # for i in range(n):
    #     sm.send((images[i], labels[i]))
    # show_imgs(images, cifar10.label2text(labels))

    """ test CIFAR100 """
    # cifar100 = CIFAR100(r'datasets/cifar100')
    # train_x, train_y, test_x, test_y = cifar100.load_cifar100()
    #
    # n = 10
    # images = train_x[0:n].transpose(0, 2, 3, 1)
    # labels = train_y[0:n]
    # sm = show_img()  # coroutine by generator
    # sm.__next__()
    # for i in range(n):
    #     sm.send((images[i], labels[i]))
    # show_imgs(images, label2name(index_array=labels, label_array=cifar100.fine_label_names))

    """ test TEXT """
    # lyrics = TEXT('../datasets/text')
    # lyrics_data = lyrics.load()

    # print(lyrics.char_to_idx(np.array([['想', '要', '有'], ['想', '要', '有']])))

    # print(lyrics_data[0:10])
    # print(lyrics.char_to_idx(lyrics_data[0:10]))
    # print(lyrics_data[1:11])
    # print(lyrics.char_to_idx(lyrics_data[1:11]))
    
    """ test HousePrices """
    hp = HousePrices('../data')
    ret = hp.load()
    print(ret[0].shape, ret[1].shape, ret[2].shape)
    