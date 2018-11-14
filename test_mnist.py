from common.datasets import MNIST
from common.datasets import MNISTCSV
import os
import psutil
import numpy as np
import pandas as pd

if __name__ == '__main__':
    mnist = MNIST('data\\MNIST')
    # mnist = MNISTCSV('data\\MNIST')
    train_x, train_y, test_x, test_y = mnist.load(normalize=True, image_flat=False, label_one_hot=False)
    print(train_x.dtype, train_x.shape)
    print(train_y.dtype, train_y.shape)
    print(train_x[0][0][10])
    print(train_y[0])
    
    """
    # a = pd.read_csv('data\\MNIST\\train.csv')
    a = pd.read_csv('data\\MNIST\\train.csv').astype('uint8')
    a.info()
    """
    
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)

