import struct
import numpy as np
import tensorflow as tf

# struct.calcsize(fmt)  # 获取格式对应的大小
# struct.unpack_from(fmt, buf, index)  # 解包数据

def get_train_data():
    Num_Train_IMG = 60000

    with open('datasets/MNIST/train-images.idx3-ubyte', 'rb') as f:
        train_images = f.read()
    with open('datasets/MNIST/train-labels.idx1-ubyte', 'rb') as f:
        train_labels = f.read()

    imgs = []
    index = struct.calcsize('>IIII')    # 跳过四个整形数据
    for i in range(Num_Train_IMG):
        tmp = struct.unpack_from('>784B', train_images, index)
        imgs.append(np.reshape(tmp, (28, 28, 1)))
        index += struct.calcsize('>784B')

    labels = []
    index = struct.calcsize('>II')
    for i in range(Num_Train_IMG):
        tmp = struct.unpack_from('>1B', train_labels, index)
        labels.append(tmp[0])   # 插入元素而不是一个元组
        index += struct.calcsize('>1B')

    return np.array(imgs), np.array(labels)

def get_test_data():
    Num_Test_IMG = 10000

    with open('datasets/MNIST/t10k-images.idx3-ubyte', 'rb') as f:
        test_images = f.read()
    with open('datasets/MNIST/t10k-labels.idx1-ubyte', 'rb') as f:
        test_labels = f.read()

    imgs = []
    index = struct.calcsize('>IIII')
    for i in range(Num_Test_IMG):
        tmp = struct.unpack_from('784B', test_images, index)
        imgs.append(np.reshape(tmp, (28, 28, 1)))
        index += struct.calcsize('>784B')

    labels = []
    index = struct.calcsize('>II')
    for i in range(Num_Test_IMG):
        tmp = struct.unpack_from('>1B', test_labels, index)
        labels.append(tmp[0])  # 插入元素而不是一个元组
        index += struct.calcsize('>1B')

    return np.array(imgs), np.array(labels)
