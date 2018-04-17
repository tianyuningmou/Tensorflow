# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: lstm_scan.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/4/16 下午5:15

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/4/16 下午5:15
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


class LSTM_Cell(object):

    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0):
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        # 输入门的三个参数
        self.W_xi = initializer([self.D_input, self.D_cell])
        self.W_hi = initializer([self.D_cell, self.D_cell])
        self.b_i = tf.Variable(tf.zeros([self.D_cell]))
        # 遗忘门的三个参数
        self.W_xf = initializer([self.D_input, self.D_cell])
        self.W_hf = initializer([self.D_cell, self.D_cell])
        self.b_f = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))
        # 输出门的三个参数
        self.W_xo = initializer([self.D_input, self.D_cell])
        self.W_ho = initializer([self.D_cell, self.D_cell])
        self.b_o = tf.Variable(tf.zeros([self.D_cell]))
        # 计算新信息的三个参数
        self.W_xc = initializer([self.D_input, self.D_cell])
        self.W_hc = initializer([self.D_cell, self.D_cell])
        self.b_c = tf.Variable(tf.zeros([self.D_cell]))
        # 最初的hidden state、memory cell值，二者的形状都是[n_samples, D_cell]，如果没有特殊说明，初始值置为0
        init_for_both = tf.matmul(self.incoming[:, 0, :], tf.zeros([self.D_input, self.D_cell]))
        self.hid_init = init_for_both
        self.cell_init = init_for_both
        self.previous_h_c_tuple = tf.stack([self.hid_init, self.cell_init])
        # 需要将数据由[n_samples, n_steps, D_cell]变为形状[n_steps, n_samples, D_cell]
        self.incoming = tf.transpose(self.incoming, perm=[1, 0, 2])

    def one_step(self, previous_h_c_tuple, current_x):
        # 将hidden state、memory cell拆分开
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)
        # 此时，current_x是当前的输入，prev_h是上一个时刻的hidden state，prev_c是上一个时刻的memory cell
        # 计算输入门
        i = tf.sigmoid(tf.matmul(current_x, self.W_xi) + tf.matmul(prev_h, self.W_hi) + self.b_i)
        # 计算遗忘门
        f = tf.sigmoid(tf.matmul(current_x, self.W_xf) + tf.matmul(prev_h, self.W_hf) + self.b_f)
        # 计算输出门
        o = tf.sigmoid(tf.matmul(current_x, self.W_xo) + tf.matmul(prev_h, self.W_ho) + self.b_o)
        # 计算新的数据信息
        c = tf.tanh(tf.matmul(current_x, self.W_xc) + tf.matmul(prev_h, self.W_hc) + self.b_c)
        # 计算当前时刻的memory cell
        current_c = f * prev_c + i * c
        # 计算当前时刻的hidden state
        current_h = o * tf.tanh(current_c)
        # 将hidden state、memory cell合并后返回
        return tf.stack([current_h, current_c])

    def all_steps(self):
        # 输入形状：[n_samples, n_steps, D_input]
        # 输出形状：[n_steps, n_samples, D_output]
        hstates = tf.scan(fn=self.one_step, elems=self.incoming, initializer=self.previous_h_c_tuple,
                          name='hstates')[:, 0, :, :]
        return hstates


# 权重初始化
# 合理初始化权重，可以降低网络在学习时卡在鞍点或极小值的损害，增加学习速度和效果
def weight_init(shape):
    initial = tf.random_uniform(shape=shape,
                                minval=-np.sqrt(5)*np.sqrt(1.0/shape[0]),
                                maxval=np.sqrt(5)*np.sqrt(1.0/shape[0]))
    return tf.Variable(initial, trainable=True)


# 初始化为0
def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape=shape))
    return tf.Variable(initial, trainable=True)


# 正交矩阵初始化
def orthogonal_initializer(shape, scale=1.0):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.Variable(scale * q[: shape[0], : shape[1]], trainable=True, dtype=tf.float32)


def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


# 洗牌
def shufflelists(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


def standardize(seq):
    centerized = seq - np.mean(seq, axis=0)
    normalized = centerized / np.std(centerized, axis=0)
    return normalized


# 读取输入和输出数据
mfc = np.load('sound_mouth/X.npy', encoding='bytes')
art = np.load('sound_mouth/Y.npy', encoding='bytes')
totalsamples = len(mfc)
# 20%的数据作为validation set
vali_size = 0.2


# 将每个样本的输入和输出数据合成list, 再将所有的样本合成list
# 输入数据的形状是[n_samples, n_steps, D_input]
# 输出数据的形状是[n_samples, D_output]
def data_prer(X, Y):
    D_input = X[0].shape[1]
    data = []
    for x, y in zip(X, Y):
        data.append([standardize(x).reshape((1, -1, D_input)).astype('float32'), standardize(y).astype('float32')])
    return data


# 处理数据
data = data_prer(mfc, art)
# 分训练集和验证集
train = data[int(totalsamples * vali_size):]
test = data[: int(totalsamples * vali_size)]
# print('num of train sequences:%s' %len(train))
# print('num of test sequences:%s' %len(test))
# print('shape of inputs:' ,test[0][0].shape)
# print('shape of labels:' ,test[0][1].shape)


# 构建网络
D_input = 39
D_label = 24
learning_rate = 7e-5
num_units = 1024
# 样本的输入和标签
inputs = tf.placeholder(tf.float32, [None, None, D_input], name='inputs')
labels = tf.placeholder(tf.float32, [None, D_label], name='labels')
# 实例LSTM类
rnn_cell = LSTM_Cell(inputs, D_input, num_units, orthogonal_initializer)
# 调用scan计算所有的hidden states
rnn_zero = rnn_cell.all_steps()
# 将三维tensor[n_steps, n_samples, D_cell]转成矩阵[n_steps * n_samples, D_cell]，用于计算outputs
rnn = tf.reshape(rnn_zero, [-1, num_units])
# 输出层的学习参数
W = weight_init([num_units, D_label])
b = bias_init([D_label])
output = tf.matmul(rnn, W) + b
# 损失
loss = tf.reduce_mean((output - labels) ** 2)
# 训练学习速率
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# 训练网络
# 建立session并实际初始化参数
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# 训练并记录
def train_epoch(epoch):
    for k in range(epoch):
        train_zero = shufflelists(train)
        for i in range(len(train)):
            sess.run(train_step, feed_dict={inputs: train_zero[i][0], labels: train_zero[i][1]})
        train_l = 0
        test_l = 0
        for i in range(len(test)):
            test_l += sess.run(loss, feed_dict={inputs: test[i][0], labels: test[i][1]})
        for i in range(len(train)):
            train_l += sess.run(loss, feed_dict={inputs: train[i][0], labels: train[i][1]})
        print(k, 'train:', round(train_l / 83, 3), 'test:', round(test_l / 20, 3))


start = time.time()
train_epoch(10)
end = time.time()
print(' %f seconds' % round((end - start), 2))

pY = sess.run(output, feed_dict={inputs: test[10][0]})
plt.plot(pY[:, 8])
plt.plot(test[10][1][:, 8])
plt.title('test')
plt.legend(['predicted', 'real'])

pY = sess.run(output, feed_dict={inputs: train[1][0]})
plt.plot(pY[:, 6])
plt.plot(train[1][1][:, 6])
plt.title('train')
plt.legend(['predicted', 'real'])
