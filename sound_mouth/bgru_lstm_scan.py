# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: bgru_lstm_scan.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/4/17 下午1:08

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/4/17 下午1:08
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


class LSTM_Cell(object):

    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0, L2=False, h_act=tf.tanh, init_h=None,
                 init_c=None):
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        # 初始化方法
        self.initializer = initializer
        # 遗忘门的初始偏移量
        self.f_bias = f_bias
        # 可以选择LSTM的hidden state的激活函数
        self.h_act = h_act
        # 区分GRU
        self.type = 'lstm'
        # 如果没有提供初始的hidden state、memory cell，全部初始化为0
        if init_h is None and init_c is None:
            self.init_h = tf.matmul(self.incoming[0, :, :], tf.zeros([self.D_input, self.D_cell]))
            self.init_c = self.init_h
            self.previous = tf.stack([self.init_h, self.init_c])

        # 输入门、遗忘门、输出门，每个都是(W_x, W_h, b_f)
        self.igate = self.Gate()
        self.fgate = self.Gate(bias=f_bias)
        self.ogate = self.Gate()
        self.cell = self.Gate()

        # 因为所有的gate都会乘以当前的输入和上一时刻的hidden state，将矩阵concat在一起，计算后再逐一分离，加快运行速度
        self.W_x = tf.concat(values=[self.igate[0], self.fgate[0], self.ogate[0], self.cell[0]], axis=1)
        self.W_h = tf.concat(values=[self.igate[1], self.fgate[1], self.ogate[0], self.cell[1]], axis=1)
        self.b = tf.concat(values=[self.igate[2], self.fgate[2], self.ogate[2], self.cell[2]], axis=0)

        # 对LSTM的权重进行L2 regularization
        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    # 初始化gate的函数
    def Gate(self, bias=0.001):
        Wx = self.initializer([self.D_input, self.D_cell])
        Wh = self.initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx, Wh, b

    # 大矩阵乘法运算完后，方便分离gate
    def slice_W(self, x, n):
        return x[:, n * self.D_cell: (n + 1) * self.D_cell]

    def step(self, previous_h_c_tuple, current_x):
        # 将hidden state、memory cell拆分开
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)
        # 统一在concat成的大矩阵中一次完成所有的gates计算
        gates = tf.matmul(current_x, self.W_x) + tf.matmul(prev_h, self.W_h) + self.b
        # 计算
        i = tf.sigmoid(self.slice_W(gates, 0))
        f = tf.sigmoid(self.slice_W(gates, 1))
        o = tf.sigmoid(self.slice_W(gates, 2))
        c = tf.tanh(self.slice_W(gates, 3))
        # 计算当前时刻的memory cell
        current_c = f * prev_c + i * c
        # 计算当前时刻的hidden state
        current_h = o * self.h_act(current_c)
        # 将hidden state、memory cell合并后返回
        return tf.stack([current_h, current_c])


class GRU_Cell(object):

    def __init__(self, incoming, D_input, D_cell, initializer, L2=False, init_h=None):
        # 属性
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        self.initializer = initializer
        self.type = 'gru'
        # 如果没有提供最初的hidden state，会初始为0
        # 注意GRU中并没有LSTM中的memory cell，其功能是由hidden state完成的
        if init_h is None:
            # If init_h is not provided, initialize it
            # the shape of init_h is [n_samples, D_cell]
            self.init_h = tf.matmul(self.incoming[0, :, :], tf.zeros([self.D_input, self.D_cell]))
            self.previous = self.init_h
        # 如果没有提供最初的hidden state，会初始为0
        # 注意GRU中并没有LSTM中的memory cell，其功能是由hidden state完成的
        self.rgate = self.Gate()
        self.ugate = self.Gate()
        self.cell = self.Gate()
        # 因为所有的gate都会乘以当前的输入和上一时刻的hidden state，将矩阵concat在一起，计算后再逐一分离，加快运行速度
        # W_x的形状是[D_input, 3*D_cell]
        self.W_x = tf.concat(values=[self.rgate[0], self.ugate[0], self.cell[0]], axis=1)
        self.W_h = tf.concat(values=[self.rgate[1], self.ugate[1], self.cell[1]], axis=1)
        self.b = tf.concat(values=[self.rgate[2], self.ugate[2], self.cell[2]], axis=0)
        # 对LSTM的权重进行L2 regularization
        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    # 初始化gate的函数
    def Gate(self, bias=0.001):
        # Since we will use gate multiple times, let's code a class for reusing
        Wx = self.initializer([self.D_input, self.D_cell])
        Wh = self.initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx, Wh, b

    # 大矩阵乘法运算完毕后，方便用于分离各个gate
    def slice_W(self, x, n):
        # split W's after computing
        return x[:, n * self.D_cell:(n + 1) * self.D_cell]

    # 每个time step需要运行的步骤
    def step(self, prev_h, current_x):
        # 分两次，统一在concat成的大矩阵中完成gates所需要的计算
        Wx = tf.matmul(current_x, self.W_x) + self.b
        Wh = tf.matmul(prev_h, self.W_h)
        # 分离和组合reset gate
        r = tf.sigmoid(self.slice_W(Wx, 0) + self.slice_W(Wh, 0))
        # 分离和组合update gate
        u = tf.sigmoid(self.slice_W(Wx, 1) + self.slice_W(Wh, 1))
        # 分离和组合新的更新信息
        # 注意GRU中，在这一步就已经有reset gate的干涉了
        c = tf.tanh(self.slice_W(Wx, 2) + r * self.slice_W(Wh, 2))
        # 计算当前hidden state，GRU将LSTM中的input gate和output gate的合设成1，
        # 用update gate完成两者的工作
        current_h = (1 - u) * prev_h + u * c
        return current_h


def RNN(cell, cell_b=None, merge='sum'):
    """
      该函数接受的数据需要是[n_steps, n_sample, D_output],
      函数的输出也是[n_steps, n_sample, D_output].
      如果输入数据不是[n_steps, n_sample, D_input],
      使用'inputs_T = tf.transpose(inputs, perm=[1,0,2])'.
      """
    # 正向rnn的计算
    hstates = tf.scan(fn=cell.step, elems=cell.incoming, initializer=cell.previous, name='hstates')
    # lstm的step经过scan计算后会返回4维tensor，其中[:,0,:,:]表示hidden state，[:,1,:,:]表示memory cell，这里只需要hidden state
    if cell.type == 'lstm':
        hstates = hstates[:, 0, :, :]
    # 如果提供了第二个cell，将进行反向rnn的计算
    if cell_b is not None:
        # 将输入数据变为反向
        incoming_b = tf.reverse(cell.incoming, axis=[0])
        # scan计算反向rnn
        b_hstates_rev = tf.scan(fn=cell_b.step, elems=incoming_b, initializer=cell_b.previous, name='b_hstates')
        if cell_b.type == 'lstm':
            b_hstates_rev = b_hstates_rev[:, 0, :, :]
        # 用scan计算好的反向rnn需要再反向回来与正向rnn所计算的数据进行合并
        b_hstates = tf.reverse(b_hstates_rev, axis=[0])
        # 合并方式可以选择直接相加，也可以选择concat
        if merge == 'sum':
            hstates = hstates + b_hstates
        else:
            hstates = tf.concat(values=[hstates, b_hstates], axis=2)
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
mfc = np.load('X.npy', encoding='bytes')
art = np.load('Y.npy', encoding='bytes')
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
L2_penalty = 1e-4
# 样本的输入和标签
inputs = tf.placeholder(tf.float32, [None, None, D_input], name='inputs')
labels = tf.placeholder(tf.float32, [None, D_label], name='labels')
# 保持多少节点不被dropout掉
drop_keep_rate = tf.placeholder(dtype=tf.float32, name='drop_keep_rate')
# 用于reshape
n_steps = tf.shape(inputs)[1]
n_samples = tf.shape(inputs)[0]
# reshape for dense layer:  [n_samples, n_steps, D_input] to [n_samples*n_steps, D_input]，用于feedforward layer的使用
re1 = tf.reshape(inputs, [-1, D_input])
# 第一层
Wf0 = weight_init([D_input, num_units])
bf0 = bias_init([num_units])
h1 = tf.nn.relu(tf.matmul(re1, Wf0) + bf0)
# dropout
h1d = tf.nn.dropout(h1, drop_keep_rate)
# 第二层
Wf1 = weight_init([num_units, num_units])
bf1 = bias_init([num_units])
h2 = tf.nn.relu(tf.matmul(h1d, Wf1) + bf1)
# dropout
h2d = tf.nn.dropout(h2, drop_keep_rate)

# reshape for lstm: [n_samples*n_steps, D_input] to [n_samples, n_steps, D_input]，用于双向rnn layer的使用
re2 = tf.reshape(h2d, [n_samples,n_steps, num_units])
inputs_T = tf.transpose(re2, perm=[1,0,2])

# lstm
#rnn_fcell = LSTM_Cell(inputs_T, num_units, num_units, zero_init, f_bias=1.0)
#rnn_bcell = LSTM_Cell(inputs_T, num_units, num_units, zero_init, f_bias=1.0)
#cell = Bi_LSTM_cell(inputs_T, num_units, num_units, 1)
#rnn0 = cell.get_states()

rnn_fcell = GRU_Cell(inputs_T, num_units, num_units, orthogonal_initializer)
rnn_bcell = GRU_Cell(inputs_T, num_units, num_units, orthogonal_initializer)
rnn0 = RNN(rnn_fcell, rnn_bcell)

# reshape for output layer，用于feedforward layer的使用
rnn1 = tf.reshape(rnn0, [-1, num_units])
# dropout
rnn2 = tf.nn.dropout(rnn1, drop_keep_rate)

# 第三层
W0 = weight_init([num_units, num_units])
b0 = bias_init([num_units])
rnn3 = tf.nn.relu(tf.matmul(rnn2, W0) + b0)
rnn4 = tf.nn.dropout(rnn3, drop_keep_rate)
# 第四层
W1 = weight_init([num_units, num_units])
b1 = bias_init([num_units])
rnn5 = tf.nn.relu(tf.matmul(rnn4, W1) + b1)
rnn6 = tf.nn.dropout(rnn5, drop_keep_rate)
# 输出层
W = weight_init([num_units, D_label])
b = bias_init([D_label])
output = tf.matmul(rnn6, W) + b

# loss
loss=tf.reduce_mean((output-labels)**2)
L2_total = tf.nn.l2_loss(Wf0) + tf.nn.l2_loss(Wf1)+ tf.nn.l2_loss(W0) + tf.nn.l2_loss(W1) + tf.nn.l2_loss(W)

# 训练所需
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss + L2_penalty*L2_total)


# 训练网络
# 建立session并实际初始化参数
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# 训练并记录
def train_epoch(epoch):
    for k in range(epoch):
        train_zero = shufflelists(train)
        for i in range(len(train)):
            sess.run(train_step, feed_dict={inputs: train_zero[i][0], labels: train_zero[i][1], drop_keep_rate: 0.7})
        train_l = 0
        test_l = 0
        for i in range(len(test)):
            test_l += sess.run(loss, feed_dict={inputs: test[i][0], labels: test[i][1], drop_keep_rate: 1.0})
        for i in range(len(train)):
            train_l += sess.run(loss, feed_dict={inputs: train[i][0], labels: train[i][1], drop_keep_rate: 1.0})
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

