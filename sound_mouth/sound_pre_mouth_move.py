# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: sound_pre_mouth_move.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/4/11 下午4:03

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/4/11 下午4:03
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class FNN(object):

    def __init__(self, learning_rate, Layers, N_hidden, D_input, D_label, Task_type='regression', L2_lambda=0.0):
        self.learning_rate = learning_rate
        self.Layers = Layers
        self.N_hidden = N_hidden
        self.D_input = D_input
        self.D_label = D_label
        # 类型控制loss函数的选择
        self.Task_type = Task_type
        # l2 regularization的惩罚强弱，过高会使得输出都拉向0
        self.L2_lambda = L2_lambda
        # 用于存放所累积的每层l2 regularization
        self.l2_penalty = tf.constant(0.0)

        # 用于生成tensorflow的缩略图，括号里起名字
        with tf.name_scope('Input'):
            self.inputs = tf.placeholder(tf.float32, [None, D_input], name='inputs')
        with tf.name_scope('Label'):
            self.labels = tf.placeholder(tf.float32, [None, D_label], name='label')
        with tf.name_scope('keep_rate'):
            self.drop_keep_rate = tf.placeholder(tf.float32, name='dropout_keep')

        # 初始化的时候直接生成，build方法是后面建立的
        self.build('F')

    def weight_init(self, shape):
        # shape: list[in_dim, out_dim]
        # 在这里更改初始化方法
        # 方式1：下面的权重初始化若用ReLU激活函数，可以使用带有6个隐藏层的神经网络
        #       若过深，则使用dropout会难以拟合。
        # initial = tf.truncated_normal(shape, stddev=0.1) / np.sqrt(shape[1])
        # 方式2：下面的权重初始化若用ReLU激活函数，可以扩展到15个隐藏层以上（通常不会用那么多）
        initial = tf.random_uniform(shape,
                                    minval=-np.sqrt(5)*np.sqrt(1.0/shape[0]),
                                    maxval=np.sqrt(5)*np.sqrt(1.0/shape[0]))
        return tf.Variable(initial)

    def bias_init(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self, var, name):
        with tf.name_scope(name+'_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/'+name, mean)
        with tf.name_scope(name+'_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('_stddev/'+name, stddev)
        tf.summary.scalar('_max/'+name, tf.reduce_max(var))
        tf.summary.scalar('_min/'+name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

    def layer(self, in_tensor, in_dim, out_dim, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
            with tf.name_scope(layer_name+'_weights'):
                # 用所建立的weight_inith函数进行初始化
                weights = self.weight_init([in_dim, out_dim])
                # 存放着每一个权重
                self.W.append(weights)
                # 对权重进行统计
                self.variable_summaries(weights, layer_name+'/weights')
            with tf.name_scope(layer_name+'_biases'):
                biases = self.bias_init([out_dim])
                self.b.append(biases)
                self.variable_summaries(biases, layer_name+'/biases')
            with tf.name_scope(layer_name+'_Wx_plus_b'):
                # 计算Wx+b
                pre_activate = tf.matmul(in_tensor, weights) + biases
                # 记录直方图
                tf.summary.histogram(layer_name+'/pre_activations', pre_activate)
            activations = act(pre_activate, name='activation')
            tf.summary.histogram(layer_name+'/activations', activations)
        # 最终返回该层的输出，以及权重W的L2
        return activations, tf.nn.l2_loss(weights)

    def drop_layer(self, in_tensor):
        dropped = tf.nn.dropout(in_tensor, self.drop_keep_rate)
        return dropped

    def build(self, prefix):
        # 建立网络。incoming也代表当前tensor的流动位置
        incoming = self.inputs
        # 是否有隐藏层
        if self.Layers != 0:
            layer_nodes = [self.D_input] + self.N_hidden
        else:
            layer_nodes = [self.D_input]
        # hidden_layers用于存储所有隐藏层的输出
        self.hidden_layers = []
        # W用于存储所有层的权重
        self.W = []
        # b用于存储所有层的偏移
        self.b = []
        # total_l2用于存储所有层的L2
        self.total_l2 = []

        # 开始叠加隐藏层，这个跟千层饼没什么区别
        for i in range(self.Layers):
            # 使用刚才编写的函数来建立层，并更新incoming的位置
            incoming, l2_loss = self.layer(incoming, layer_nodes[i], layer_nodes[i + 1],
                                           prefix + '_hid_' + str(i+1), act=tf.nn.relu)
            # 累计l2
            self.total_l2.append(l2_loss)
            # 输出一些信息，让我们知道网络在建造中做了什么
            print('Add dense layer: relu')
            print('    %sD --> %sD' % (layer_nodes[i], layer_nodes[i+1]))
            # 存储所有隐藏层的输出
            self.hidden_layers.append(incoming)
            # 加入dropout层
            incoming = self.drop_layer(incoming)

        # 输出层的建立。输出层需要特别对待的原因是输出层的activation function要根据任务来变
        # 回归任务的话，下面用的是tf.identify，也就是没有activation function
        if self.Task_type == 'regression':
            out_act = tf.identity
        else:
            # 分类任务使用softmax来拟合概率
            out_act = tf.nn.softmax

        self.output, l2_loss = self.layer(incoming, layer_nodes[-1], self.D_label, layer_name='output', act=out_act)
        self.total_l2.append(l2_loss)
        print('Add output layer: linear')
        print('    %sD --> %sD' % (layer_nodes[-1], self.D_label))

        # l2_loss的缩放图
        with tf.name_scope('total_l2'):
            for l2 in self.total_l2:
                self.l2_penalty += l2
            tf.summary.scalar('l2_penalty', self.l2_penalty)

        # 不同任务的loss
        # 若为回归，loss则用于判断所有预测值和实际值差别的函数
        if self.Task_type == 'regression':
            with tf.name_scope('SSE'):
                self.loss = tf.reduce_mean((self.output - self.labels) ** 2)
                self.loss2 = tf.nn.l2_loss(self.output - self.labels)
                tf.summary.scalar('loss', self.loss)
        else:
            # 若为分类，cross, entropy的loss function
            entropy = tf.nn.softmax_cross_entropy_with_logits(self.output, self.labels)
            with tf.name_scope('cross entropy'):
                self.loss = tf.reduce_mean(entropy)
                tf.summary.scalar('loss', self.loss)
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', self.accuracy)

        # 整合所有loss，形成最终loss
        with tf.name_scope('total_loss'):
            self.total_loss = self.loss + self.l2_penalty * self.L2_lambda
            tf.summary.scalar('total_loss', self.total_loss)

        # 训练操作
        with tf.name_scope('train'):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

    # 洗牌功能
    def shuffle_lists(self, lists):
        ri = np.random.permutation(len(lists[1]))
        out = []
        for l in lists:
            out.append(l[ri])
        return out


def standardize(seq):
    centerized = seq - np.mean(seq, axis=0)
    normalized = centerized / np.std(centerized, axis=0)
    return normalized


def make_windows(in_data, window_size=41):
    out_data = []
    mid = int(window_size / 2)
    in_data = np.vstack((np.zeros((mid, in_data.shape[1])), in_data, np.zeros((mid, in_data.shape[1]))))
    for i in range(in_data.shape[0] - window_size + 1):
        out_data.append(np.hstack(in_data[i: i + window_size]))
    return np.array(out_data)


mfc = np.load('X.npy', encoding='bytes')
art = np.load('Y.npy', encoding='bytes')
x = []
y = []
for i in range(len(mfc)):
    x.append(make_windows(standardize(mfc[i])))
    y.append(standardize(art[i]))
vali_size = 20
total_samples = len(np.vstack(x))
X_train = np.vstack(x)[int(total_samples / vali_size):].astype('float32')
Y_train = np.vstack(y)[int(total_samples / vali_size):].astype('float32')

X_test = np.vstack(x)[: int(total_samples / vali_size)].astype('float32')
Y_test = np.vstack(y)[: int(total_samples / vali_size)].astype('float32')

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

ff = FNN(learning_rate=7e-5, Layers=5, N_hidden=[2048, 1024, 512, 256, 128], D_input=1599, D_label=24, L2_lambda=1e-4)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log3' + '/train', sess.graph)
test_writer = tf.summary.FileWriter('log3' + '/test')


def plots(T, P, i, length=400):
    m = 0
    plt.figure(figsize=(20, 16))
    plt.subplot(411)
    plt.plot(T[m: m + length, 7], '--')
    plt.plot(P[m: m + length, 7])

    plt.subplot(412)
    plt.plot(T[m: m + length, 8], '--')
    plt.plot(P[m: m + length, 8])

    plt.subplot(413)
    plt.plot(T[m: m + length, 15], '--')
    plt.plot(P[m: m + length, 15])

    plt.subplot(414)
    plt.plot(T[m: m + length, 16], '--')
    plt.plot(P[m: m + length, 16])

    plt.legend(['True', 'Predicted'])
    plt.savefig('epoch' + str(i) + '.png')
    plt.close()


k = 0
Batch = 32
for i in range(50):
    index = 0
    X0, Y0 = ff.shuffle_lists([X_train, Y_train])
    while index < X_train.shape[0]:
        summary, _ = sess.run([merged, ff.train_step], feed_dict={ff.inputs: X0[index: index + Batch],
                                                                  ff.labels: Y0[index: index + Batch],
                                                                  ff.drop_keep_rate: 1.0})
        index += Batch
        k += 1
        train_writer.add_summary(summary, k)

    # test
    summary, pY, pL = sess.run([merged, ff.output, ff.loss], feed_dict={ff.inputs: X_test,
                                                                        ff.labels: Y_test,
                                                                        ff.drop_keep_rate: 1.0})
    plots(Y_test, pY, i)
    test_writer.add_summary(summary, k)
    print('epoch{epoch} | train_loss:{train_loss} | test_loss:{test_loss}'.format(
        epoch=i,
        train_loss=sess.run(ff.loss, feed_dict={ff.inputs: X0,
                                                ff.labels: Y0,
                                                ff.drop_keep_rate: 1.0}),
        test_loss=sess.run(ff.loss, feed_dict={ff.inputs: X_test,
                                               ff.labels: Y_test,
                                               ff.drop_keep_rate: 1.0})
    ))
