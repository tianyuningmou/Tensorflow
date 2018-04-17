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

def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape=shape))
    return tf.Variable(initial, trainable=True)

def orthogonal_initializer(shape, scale=1.0):
    scale = 1.0

