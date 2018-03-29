# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: cross_entropy.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/29 上午11:29

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/29 上午11:29
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 交叉熵
def cross_entropy():
    # 生成x_data数据
    x_data = np.linspace(0, 0.5, 200)[:, None]
    noise_data = np.random.uniform(-0.02, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise_data

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    weight_layer_one = tf.Variable(tf.random_normal([1, 100]))
    output_layer_one = tf.nn.sigmoid(tf.matmul(x, weight_layer_one))

    weight_layer_two = tf.Variable(tf.random_normal([100, 1]))
    logits = tf.matmul(output_layer_one, weight_layer_two)
    predicts = tf.nn.sigmoid(logits)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    train = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for _ in range(1, 10000):
            session.run(train, feed_dict={x: x_data, y: y_data})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.scatter(x_data, session.run(predicts, feed_dict={x: x_data, y: y_data}), c='r')
        plt.show()


if __name__ == '__main__':
    cross_entropy()
