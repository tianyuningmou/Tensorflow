# -*- coding:utf-8 -*-

"""
@author tianyuningmou
@time 2017/11/1 下午4:10
"""


import tensorflow as tf


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[3.]}))
