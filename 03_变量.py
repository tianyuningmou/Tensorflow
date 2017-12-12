# -*- coding:utf-8 -*-

"""
@author tianyuningmou
@time 2017/11/1 下午2:50
"""


import tensorflow as tf


# 创建一个变量，初始化为标量0
state = tf.Variable(0, name='counter')

# 创建一个OP，起作用是使state增加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后，必须先经过'初始化'op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
