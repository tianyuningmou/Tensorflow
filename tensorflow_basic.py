# -*- coding:utf-8 -*-

"""
@author tianyuningmou
@time 2017/11/1 上午11:54
"""


import tensorflow as tf

# 创建一个常量op，产生一个1*2的矩阵
matrix1 = tf.constant([[3., 3.]])
# 创建另一个常量op，产生一个2*1的矩阵
matrix2 = tf.constant([[2.],[2.]])
# 创建一个矩阵乘法matmul
product = tf.matmul(matrix1, matrix2)
# 启动默认图
with tf.Session() as sess:
    result = sess.run(product)
print(result)


sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()
sub = tf.subtract(x, a)
print(sub.eval())


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


input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[3.]}))
