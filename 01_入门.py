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
