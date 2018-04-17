# -*- coding: utf-8 -*-

"""
Copyright () 2017

All rights reserved by tianyuningmou

FILE: gdo_exercise.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2017/12/4 上午10:10

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2017/12/4 上午10:10
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 第一部分
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 添加层并返回层的输出结果
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 训练的数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 定义节点接收数据
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义神经层、隐藏层和预测层
hide_layer = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
predict_layer = add_layer(hide_layer, 10, 1)

# 定义loss表达式
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predict_layer), reduction_indices=[1]))

# 选择optimizer使loss最小
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
# 开始运算
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
#使plt不会在show之后停止而是继续运行
plt.ion()
plt.show()

# 迭代 1000 次学习，sess.run optimizer
for i in range(1000):
   # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
   sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
   if i % 50 == 0:
       # to visualize the result and improvement
       try:
           ax.lines.remove(lines[0])  # 在每一次绘图之前先讲上一次绘图删除，使得画面更加清晰
       except Exception:
           pass
       prediction_value = sess.run(predict_layer, feed_dict={xs: x_data, ys: y_data})
       # plot the prediction
       lines = ax.plot(x_data, prediction_value, 'r-', lw=5)  # 'r-'指绘制一个红色的线
       plt.pause(0.5)  # 指等待一秒钟


# 第二部分
# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
