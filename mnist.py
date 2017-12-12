# -*- coding: utf-8 -*-

"""
Copyright () 2017

All rights reserved by easyto

FILE: mnist.py
AUTHOR:  tianyuningmou
DATE CREATED:  @Time : 2017/11/27 上午10:23

DESCRIPTION:  .

VERSION: : #1 $
CHANGED By: : tianyuningmou $
CHANGE:  :  $
MODIFIED: : @Time : 2017/11/27 上午10:23
"""


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
