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

        self.W_xi = initializer([self.D_input, self.D_cell])
        self.W_hi = initializer([self.D_cell, self.D_cell])
        self.b_i = tf.Variable(tf.zeros([self.D_cell]))

        self.W_xf = initializer([self.D_input, self.D_cell])
        self.W_hf = initializer([self.D_cell, self.D_cell])
        self.b_f = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))

        self.W_xo = initializer([self.D_input, self.D_cell])
        self.W_ho = initializer([self.D_cell, self.D_cell])
        self.b_o = tf.Variable(tf.zeros([self.D_cell]))

        pass
