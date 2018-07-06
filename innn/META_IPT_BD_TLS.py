#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:26:25 2018

@author: cadu
"""

print("66")
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt
import sys
sys.path.append("../src")
from data_prov2 import get_tt
## load data
data_train,label_train=get_tt()
############################################
label_train = np.resize(label_train,(len(label_train),))
X_train,X_test,y_train,y_test=cross_validation.train_test_split(data_train,label_train,test_size=0.2)
X_val,X_test,y_val,y_test=cross_validation.train_test_split(X_test,y_test,test_size=0.5)
print(X_train.shape,X_val.shape,X_test.shape,y_test.shape)
############################################
def extract_batch_size(_train, step, batch_size):# Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
   
    shape = list(_train.shape) #_X  7352 128 9
    shape[0] = batch_size      # 1500 128 9
    batch_s = np.empty(shape)
    for i in range(batch_size):
    # Loop index
        index = ((step-1)*batch_size + i) % len(_train) # step=1 
        batch_s[i] = _train[index] 

    return batch_s
############################################
import tensorflow as tf
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 671])
W = tf.Variable(tf.zeros([671, 8]))
b = tf.Variable(tf.zeros([8]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.int64, [None])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.losses.sparse_softmax_cross_entropy on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
batch_size=256
print_fre=10
for _ in range(1000):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs = extract_batch_size(X_train,_ ,batch_size)
    batch_xss = extract_batch_size(X_test,_ ,batch_size)
    batch_ys = extract_batch_size(y_train,_,batch_size)
    batch_yss = extract_batch_size(y_test,_,batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if _%print_fre==0:
        print(sess.run(
          accuracy, feed_dict={
              x: batch_xss,
              y_: batch_yss
        }))
        print(sess.run(
          tf.argmax(y, 1), feed_dict={
              x: batch_xss,
              y_: batch_yss
        }))
    