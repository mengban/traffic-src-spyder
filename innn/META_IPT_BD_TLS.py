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
import xgboost as xgb
import time
import matplotlib.pyplot as plt
#%matplotlib inline
def loaddata(Filename):
    data = pd.read_csv(Filename,sep=',',header = None)
    return np.array(data)
# dataset
data1 = loaddata("../data/dataout2018-05-03_win16_0_1.csv")
data2 = loaddata("../data/dataout2018-05-03_win11_1_1.csv")
data3 = loaddata("../data/dataout2018-04-04_win16_2_1.csv")
data4 = loaddata("../data/dataoutcapture_win11_3_1.csv")
data5 = loaddata("../data/dataout2018-03-01_win11_4_1.csv")

print(data1.shape,data2.shape,data3.shape)
data_train = np.vstack((data1[:len(data1)-1],data2[:len(data2)-1]))
data_train = np.vstack((data_train,data3[:len(data3)-1]))
data_train = np.vstack((data_train,data4[:len(data4)-1]))
data_train = np.vstack((data_train,data5[:len(data5)-1]))

print('This is data_train',type(data_train),data_train.shape)
#label
data1_ = loaddata("../data/labelout2018-05-03_win16_0_1.csv")
data2_ = loaddata("../data/labelout2018-05-03_win11_1_1.csv")
data3_ = loaddata("../data/labelout2018-04-04_win16_2_1.csv")
data4_= loaddata("../data/labeloutcapture_win11_3_1.csv")
data5_ = loaddata("../data/labelout2018-03-01_win11_4_1.csv")
print(data1_.shape,data2_.shape,data3_.shape,data4_.shape,data5_.shape)


label_train = np.vstack((data1_[:len(data1_)-1],data2_[:len(data2_)-1]))
label_train = np.vstack((label_train,data3_[:len(data3_)-1]))
label_train = np.vstack((label_train,data4_[:len(data4_)-1]))
label_train = np.vstack((label_train,data5_[:len(data5_)-1]))
#print(label_test[100:800])
#train_X,test_X,train_Y,test_Y=cross_validation.train_test_split(data_train,label_train,test_size=0.1)

#print('This is test_X',type(test_X),test_X.shape)
#print('This is test_Y',type(test_Y),test_Y.shape)
#print('This cell has done...')
############################################
label_train = np.resize(label_train,(len(label_train),))
print(label_train.shape)
X_train,X_test,y_train,y_test=cross_validation.train_test_split(data_train,label_train,test_size=0.2)
X_val,X_test,y_val,y_test=cross_validation.train_test_split(X_test,y_test,test_size=0.5)
print(X_train.shape,X_val.shape,X_test.shape,y_test.shape)
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
x = tf.placeholder(tf.float32, [None, 661])
W = tf.Variable(tf.zeros([661, 5]))
b = tf.Variable(tf.zeros([5]))
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
for _ in range(1000):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs = extract_batch_size(X_train,_ ,128)
    batch_xss = extract_batch_size(X_test,_ ,128)
    batch_ys = extract_batch_size(y_train,_,128)
    batch_yss = extract_batch_size(y_test,_,128)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(
      accuracy, feed_dict={
          x: batch_xss,
          y_: batch_yss
    }))