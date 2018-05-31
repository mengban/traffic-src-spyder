#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:19:14 2018

@author: cadu
"""

print("SVM")
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
train_X,test_X,train_Y,test_Y=cross_validation.train_test_split(data_train,label_train,test_size=0.1)

print('This is test_X',type(test_X),test_X.shape)
print('This is test_Y',type(test_Y),test_Y.shape)
print('This cell has done...')
#####################################
classifier = svm.SVC(gamma=0.001)

timeStart=time.time()
# learn
print('Training starts...')
classifier.fit(train_X, train_Y)

timeHalf=time.time()
print('Train train total time is:',timeHalf-timeStart)
#predicted

print('Test starts...')
expected =test_Y
predicted=classifier.predict(test_X)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
timeEnd=time.time()
print('The test time is',timeEnd-timeHalf)