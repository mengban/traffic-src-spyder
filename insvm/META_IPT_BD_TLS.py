#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:19:14 2018

@author: cadu
"""

import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn import cross_validation
import xgboost as xgb
import time
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
sys.path.append("../src")
from data_prov2 import get_tt
## load data
data_train,label_train=get_tt()
#############
label_train=np.resize(label_train,(len(label_train),))
X_train,X_test,y_train,y_test=cross_validation.train_test_split(data_train,label_train,test_size=0.2)
X_val,X_test,y_val,y_test=cross_validation.train_test_split(X_test,y_test,test_size=0.5)
print(X_train.shape,X_val.shape,X_test.shape,y_test.shape)
X_train=X_train[:,0:660]
X_test=X_test[:,0:660]
X_val=X_val[:,0:660]
print(X_train.shape,X_test.shape,X_val.shape)
#####################################
classifier = svm.SVC(gamma=0.001)

timeStart=time.time()
# learn
print('Training starts...')
classifier.fit(X_train, y_train)

timeHalf=time.time()
print('Train train total time is:',timeHalf-timeStart)
#predicted

print('Test starts...')
expected =y_test
predicted=classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
timeEnd=time.time()
print('The test time is',timeEnd-timeHalf)