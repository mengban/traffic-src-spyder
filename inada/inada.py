#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:07:59 2018

@author: cadu
"""
from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, svm, metrics
from sklearn import cross_validation
import numpy as np


import sys
sys.path.append("../src")
from data_prov2 import get_tt
## load data
data_train,label_train=get_tt(select_method='without')
#############
label_train=np.resize(label_train,(len(label_train),))
X_train,X_test,y_train,y_test=cross_validation.train_test_split(data_train,label_train,test_size=0.2)
X_val,X_test,y_val,y_test=cross_validation.train_test_split(X_test,y_test,test_size=0.5)
print(X_train.shape,X_val.shape,X_test.shape,y_test.shape)
###
#X_train=X_train[:,0:X_train.shape[1]]
X_train = X_train[:,666:X_train.shape[1]]
#X_train = np.delete(X_train,[0+661,3+661,8+661],axis=1)
X_test=X_test[:,666:X_test.shape[1]]
#X_test = np.delete(X_test,[0+661,3+661,8+661],axis=1)
X_val=X_val[:,666:X_val.shape[1]]
#X_val = np.delete(X_val,[0+661,3+661,8+661],axis=1)
print('X_train.shape',X_train.shape,X_test.shape,X_val.shape)

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")

bdt_real.fit(X_train, y_train)
bdt_discrete.fit(X_train, y_train)

real_test_errors = []
discrete_test_errors = []
#real_pred = bdt_real.staged_predict(X_test)
#discrete_pred = bdt_discrete.staged_predict(X_test)
'''
print ('real predicting, classification error=%f' 
       % (sum( int(real_pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))
print ('discrete predicting, classification error=%f' 
       % (sum( int(discrete_pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))
'''
real_pred = []
dis_pred = []
for real_test_predict, discrete_train_predict in zip(
        bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(
        1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))
    real_pred.append(real_test_predict)
    dis_pred.append(discrete_train_predict)
    
min_index = real_test_errors.index(min(real_test_errors))
pred = real_pred[min_index]
from myplot import plot_roc,plot_confusionM  
print('##############adaboost#############')
print("Classification report for classifier %s:\n%s\n"
      % ('adaboost', metrics.classification_report(y_test, pred,digits=4)))
np.set_printoptions(threshold=100)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, pred))
plot_roc(y_test,pred,'')
plot_confusionM(y_test,pred)


n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1),
         discrete_test_errors, c='black', label='SAMME')
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='SAMME.R')
plt.legend()
plt.ylim(0.18, 1.)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')

plt.subplot(132)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
         "b", label='SAMME', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='SAMME.R', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
         max(real_estimator_errors.max(),
             discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
         "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()
