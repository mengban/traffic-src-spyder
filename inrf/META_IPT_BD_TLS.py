#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:44:40 2018

@author: cadu
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
sys.path.append("../src")
from data_prov2 import get_tt
## load data
data_train,label_train=get_tt(select_method='chi2')
#############
label_train=np.resize(label_train,(len(label_train),))
X_train,X_test,y_train,y_test=cross_validation.train_test_split(data_train,label_train,test_size=0.2)
X_val,X_test,y_val,y_test=cross_validation.train_test_split(X_test,y_test,test_size=0.5)
print(X_train.shape,X_val.shape,X_test.shape,y_test.shape)
X_train=X_train[:,0:671]
X_test=X_test[:,0:671]
X_val=X_val[:,0:671]
print(X_train.shape,X_test.shape,X_val.shape)
#####################################

clf = RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(X_train, y_train)
pre=clf.predict(X_test)
#print("Classification report for classifier %s:\n%s\n"
      #% (clf, metrics.classification_report(y_test, pre,digits=4)))
print("        **************Random Forest**************")
print(metrics.classification_report(y_test, pre,digits=4))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test,pre))
print(clf.feature_importances_)






print(clf.feature_importances_)

#print(clf.predict([[0, 0, 0, 0]]))
