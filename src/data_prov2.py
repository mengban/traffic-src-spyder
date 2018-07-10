#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:11:50 2018

@author: cadu
"""

import pandas as pd
import numpy as np
import os
import time
import cv2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
def loaddata(Filename):
    data = pd.read_csv(Filename,sep=',',header = None)
    return np.array(data)
def get_tt(select_method='without'):
    filelist=os.listdir("../data/data")
    print(filelist)
    abspath=os.path.abspath("..")
    for i in range(len(filelist)):
        tmp=loaddata(abspath+"/data/data/"+filelist[i])
        if(len(tmp)<20000):
            tmp_copy = tmp
            for j in range(int(20000/len(tmp))+2):
                tmp = np.vstack((tmp,tmp_copy))
                print("x",int(20000/len(tmp))+2,"times")
        if i==0:
            print('define dara_train')
            data_train = tmp
            label_train =np.full((len(tmp),1),i)
        else:
            data_train = np.vstack((data_train,tmp[:len(tmp)-2]))
            tmp_label = np.full((len(tmp),1),i)
            label_train = np.vstack((label_train,tmp_label[:len(tmp_label)-2]))
        print(tmp.shape)
    # chi2
    #data_train = feature_select_chi(data_train,label_train)
    if select_method=='chi2':
        print('feature select method:chi2')
        data_train = feature_select_chi2(data_train[:,:661],label_train)
    elif select_method=='basedtree':
        print('feature select method:basedtree')
        data_train = feature_select_basetree(data_train[:,:661],label_train)
    elif select_method=='without':
        print('do nothing with the raw data')
    return data_train,label_train
        #print(abspath+"/data/data/"+filelist[i])
    pass
def feature_select_chi2(X_train,y_train):
    
    X_new_traffic= SelectKBest(chi2,k=100).fit_transform(X_train,y_train)
    return X_new_traffic
    pass
def feature_select_basetree(X_train,y_train):
    X=X_train
    y=y_train
    ##(150, 4)
    clf = ExtraTreesClassifier()
    ##clf =GradientBoostingClassifier()
    clf = clf.fit(X, y)
    clf.feature_importances_  
    ##array([ 0.04...,  0.05...,  0.4...,  0.4...])
    model = SelectFromModel(clf, prefit=True)
    X_new_traffic = model.transform(X)
    return X_new_traffic
def cal_correction(X_Matrix):
    print("")
    #X_Matrix = X_Matrix[0:100]
    print('X_M shape is:',X_Matrix.shape)
    print('相关系数为：')
    corrc = np.corrcoef(X_Matrix,rowvar=0)
    cv2.imwrite('chioo.jpg',np.rint(corrc*256))
    print(corrc)
    pd.DataFrame(corrc).to_csv('out.csv')
    return corrc
    pass
if __name__=="__main__":
    print('Data loading starts...')
    startT = -time.time()
    X_Matrix,label = get_tt(select_method='chi2')
    cal_correction(X_Matrix)
    print('Data loading is ...OK.')
    print('Total time is: ',time.time()+startT)
