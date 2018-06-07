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
def loaddata(Filename):
    data = pd.read_csv(Filename,sep=',',header = None)
    return np.array(data)
def get_tt():
    filelist=os.listdir("../data/data")
    print(filelist)
    abspath=os.path.abspath("..")
    for i in range(len(filelist)):
        tmp=loaddata(abspath+"/data/data/"+filelist[i])
        if i==0:
            data_train = tmp
            label_train =np.full((len(tmp),1),i)
        else:
            data_train = np.vstack((data_train,tmp[:len(tmp)-2]))
            tmp_label = np.full((len(tmp),1),i)
            label_train = np.vstack((label_train,tmp_label[:len(tmp_label)-2]))
        print(data_train.shape,label_train.shape)
    return data_train,label_train
        #print(abspath+"/data/data/"+filelist[i])
    pass
if __name__=="__main__":
    print('Test')
    get_tt()
