#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:53:58 2018

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
# dataset
    print("Data loading...")
    startT=-time.time()
    abspath=os.path.abspath("..")
    data1 = loaddata(abspath+"/data/dataout2018-05-03_win16_0_1.csv")
    data2 = loaddata(abspath+"/data/dataout2018-05-03_win11_1_1.csv")
    data3 = loaddata(abspath+"/data/dataout2018-04-04_win16_2_1.csv")
    data4 = loaddata(abspath+"/data/dataoutcapture_win11_3_1.csv")
    data5 = loaddata(abspath+"/data/dataout2018-03-01_win11_4_1.csv")    
    
    print("src data is:",data1.shape,data2.shape,data3.shape,data4.shape,data5.shape)
    data_train = np.vstack((data1[:len(data1)-1],data2[:len(data2)-1]))
    data_train = np.vstack((data_train,data3[:len(data3)-1]))
    data_train = np.vstack((data_train,data4[:len(data4)-1]))
    data_train = np.vstack((data_train,data5[:len(data5)-1]))
    
    print('This is data_train',type(data_train),data_train.shape)
    #label
    data1_ = loaddata(abspath+"/data/labelout2018-05-03_win16_0_1.csv")
    data2_ = loaddata(abspath+"/data/labelout2018-05-03_win11_1_1.csv")
    data3_ = loaddata(abspath+"/data/labelout2018-04-04_win16_2_1.csv")
    data4_ = loaddata(abspath+"/data/labeloutcapture_win11_3_1.csv")
    data5_ = loaddata(abspath+"/data/labelout2018-03-01_win11_4_1.csv")
    #print(data1_.shape,data2_.shape,data3_.shape,data4_.shape,data5_.shape)
    
    
    label_train = np.vstack((data1_[:len(data1_)-1],data2_[:len(data2_)-1]))
    label_train = np.vstack((label_train,data3_[:len(data3_)-1]))
    label_train = np.vstack((label_train,data4_[:len(data4_)-1]))
    label_train = np.vstack((label_train,data5_[:len(data5_)-1]))
    print("Data loading is ...OK.")
    print("The total time is :",time.time()+startT)
    return data_train,label_train
if __name__=='__main__':
    X,Y=get_tt()
    print(X.shape,Y.shape)