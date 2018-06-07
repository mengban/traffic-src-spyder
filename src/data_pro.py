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
    #data6 = loaddata(abspath+"/data/dataoutcapture_win15_5_1.csv")
    data7 = loaddata(abspath+"/data/dataoutcapture_win12_6_1.csv")
    data8 = loaddata(abspath+"/data/dataout2018-01-30_win10_7_1.csv")
    data9 = loaddata(abspath+"/data/dataout2018-01-30_win9_8_1.csv")
    data10 = loaddata(abspath+"/data/dataout2018-01-29_win7_9_1.csv")
    
    print("src data is:",data1.shape,data2.shape,data3.shape,data4.shape,data5.shape)
    print("src data is:",data5.shape,data7.shape,data8.shape,data9.shape,data10.shape)
    data_train = np.vstack((data1[:len(data1)-2],data2[:len(data2)-2]))
    data_train = np.vstack((data_train,data3[:len(data3)-2]))
    data_train = np.vstack((data_train,data4[:len(data4)-2]))
    data_train = np.vstack((data_train,data5[:len(data5)-2]))
    #data_train = np.vstack((data_train,data6[:len(data6)-1]))
    data_train = np.vstack((data_train,data7[:len(data7)-2]))
    data_train = np.vstack((data_train,data8[:len(data8)-2]))
    data_train = np.vstack((data_train,data9[:len(data9)-2]))
    data_train = np.vstack((data_train,data10[:len(data10)-2]))
    
    print('This is data_train',type(data_train),data_train.shape)
    #label
    data1_ = loaddata(abspath+"/data/labelout2018-05-03_win16_0_1.csv")
    data2_ = loaddata(abspath+"/data/labelout2018-05-03_win11_1_1.csv")
    data3_ = loaddata(abspath+"/data/labelout2018-04-04_win16_2_1.csv")
    data4_ = loaddata(abspath+"/data/labeloutcapture_win11_3_1.csv")
    data5_ = loaddata(abspath+"/data/labelout2018-03-01_win11_4_1.csv")
    #data6_ = loaddata(abspath+"/data/labeloutcapture_win15_5_1.csv")
    data7_ = loaddata(abspath+"/data/labeloutcapture_win12_6_1.csv")
    data8_ = loaddata(abspath+"/data/labelout2018-01-30_win10_7_1.csv")
    data9_ = loaddata(abspath+"/data/labelout2018-01-30_win9_8_1.csv")
    data10_ = loaddata(abspath+"/data/labelout2018-01-29_win7_9_1.csv")
    #print(data1_.shape,data2_.shape,data3_.shape,data4_.shape,data5_.shape)
    
    
    label_train = np.vstack((data1_[:len(data1_)-2],data2_[:len(data2_)-2]))
    label_train = np.vstack((label_train,data3_[:len(data3_)-2]))
    label_train = np.vstack((label_train,data4_[:len(data4_)-2]))
    label_train = np.vstack((label_train,data5_[:len(data5_)-2]))
    #label_train = np.vstack((label_train,data6_[:len(data6)-1]))
    label_train = np.vstack((label_train,data7_[:len(data7)-2]-1))
    label_train = np.vstack((label_train,data8_[:len(data8)-2]-1))
    label_train = np.vstack((label_train,data9_[:len(data9)-2]-1))
    label_train = np.vstack((label_train,data10_[:len(data10)-2]-1))
    print("Data loading is ...OK.")
    print("The total time is :",time.time()+startT)
    print(label_train.shape)
    return data_train,label_train
if __name__=='__main__':
    X,Y=get_tt()
    print(X.shape,Y.shape)