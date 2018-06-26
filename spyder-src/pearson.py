#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:53:02 2018

@author: cadu
"""
import numpy as np
import os
from scipy.stats import pearsonr
np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
print("Higher noise", pearsonr(x, x + np.random.normal(0, 100, size)))


def file_name(file_dir):   
    for root, dirs, files in os.walk(file_dir):  
        print(root) #当前目录路径  
        print(dirs) #当前路径下所有子目录  
        print(files) #当前路径下所有非目录子文件  
print(file_name("."))