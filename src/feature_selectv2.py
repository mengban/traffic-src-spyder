#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 19:26:09 2018

@author: cadu
"""

from data_prov2 import *
from feature_selector import FeatureSelector
import time
import pandas as pd
X_Matrix,label = get_tt(select_method='basedtree')

startT = -time.time()
fs = FeatureSelector(data=pd.DataFrame(X_Matrix),labels=label)
fs.identify_collinear(correlation_threshold = 0.8)
fs.plot_collinear()
# Pass in the appropriate parameters
fs.identify_zero_importance(task = 'classification', 
 eval_metric = 'auc', 
 n_iterations = 10, 
 early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']
#63 features with zero importance after one-hot encoding.
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)

fs.identify_low_importance(cumulative_importance = 0.99)

X_Matrix,label = get_tt(select_method='without')
startT = -time.time()
fs = FeatureSelector(data=pd.DataFrame(X_Matrix),labels=label)
fs.identify_collinear(correlation_threshold = 0.8)
fs.plot_collinear(plot_all=True)
# Pass in the appropriate parameters
fs.identify_zero_importance(task = 'classification', 
 eval_metric = 'auc', 
 n_iterations = 10, 
 early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']
#63 features with zero importance after one-hot encoding.
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)

fs.identify_low_importance(cumulative_importance = 0.99)