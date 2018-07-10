import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn import cross_validation
import xgboost as xgb
import time
import matplotlib.pyplot as plt
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
X_train=X_train[:,0:X_train.shape[1]]
#X_train = X_train[:,666:X_train.shape[1]]
#X_train = np.delete(X_train,[0+661,3+661,8+661],axis=1)
X_test=X_test[:,0:X_test.shape[1]]
#X_test = np.delete(X_test,[0+661,3+661,8+661],axis=1)
X_val=X_val[:,0:X_val.shape[1]]
#X_val=X_val[:,666:X_val.shape[1]]
#X_val = np.delete(X_val,[0+661,3+661,8+661],axis=1)
print('X_train.shape',X_train.shape,X_test.shape,X_val.shape)
#####################################
from xgboost import plot_importance
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.8
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 14
param['num_class'] = 8
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 30     ## boosting迭代计算次数
bst = xgb.train(param, xg_train, num_round, watchlist )

plot_importance(bst)
plt.show()

#importance  = bst.get_fscore(fmap='xgb.fmp')
# get prediction
pred = bst.predict( xg_test );
print(pred)

print ('predicting, classification error=%f' 
       % (sum( int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))

####################################
#import pandas as pd
#importance = bst.get_fscore()
#print(pd.Series(importance).sort_values(ascending=False))
####################################
#  pred,
print('##############xgboost#############')
print("Classification report for classifier %s:\n%s\n"
      % (bst, metrics.classification_report(y_test, pred,digits=4)))
np.set_printoptions(threshold=100)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, pred))
confusion_m=metrics.confusion_matrix(y_test, pred)
#for i in range(len(confusion_m)):
    #print(confusion_m[i])
    
from myplot import plot_roc,plot_confusionM  
plot_roc(y_test,pred)
plot_confusionM(y_test,pred)
