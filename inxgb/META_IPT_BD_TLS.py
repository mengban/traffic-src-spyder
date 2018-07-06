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
data_train,label_train=get_tt(select_method='basedtree')
#############
label_train=np.resize(label_train,(len(label_train),))
X_train,X_test,y_train,y_test=cross_validation.train_test_split(data_train,label_train,test_size=0.2)
X_val,X_test,y_val,y_test=cross_validation.train_test_split(X_test,y_test,test_size=0.5)
print(X_train.shape,X_val.shape,X_test.shape,y_test.shape)
###
#X_train=X_train[:,0:X_train.shape[1]]
X_train = X_train[:,0:661]
X_test=X_test[:,0:X_train.shape[1]]
X_val=X_val[:,0:X_train.shape[1]]
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
# Results
print("        **************xgboost**************")
print(metrics.classification_report(y_test, pred,digits=4))
print("Classification report for classifier %s:\n%s\n"
      % (bst, metrics.classification_report(y_test, pred,digits=4)))
np.set_printoptions(threshold=100)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, pred))
confusion_m=metrics.confusion_matrix(y_test, pred)
for i in range(len(confusion_m)):
    print(confusion_m[i])
'''
import matplotlib.pyplot as plt
LABELS = [
    "0",
    "1",
    "2",
    "3",
    "4",
]
print("")
print("Precision: {}%".format(100*metrics.precision_score(expected, predicted, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(expected, predicted, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(expected, predicted, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(expected, predicted)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 9
height = 9
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(5) # n_classes
plt.xticks(tick_marks, LABELS, rotation=45)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
'''