import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn import cross_validation
import xgboost as xgb
import time
import matplotlib.pyplot as plt
#%matplotlib inline
def loaddata(Filename):
    data = pd.read_csv(Filename,sep=',',header = None)
    return np.array(data)
# dataset
data1 = loaddata("../data/dataout2018-05-03_win16_0_1.csv")
data2 = loaddata("../data/dataout2018-05-03_win11_1_1.csv")
data3 = loaddata("../data/dataout2018-04-04_win16_2_1.csv")
data4 = loaddata("../data/dataoutcapture_win11_3_1.csv")
data5 = loaddata("../data/dataout2018-03-01_win11_4_1.csv")

print(data1.shape,data2.shape,data3.shape)
data_train = np.vstack((data1[:len(data1)-1],data2[:len(data2)-1]))
data_train = np.vstack((data_train,data3[:len(data3)-1]))
data_train = np.vstack((data_train,data4[:len(data4)-1]))
data_train = np.vstack((data_train,data5[:len(data5)-1]))

print('This is data_train',type(data_train),data_train.shape)
#label
data1_ = loaddata("../data/labelout2018-05-03_win16_0_1.csv")
data2_ = loaddata("../data/labelout2018-05-03_win11_1_1.csv")
data3_ = loaddata("../data/labelout2018-04-04_win16_2_1.csv")
data4_= loaddata("../data/labeloutcapture_win11_3_1.csv")
data5_ = loaddata("../data/labelout2018-03-01_win11_4_1.csv")
print(data1_.shape,data2_.shape,data3_.shape,data4_.shape,data5_.shape)


label_train = np.vstack((data1_[:len(data1_)-1],data2_[:len(data2_)-1]))
label_train = np.vstack((label_train,data3_[:len(data3_)-1]))
label_train = np.vstack((label_train,data4_[:len(data4_)-1]))
label_train = np.vstack((label_train,data5_[:len(data5_)-1]))
#print(label_test[100:800])
train_X,test_X,train_Y,test_Y=cross_validation.train_test_split(data_train,label_train,test_size=0.1)

print('This is test_X',type(test_X),test_X.shape)
print('This is test_Y',type(test_Y),test_Y.shape)
print('This cell has done...')
#####################################
from xgboost import plot_importance
xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.8
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 12
param['num_class'] = 5
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 20     ## boosting迭代计算次数
bst = xgb.train(param, xg_train, num_round, watchlist )

plot_importance(bst)
plt.show()

#importance  = bst.get_fscore(fmap='xgb.fmp')
# get prediction
pred = bst.predict( xg_test );
print(pred)

print ('predicting, classification error=%f' 
       % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))

####################################
import pandas as pd
importance = bst.get_fscore()
print(pd.Series(importance).sort_values(ascending=False))
####################################
# Results
expected = test_Y  
predicted = pred
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
width = 12
height = 12
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