from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data_xtrain= pd.read_csv('x_train.csv')
data_ytrain= pd.read_csv('y_train.csv',header=None)
data_xtest= pd.read_csv('x_test.csv')
data_ytest= pd.read_csv('y_test.csv',header=None)

print('data_xtrain=',data_xtrain)
print('data_ytrain=',data_ytrain)

meanerror_train=[]
meanerror_test=[]
error_std_array = []

for i in range(30):
    error_train=[]
    error_test=[]
    for j in range(10):
        X_train, X_train_pick, y_train, y_train_pick = train_test_split(data_xtrain, data_ytrain, test_size=1/3)
        RF= RandomForestClassifier(n_estimators=i+1,bootstrap=True, max_features=3)
        RF.fit(X_train_pick,y_train_pick.values.ravel())
        train_predict=RF.predict(data_xtrain)
        test_predict=RF.predict(data_xtest)
        error_train=np.append(error_train,1-metrics.accuracy_score(data_ytrain,train_predict))
        error_test=np.append(error_test, 1 - metrics.accuracy_score(data_ytest, test_predict))
    meanerror_train=np.append(meanerror_train,np.mean(error_train))
    meanerror_test=np.append(meanerror_test,np.mean(error_test))
    error_std_array=np.append(error_std_array,np.std(error_test))
print('meanerror_train=',meanerror_train)
print('meanerror_test=',meanerror_test)
print('sample of std of error rate on testing set=',error_std_array)

X = np.arange(1,31)
plt.plot(X, meanerror_train,label='train data')
plt.plot(X, meanerror_test,label='test data')
plt.ylabel('Mean error')
plt.xlabel('Number of trees')
plt.legend()
plt.show()

plt.plot(X, error_std_array)
plt.ylabel('Sample standard deviation of error rate on testing set')
plt.xlabel('Number of trees')
plt.show()

