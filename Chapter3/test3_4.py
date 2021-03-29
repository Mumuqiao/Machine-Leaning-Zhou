import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut

data_set = np.loadtxt('iris.csv',delimiter=',', skiprows=1, usecols=(0,1,2,3))
X = data_set[:, 0:4]
Y = np.loadtxt('iris.csv',dtype=str,delimiter=',', skiprows=1, usecols=(4))
#print(Y)
f1 = plt.figure(1)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.scatter(X[Y=='Iris-setosa',0],X[Y=='Iris-setosa',1],color='b',marker='o',
            s=100,label='setosa')
plt.scatter(X[Y=='Iris-versicolor',0],X[Y=='Iris-versicolor',1],color='g',marker='o',
            s=100,label='versicolor')
plt.scatter(X[Y=='Iris-virginica',0],X[Y=='Iris-virginica',1],color='r',marker='o',
            s=100,label='virginica')
plt.legend(loc='upper right')
plt.show()


log_model = LogisticRegression()
y_pred = cross_val_predict(log_model,X,Y,cv=10)
print(metrics.accuracy_score(Y,y_pred))

loo = LeaveOneOut()
accuracy = 0
for train,test in loo.split(X):
    log_model.fit(X[train],Y[train])
    y_pred = log_model.predict(X[test])
    if y_pred == Y[test] :
        accuracy+=1
print(accuracy/X.shape[0])