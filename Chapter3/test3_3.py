import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


dataset = np.loadtxt('watermelon_3a.csv', delimiter=',')

X = dataset[:, 1:3]
Y = dataset[:, 3]
# print(Y,Y.shape)
# f1 = plt.figure(1)
# plt.title('watermelon_3a')
# plt.xlabel('density')
# plt.ylabel('ratio_sugar')
# plt.scatter(X[Y == 0,0], X[Y == 0,1],marker='o', color ='b', s=100,label='bad')
# plt.scatter(X[Y==1,0],X[Y==1,1],marker='o',color='g',s=100,label='good')
# plt.legend(loc='upper right')
# plt.show()

# generalization of train and test dataset
Xtrain, Xtest, Ytrain, Ytest = model_selection.train_test_split(X,Y,test_size=0.5,random_state=0)

#model training
log_model = LogisticRegression()
log_model.fit(Xtrain,Ytrain)

y_pred = log_model.predict(Xtest)

print(metrics.confusion_matrix(Ytest,y_pred))
print(metrics.classification_report(Ytest,y_pred))

