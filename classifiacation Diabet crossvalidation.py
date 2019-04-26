import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , KFold, cross_val_score
from sklearn import datasets
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import time
import os
from sklearn.cross_validation import cross_val_score, cross_val_predict
#-------Partie 1 -------
#--------Data cleaning----1------
os.getcwd()
os.chdir("C:\\Users\\ASUS\\Documents\\2emeIngDESC\\2semestre\\PythonForDataSc\\tp5")
Dataset = pd.read_csv('diabetes.csv', encoding='utf-8')
label=Dataset['Outcome']
data=Dataset.drop(["Outcome"],axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
scoresCV = []
model_svr = svm.SVR(kernel='rbf')
cv = KFold(n_splits=10, random_state=0, shuffle=False)
for train_index, test_index in cv.split(data):
     print("Train Index: ", train_index, "\n")
     print("Test Index: ", test_index)
     X_train, X_test, y_train, y_test = data[train_index], data[test_index], label[train_index], label[test_index]
     starttime=time.time()
     model_svr.fit(X_train, y_train)
     time_CV=time.time()-starttime
     scoresCV.append(model_svr.score(X_test, y_test))
     mean=np.mean(scoresCV)
     predict=cross_val_predict(model_svr, data, label, cv=10)
     plt.scatter(label,predict)
     accuracy = metrics.r2_score(label, predict)
     fpr,tpr,t=roc_curve(label,predict,pos_label=1)
     plt.figure()
     plt.plot(fpr,tpr)
     plt.xlabel("frp")
     plt.ylabel("trp")
     auc=auc(fpr,tpr)*100
     p=tpr/(tpr+fpr)*100
     meanP=np.mean(p)
     cm=confusion_matrix(label,predict.round())
     
    
     

    
     




