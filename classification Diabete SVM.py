import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import time
#-------Partie 1 -------
#--------Data cleaning----1------
Dataset = pd.read_csv('diabetes.csv', encoding='utf-8')
#------2----------
label=Dataset['Outcome']
#------3 SUPRIMER UNE COLONE ----------
data=Dataset.drop(['Outcome'],axis=1)
#------CLASSIFICATION---------
#----4---------
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.33,random_state=0)
#---5---------- svm.SVC-------train
#######---------classification SVM binaire KERNEL RBF-------------
model=svm.SVC(kernel='rbf',C=0.7,gamma=1)
#---------fitting--------
model.fit(x_train,y_train)
#------6----test-------
y_predict=model.predict(x_test)
#-------7---calculate ACC------
acc=accuracy_score(y_test,y_predict)
#################-----Kernel lineaire SVM binaire  --------
model=svm.SVC(kernel='linear',C=0.7)
#---------fitting--------
#------ COMPLEXITE DE CALCULE ------
starttime=time.time()
model.fit(x_train,y_train)
time=time.time()-starttime
#------6----test-------
y_predict=model.predict(x_test)
#-------7---calculate ACC------
acc=accuracy_score(y_test,y_predict)
#-------ROC_CURVE----
fpr,tpr,t=roc_curve(y_test,y_predict,pos_label=1)
plt.figure()
plt.plot(fpr,tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.legend()
#-----AUC-------
auc1=auc(fpr,tpr)*100
#------MATRICE DE CONFUSION--------
cm=confusion_matrix(y_test,y_predict)
#------PRECESION-----
p=tpr/(tpr+fpr)*100
#---------problem overfitiing---------
xpredict=model.predict(x_train)
accu=accuracy_score(y_train,xpredict)
##------------8----AFFICHER LES NOMBRE D'OBSERVATION DE CHAQUE CLASSE--------
Dataset["Outcome"].value_counts()
#----9--QU'EST VOUS REMARQUEFR-------- 
#reponce 9  : unbalanced Data =>svm one class
#---10------Data cleaning ----------
dataset_selected1=Dataset.loc[Dataset['Outcome'].isin([1])]
dataset_selected0=Dataset.loc[Dataset['Outcome'].isin([0])]
label1=dataset_selected1['Outcome']
#label0=dataset_selected0['Outcome']
label0=dataset_selected0['Outcome']-1
data1=dataset_selected1.drop(['Outcome'],axis=1)
data0=dataset_selected0.drop(['Outcome'],axis=1)
###---creation de model------
x_train1,x_test1,y_train1,y_test1=train_test_split(data1,label1,test_size=0.33,random_state=0)
model_oneclass=svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.5)
##-----train--------------
model_oneclass.fit(x_train1)
testData=np.concatenate((x_test1,data0),axis=0)
##----test----
y_predict1=model_oneclass.predict(testData)
##-----conatinate label-----
label_test=np.concatenate((y_test1,label0),axis=0)
##----ACCURACY-----------
accOneClass=accuracy_score(label_test,y_predict1)
#------LABEL0-------
testData1=np.concatenate((x_test1,data0),axis=0)
y_predict10=model_oneclass.predict(testData1)
##-----conatinate label-----
label_test1=np.concatenate((y_test1,label0),axis=0)
##----ACCURACY-----------
accOneClass1=accuracy_score(label_test1,y_predict10)

