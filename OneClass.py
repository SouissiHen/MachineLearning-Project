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
import os
from sklearn.cross_validation import cross_val_score, cross_val_predict
#-------Partie 1 -------
#--------Data cleaning----1------
os.getcwd()
os.chdir("C:\\Users\\ASUS\\Documents\\2emeIngDESC\\2semestre\\PythonForDataSc\\tp5")
Dataset = pd.read_csv('diabetes.csv', encoding='utf-8')
#------------ Model OneClass training ----
Dataset["Outcome"].value_counts()
#---10------Data cleaning ----------
dataset_selected1=Dataset.loc[Dataset['Outcome'].isin([1])]
dataset_selected0=Dataset.loc[Dataset['Outcome'].isin([0])]
label1=dataset_selected1['Outcome']
#label01=dataset_selected0['Outcome']
label0=dataset_selected0['Outcome']-1
data1=dataset_selected1.drop(['Outcome'],axis=1)
data0=dataset_selected0.drop(['Outcome'],axis=1)
x_train1,x_test1,y_train1,y_test1=train_test_split(data1,label1,test_size=0.33,random_state=0)
#---------concatenate x_test1:donnée de label1 |||| data0: donnée de  label0-------
testData=np.concatenate((x_test1,data0),axis=0)
##-----concatinate y_test1:label1 : outcome 1  ||| label0 : outcome 0  ----------
label_test=np.concatenate((y_test1,label0),axis=0)

model_oneclass=svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)
model_linear=svm.OneClassSVM(kernel='linear',coef0=0.3)
model_sig=svm.OneClassSVM(kernel='sigmoid',coef0=0.3)
model_poly=svm.OneClassSVM(kernel='poly',coef0=0.1,gamma=0.2,degree=8)
##-----train--------------
starttime=time.time()
model_oneclass.fit(x_train1)
time_onerbf=time.time()-starttime

starttime=time.time()
model_linear.fit(x_train1)
linear_time=time.time()-starttime

starttime=time.time()
model_sig.fit(x_train1)
time_sig=time.time()-starttime

starttime=time.time()
model_poly.fit(x_train1)
time_poly=time.time()-starttime
#--------Predection----------
y_predict1=model_oneclass.predict(testData)
y_predict_Linear=model_linear.predict(testData)
y_predict_sig=model_sig.predict(testData)
y_predict_poly=model_poly.predict(testData)
##----ACCURACY-----------
accOneClass=accuracy_score(label_test,y_predict1)
acclinear=accuracy_score(label_test,y_predict_Linear)
accSig=accuracy_score(label_test,y_predict_sig)
accPoly=accuracy_score(label_test,y_predict_poly)
#---Roc curve----
fpr_oneclass,tpr_oneclass,t_oneclass=roc_curve(label_test,y_predict1,pos_label=1)
fpr_linear,tpr_linear,t_linear=roc_curve(label_test,y_predict_Linear,pos_label=1)
fpr_sig,tpr_sig,t_sig=roc_curve(label_test,y_predict_sig,pos_label=1)
fpr_poly,tpr_poly,t_poly=roc_curve(label_test,y_predict_poly,pos_label=1)

plt.figure()
plt.plot(fpr_oneclass,tpr_oneclass)
plt.xlabel("frp_oneclass")
plt.ylabel("trp_oneclass")

plt.figure()
plt.plot(fpr_linear,tpr_linear)
plt.xlabel("frp_linear")
plt.ylabel("trp_linear")


plt.figure()
plt.plot(fpr_sig,tpr_sig)
plt.xlabel("frp_sigmoid")
plt.ylabel("trp_sigmoid")

plt.figure()
plt.plot(fpr_poly,tpr_poly)
plt.xlabel("frp_poly")
plt.ylabel("trp_poly")
#---- AUC---------
auc_onerbf=auc(fpr_oneclass,tpr_oneclass)*100
auc_linear=auc(fpr_linear,tpr_linear)*100
auc_sig=auc(fpr_sig,tpr_sig)*100
auc_poly=auc(fpr_poly,tpr_poly)*100
#------ precision---------
p_onecalss=tpr_oneclass/(tpr_oneclass+fpr_oneclass)*100
p_linear=tpr_linear/(tpr_linear+fpr_linear)*100
p_sig=tpr_sig/(tpr_sig+fpr_sig)*100
p_poly=tpr_poly/(tpr_poly+fpr_poly)*100
#------confusion matrix -------------
cm_oneclassrbf=confusion_matrix(label_test,y_predict1)
cm_linear=confusion_matrix(label_test,y_predict_Linear)
cm_sig=confusion_matrix(label_test,y_predict_sig)
cm_poly=confusion_matrix(label_test,y_predict_poly)


