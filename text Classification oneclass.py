#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:56:43 2019

@author: aymen
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
from sklearn import metrics
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

os.chdir('/home/aymen/Desktop/DSEN s2/ML/tp3 ML')

""" Partie I : Data Cleaning """
""" Méthode 1 """
df = pd.read_csv('spam.csv',delimiter=',',encoding='latin-1')

df=df.drop("Unnamed: 2",1)
df=df.drop("Unnamed: 3",1)
df=df.drop("Unnamed: 4",1)

X=df["v2"]
Y=df["v1"]

""" transformer les donnéé textuel on numerique """
from sklearn.preprocessing import LabelEncoder
Model = LabelEncoder()
Y = Model.fit_transform(Y)
Y = Y.reshape(-1,1)

""" import keras """
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

""" couper chaque phrase en des mot est les indexé par des numero de 0 à 1000 """ 
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)

""" -7- le nombre des features dans la nouvelle base sequences  
est variable selon chaque observation 
car elle represente le nembre des mots dans la phrase du l'observation """

""" -8- attribuer pour chaque observation sur la nouvelle base sequence
 une taille fixe qui est le taille maximum nombre est pour le reste on remplie
 les valeurs manquante par des zeros """
 
""" -9- """
sequence_t=sequence.pad_sequences(sequences, padding='post')
 
""" Méthode 2 """
""" -1- """
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X1 = vectorizer.fit_transform(X)
X_t=X1
print(X1)

""" -2- determiner chaque mot du dataset est donner un score qui lui represonte
 directement sous la forme d'un vector"""
 
""" Partie 2 : Classification """
""" Méthode 1 """

""" transformation y & sequence_t on une seul dataframe dataset """
df["v1"].value_counts()

label=pd.DataFrame(data=Y[0:,0],columns=list('Y'))
label["Y"].value_counts()

data=pd.DataFrame(data=sequence_t[0:,0:])
Dataset_np=np.concatenate((data, label), axis=1)
""" Dataset est une dataframe d'une concatination du label Y est le data cliné sequence_t """
Dataset=pd.DataFrame(data=Dataset_np[0:,0:])

""" on utilise la technique de one class car on a unbalenste data 
0    4825  |  ham     4825
1     747  |  spam     747
"""
""" one_class """
Dataset_selected1=Dataset.loc[Dataset[172].isin([1])]
Dataset_selected0=Dataset.loc[Dataset[172].isin([0])]

Label1=Dataset_selected1[172]
Label0=Dataset_selected0[172]

Dataset_selected1=Dataset_selected1.drop(172,1)
Dataset_selected0=Dataset_selected0.drop(172,1)

TrainData1, TestData1, Trainlabel1, Testlabel1 = train_test_split(Dataset_selected1,Label1,test_size=0.333,random_state=0)

testlabel=np.concatenate((Testlabel1, (Label0)-1 ), axis=0)

testData=np.concatenate((TestData1, Dataset_selected0 ), axis=0)

""" //////////////////////////////////////////////////////////////////// """

""" svm_oneClass_rbf """
model_rbf = svm.OneClassSVM(kernel='rbf',gamma=0.2,nu=0.1)

start_time_rbf=time.time()
model_rbf.fit(TrainData1)
calcul_time_rbf=time.time()-start_time_rbf

y_pred_rbf = model_rbf.predict(testData)

""" accuracy_rbf """
acc_rbf=accuracy_score(testlabel,y_pred_rbf)

""" auc_rbf  """
fpr_rbf,tpr_rbf,T_rbf=roc_curve(testlabel,y_pred_rbf,pos_label=1)
auc_rbf=auc(fpr_rbf,tpr_rbf)*100

""" presition_rbf  """
pres_rbf=(tpr_rbf[1]/(tpr_rbf[1]+fpr_rbf[1]))*100

""" confusion_matrix_rbf  """
cm_rbf=confusion_matrix(testlabel,y_pred_rbf)

""" //////////////////////////////////////////////////////////////////// """

""" svm_oneClass_linear """
model_linear = svm.OneClassSVM(kernel='linear',nu=0.1)

start_time_linear=time.time()
model_linear.fit(TrainData1)
calcul_time_linear=time.time()-start_time_linear

y_pred_linear = model_linear.predict(testData)

""" accuracy_linear """
acc_linear=accuracy_score(testlabel,y_pred_linear)

""" auc_linear  """
fpr_linear,tpr_linear,T_linear=roc_curve(testlabel,y_pred_linear,pos_label=1)
auc_linear=auc(fpr_linear,tpr_linear)*100

""" presition_linear  """
pres_linear=(tpr_linear[1]/(tpr_linear[1]+fpr_linear[1]))*100

""" confusion_matrix_linear  """
cm_linear=confusion_matrix(testlabel,y_pred_linear)

""" //////////////////////////////////////////////////////////////////// """

""" svm_oneClass_poly """
model_poly = svm.OneClassSVM(kernel='poly',gamma=0.2,coef0=0.1,degree=8,nu=0.1)

start_time_poly=time.time()
model_poly.fit(TrainData1)
calcul_time_poly=time.time()-start_time_poly

y_pred_poly = model_poly.predict(testData)

""" accuracy_poly """
acc_poly=accuracy_score(testlabel,y_pred_poly)

""" auc_poly  """
fpr_poly,tpr_poly,T_poly=roc_curve(testlabel,y_pred_poly,pos_label=1)
auc_poly=auc(fpr_poly,tpr_poly)*100

""" presition_poly  """
pres_poly=(tpr_poly[1]/(tpr_poly[1]+fpr_poly[1]))*100

""" confusion_matrix_poly  """
cm_poly=confusion_matrix(testlabel,y_pred_poly)

""" //////////////////////////////////////////////////////////////////// """

""" svm_oneClass_sigmoid """
model_sigmoid = svm.OneClassSVM(kernel='sigmoid',gamma=0.2,coef0=0.1,nu=0.1)

start_time_sigmoid=time.time()
model_sigmoid.fit(TrainData1)
calcul_time_sigmoid=time.time()-start_time_sigmoid

y_pred_sigmoid = model_sigmoid.predict(testData)

""" accuracy_sigmoid """
acc_sigmoid=accuracy_score(testlabel,y_pred_sigmoid)

""" auc_sigmoid  """
fpr_sigmoid,tpr_sigmoid,T_sigmoid=roc_curve(testlabel,y_pred_sigmoid,pos_label=1)
auc_sigmoid=auc(fpr_sigmoid,tpr_sigmoid)*100

""" presition_sigmoid  """
pres_sigmoid=(tpr_sigmoid[1]/(tpr_sigmoid[1]+fpr_sigmoid[1]))*100

""" confusion_matrix_sigmoid  """
cm_sigmoid=confusion_matrix(testlabel,y_pred_sigmoid)

""" //////////////////////////////////////////////////////////////////// """

""" roc_curve  """
plt.figure()

plt.plot(fpr_rbf,tpr_rbf,'g--',label='rbf', linewidth=3)
plt.plot(fpr_linear,tpr_linear,'r--',label='linear', linewidth=3)
plt.plot(fpr_poly,tpr_poly,'b--',label='poly', linewidth=3)
plt.plot(fpr_sigmoid,tpr_sigmoid,'y--',label='sigmoid', linewidth=3)

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()
plt.title('roc curve')


""" Méthode 2 """

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB



 
 
