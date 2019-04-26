from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression 
import pandas as pd
from sklearn import svm
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#----- question 1 ----------
#♥----strucutre de dataset c'est list--------
dataset = load_iris()
Data=dataset.data
label=dataset.target
#---Q3--- donnee les nombre c'est voir le target_names on trouve 3 class----- 
#---4-------------
x_train1,x_test1,y_train1,y_test1=train_test_split(Data,label,test_size=0.33,random_state=0)
model_svc=svm.SVC(kernel="rbf",C=10,gamma=0.5,)

starttime=time.time()
model_svc.fit(x_train1,y_train1)
time_svc=time.time()-starttime

y_predict=model_svc.predict(x_test1)

acc=accuracy_score(y_test1,y_predict)
#acc roc curve ne fonctionne pas dans le multi class----- fait attention ---
#--------------
print(classification_report(y_test1, y_predict))  

cm=confusion_matrix(y_test1,y_predict)
#---Parti 2 regression logistique ------------
#avec les donnée  binaire -------
os.getcwd()
os.chdir("C:\\Users\\ASUS\\Documents\\2emeIngDESC\\2semestre\\PythonForDataSc\\tp5")
Dataset = pd.read_csv('diabetes.csv', encoding='utf-8')
label=Dataset['Outcome']
data=Dataset.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.33,random_state=0)
Reg_lg=LogisticRegression()

starttime_RL=time.time()
Reg_lg.fit(x_train,y_train)
time_RL=time.time()-starttime_RL

y_predict_RL=Reg_lg.predict(x_test)
#----Acc------
acc_RL=accuracy_score(y_test,y_predict_RL)

print(classification_report(label, y_predict_RL))

CM=confusion_matrix(y_test,y_predict_RL)  

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(CM), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



