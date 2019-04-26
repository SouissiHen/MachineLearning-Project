
from PIL import Image
import matplotlib.image as mpimg
import os
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import seaborn as sns
#Q2
""" 15 class , 18 personne """
os.getcwd()
os.chdir("C:\\Users\\ASUS\\Documents\\2emeIngDESC\\2semestre\\PythonForDataSc\\tp8\\yalefaces")
#Q3
image=Image.open("subject01.centerlight")
vect=np.array(image)
#Q4
shape=vect.reshape(vect.shape[0]*vect.shape[1])
#Q5
path=glob.glob("C:/Users/ASUS/Documents/2emeIngDESC/2semestre/PythonForDataSc/tp8/yalefaces/*")
Data=[]
for img in path:
    imagef=Image.open(img)
    vect=np.array(imagef)
    shapef=vect.reshape(vect.shape[0]*vect.shape[1])
    Data.append(shapef)
#Q6
pca=PCA(n_components=165)  
pc1=pca.fit_transform(Data)
y=[]
for i in range(1,16):
    path=glob.glob("C:/Users/ASUS/Documents/2emeIngDESC/2semestre/PythonForDataSc/tp8/yalefaces/subject"+str(i).zfill(2)+"*")
    for fname in path:
        y.append(i)
x_train,x_test,y_train,y_test=train_test_split(Data,y,test_size=0.33,random_state=0)
#Q7
svc_l=svm.SVC(kernel="linear",C=0.7, decision_function_shape='ovo')
starttime_l=time.time()
svc_l.fit(x_train,y_train)
time_l=time.time()-starttime_l
y_predict=svc_l.predict(x_test)
acc_l=accuracy_score(y_test,y_predict)
print(classification_report(y_test, y_predict))  
cm=confusion_matrix(y_test,y_predict)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#kerner RBF
svc_rbf=svm.SVC(kernel="rbf", C=10,gamma=0.5,decision_function_shape='ovo')
starttime_rbf=time.time()
svc_rbf.fit(x_train,y_train)
time_rbf=time.time()-starttime_rbf
y_predictrbf=svc_rbf.predict(x_test)
acc_rbf=accuracy_score(y_test,y_predictrbf)

print(classification_report(y_test, y_predictrbf))  
cmrbf=confusion_matrix(y_test,y_predictrbf)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cmrbf), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#kerner Poly
svc_poly=svm.SVC(kernel="poly", coef0=0.1,gamma=0.2,degree=8,decision_function_shape='ovo')
starttime_poly=time.time()
svc_poly.fit(x_train,y_train)
time_poly=time.time()-starttime_poly
y_predictpoly=svc_poly.predict(x_test)
acc_poly=accuracy_score(y_test,y_predictpoly)

print(classification_report(y_test, y_predictpoly))  
cmpoly=confusion_matrix(y_test,y_predictpoly)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cmpoly), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#kerner sig
svc_sig=svm.SVC(kernel="sigmoid", C=0.7,decision_function_shape='ovo')
starttime_sig=time.time()
svc_sig.fit(x_train,y_train)
time_sig=time.time()-starttime_sig
y_predictsig=svc_sig.predict(x_test)
acc_sig=accuracy_score(y_test,y_predictsig)

print(classification_report(y_test, y_predictsig))  
cmsig=confusion_matrix(y_test,y_predictsig)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cmsig), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#-------7---calculate ACC------

#Q9LDA
lda=LDA()
NewData2=lda.fit_transform(Data,y)
x_train1,x_test1,y_train1,y_test1=train_test_split(NewData2,y,test_size=0.33,random_state=0)

starttime_l=time.time()
lda.fit(x_train1,y_train1)
time_lDA=time.time()-starttime_l

y_predict1=lda.predict(x_test1)
acc_LDA=accuracy_score(y_test1,y_predict1)

print(classification_report(y_test, y_predict1))  
cmlda=confusion_matrix(y_test,y_predict1)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cmlda), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
