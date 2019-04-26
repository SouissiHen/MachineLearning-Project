import keras
import os
import pandas as pd
#--Import Biblio keras-----
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
os.getcwd()
os.chdir('C:/Users/ASUS/Documents/2emeIngDESC/2semestre/PythonForDataSc/tp7')
#----------1------------
df = pd.read_csv('C:/Users/ASUS/Documents/2emeIngDESC/2semestre/PythonForDataSc/tp7/spam.csv',delimiter=',',encoding='latin-1')
#☺--------2------------------
df=df.drop("Unnamed: 2",1)
df=df.drop("Unnamed: 3",1)
df=df.drop("Unnamed: 4",1)
#------------3---4-------------
Y=df['v1']
X=df['v2']
#---------5----------------
Model = LabelEncoder()
Y = Model.fit_transform(Y)
Y = Y.reshape(-1,1)
#---------6----------
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
#-------7--- : nbr de feature variable 50 , 486 , kol nombre presente une mot pour corriger le prb on ajoute des zero -------
#-------8-------
#--9-----------
seq=sequence.pad_sequences(sequences,padding='post')
#----------------Method 2-----------
vectorizer = TfidfVectorizer()
X1 = vectorizer.fit_transform(X)
print(X1)
#--------------- Donc On utilise one class parsque 4825 ham , spam 747 unblanced data   -----------
df['v1'].value_counts()
#-----Partie classification----------------
x_train,x_test,y_train,y_test=train_test_split(seq,Y,test_size=0.33,random_state=0)
print(x_train)

dataset_selected_ham=df.loc[df['v1'].isin(['ham'])]
dataset_selected_spam=df.loc[df['v1'].isin(['spam'])]



#-----------SVM .SVC : 
svc_rbf=svm.SVC(kernel="rbf",C=10,gamma=0.5)
svc_rbf.fit(x_train,y_train)
y_predict=svc_rbf.predict(x_test)
acc_rbf=accuracy_score(y_test,y_predict)
fpr_rbf,tpr_rbf,t_rbf=roc_curve(y_test,y_predict,pos_label=1)
plt.figure()
plt.plot(fpr_rbf,tpr_rbf)
plt.xlabel("frp_rbf")
plt.ylabel("trp_rbf")
auc_rbf=auc(fpr_rbf,tpr_rbf)*100
p_rbf=tpr_rbf/(tpr_rbf+fpr_rbf)*100
cm_rbf=confusion_matrix(y_test,y_predict)

#----------Naive Bayes---------------
clf = GaussianNB()
clf.fit(x_train, y_train)
y_predict_clf=clf.predict(x_test)
acc_clf=accuracy_score(y_test,y_predict)
fpr_clf,tpr_clf,t_clf=roc_curve(y_test,y_predict_clf,pos_label=1)
plt.figure()
plt.plot(fpr_clf,tpr_clf)
plt.xlabel("frp_clf")
plt.ylabel("trp_clf")