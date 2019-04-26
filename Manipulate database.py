import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.model_selection import train_test_split
#--------lecture data 1 ------------#
Data1 = []
Classe1= open('Classe1.csv','r')
fichiercsv = csv.reader(Classe1)
#--------parcours de fichier ligne par ligne -------#
for row in fichiercsv:
#---------fonction append pour que chaque ligne on le data1----------#
 Data1.append(row)
 #---------lecture data 2 ----------------#
Data2 = []
Classe2= open('Classe2.csv','r') 
fichiercsv1 = csv.reader(Classe2)
for row in fichiercsv1:
    Data2.append(row)
#--------500 ligne , 1 une cologne -----------------------#
v1 = np.zeros((500,1))
v2 = np.ones((500,1))  
#--------- concatination par colonne axos=0 sinon axis=1 en ligne-----------------#
Label = np.concatenate((v1,v2),axis=0)
#------Matrice Data -------------#
Data = np.concatenate((Data1,Data2),axis=0)
#----------------creation d'un fichier py ----------------#
np.save('DataNew',Data)
#-----------remplissage du data dans le nouveau fichier DataNew
D=np.load('DataNew.npy')
#---labelnew------------#
np.save('LabelNew',Label)
V=np.load('LabelNew.npy')
#-----------------teste data train une seul ligne pas deux ligne car il en cas de deux ligne il va prendre des index incompatibla donc une seul ligne , 9ssamna data size train_size ou test_size si train_data =  ----#
Train_Data , Test_Data , Train_Label ,Test_Label = train_test_split(Data,Label,test_size=0.33)

