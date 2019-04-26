# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:45:11 2019

@author: ASUS
"""

import cv2
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
os.getcwd()
#---q1-----ouvrire des image--------
image=Image.open("cat.4001.jpg")
image1=mpimg.imread("cat.4001.jpg")
imag1=cv2.imread("cat.4001.jpg")
#----q2----- affichage des image----------
imgplot=plt.imshow(image)
imgplot1=plt.imshow(image1)
imgplotcv=plt.imshow(imag1)
#------q3---------convertion de l'image to noir  et blanc
ImageNew= cv2.cvtColor(imag1, cv2.COLOR_BGR2GRAY)
#---- q2 premier method affichage de l'image en noir et blanc 
immplot=plt.imshow(ImageNew,Cmap='gray')
#-----q4------ affichage de l'image noir et blanc en utilisant cv2 
immp=cv2.imshow('ImageNB',ImageNew)
cv2.waitKey(0)
cv2.destroyAllWindows()
#------ q5------------
ImageRGB= cv2.cvtColor(imag1,cv2.COLOR_BGR2RGB)
implot2=plt.imshow(ImageRGB)
#------q6--------
ImgNiGris=cv2.cvtColor(ImageRGB,cv2.COLOR_RGB2GRAY)
implot3=plt.imshow(ImgNiGris,Cmap='gray')
#converte image matrix  to vector
a=np.array(ImgNiGris)
shape=a.reshape(a.shape[0]*a.shape[1])
#-----q7---------
path=glob.glob("C:/Users/ASUS/Documents/2emeIngDESC/2semestre/PythonForDataSc/tp4/Animals/*.jpg")
Data=[]
for img in path:
    imagef=cv2.imread(img)
    ImageGrf=cv2.cvtColor(imagef,cv2.COLOR_BGR2GRAY)
    plt.imshow(ImageGrf,Cmap='gray')
    vect=np.array(ImageGrf)
    shapef=vect.reshape(vect.shape[0]*vect.shape[1])
    Data.append(shapef)
#-----Q8------------
Img8=cv2.imread('cat.4001.jpg')   
Zone=Img8[50:300,75:250]
plt.imshow(Zone)
#♣-----q9----
def filtre_verte(imager): 
    im = np.copy(imager) 
    for i in range(im.shape[0]): 
        for j in range(im.shape[1]): 
            r,v, b = im[i, j] 
            im[i, j] = (0,v,0) 
        return im
im=Image.open('cat.4001.jpg')
f=filtre_verte(im)
plt.imshow(f)
