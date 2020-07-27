# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:06:53 2020

@author: Angel
"""


import numpy as np
import cv2 as cv

from IPython.display import display, clear_output
from matplotlib import pyplot as plt

import os

os.chdir(os.path.dirname(__file__))

#path0 = "C:/Users/Angel/Documents/ITD/Inteligencia Artificial/GitHub/haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('./haarcascades/haarcascade_eye.xml')



##path = "C:/Users/Angel/Documents/ITD/Inteligencia Artificial/GitHub/pic.jpg"
##img = cv.imread(path)

img = cv.imread('./Assets/harr.jpg')



imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(imggray,cmap='gray',interpolation = 'bicubic') 
plt.show()


faces = face_cascade.detectMultiScale(imggray,1.1,1)

for(x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    roi_gray = imggray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for(ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

plt.imshow(img)
plt.show()

cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
    


