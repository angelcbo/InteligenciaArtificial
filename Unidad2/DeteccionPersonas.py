# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:28:40 2020

@author: Angel
"""


import numpy as np 
import cv2 as cv
from IPython.display import display, clear_output
from matplotlib import pyplot as plt

import os

os.chdir(os.path.dirname(__file__))


body_cascade = cv.CascadeClassifier('./haarcascades/haarcascade_fullbody.xml')

img = cv.imread('./Assets/personas4.jpg')
imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

bodys = body_cascade.detectMultiScale(imggray,1.1,3)


for(x,y,w,h) in bodys:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    
    
plt.imshow(img)
plt.show()


cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()

