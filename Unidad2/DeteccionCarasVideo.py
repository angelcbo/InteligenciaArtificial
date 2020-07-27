# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:12:43 2020

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


#cam = cv.VideoCapture(0)

cam = cv.VideoCapture('./Assets/video.mp4')


while(True):
    ret, frame = cam.read()
    imggray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imggray,1.3,2)
    for (x,y,w,h) in faces:
        
        cv.rectangle(frame,(x,y),(x+w, y+h), (255,255,255),3)
        
        roi_gray = imggray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    cv.imshow('Precione la tecla Esc para salir',frame)
    k = cv.waitKey(1) & 0xFF
    if k==27:
        break;
cam.release()
    
    
