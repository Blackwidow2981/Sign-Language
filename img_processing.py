import numpy as np
import cv2
Thresh_val=70
def preprocess(path):
    frame = cv2.imread(path) #get the image from source path
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray,(5,5),2) 
    
    ths = cv2.adaptiveThreshold(gauss,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, result = cv2.threshold(ths, Thresh_val, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return result

