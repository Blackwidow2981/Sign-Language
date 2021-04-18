import numpy as np
from keras.models import model_from_json
import operator
import cv2
from img_processing import *
import sys, os

# Loading the model
json_file = open("model-slr.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-slr.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

# Category dictionary
categories =  { '1': 0,
                '2': 1,
                '3': 2,
                '4': 3,
                '5': 4,
                '6': 5,
                '7': 6,
                '8': 7,
                '9': 8,
                'A': 9,
                'B': 10,
                'C': 11,
                'D': 12,
                'E': 13,
                'F': 14,
                'G': 15,
                'H': 16,
                'I': 17,
                'J': 18,
                'K': 19,
                'L': 20,
                'M': 21,
                'N': 22,
                'O': 23,
                'P': 24,
                'Q': 25,
                'R': 26,
                'S': 27,
                'T': 28,
                'U': 29,
                'V': 30,
                'W': 31,
                'X': 32,
                'Y': 33,
                'Z': 34}

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (128, 128)) 
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray,(5,5),2) 
    
    ths = cv2.adaptiveThreshold(gauss,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image= cv2.threshold(ths, Thresh_val, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow("test", test_image)
    # Batch of 1
    result = loaded_model.predict(test_image.reshape(1, 128, 128, 1))
    prediction = {'1': result[0][0], 
                  '2': result[0][1],
                  '3': result[0][2],
                  '4': result[0][3],
                  '5': result[0][4],
                  '6': result[0][5],
                  '7': result[0][6], 
                  '8': result[0][7],
                  '9': result[0][8],
                  'A': result[0][9],
                  'B': result[0][10],
                  'C': result[0][11],
                  'D': result[0][12],
                  'E': result[0][13],
                  'F': result[0][14],
                  'G': result[0][15],
                  'H': result[0][16],
                  'I': result[0][17],
                  'J': result[0][18],
                  'K': result[0][19],
                  'L': result[0][20],
                  'M': result[0][21],
                  'N': result[0][22],
                  'O': result[0][23],
                  'P': result[0][24],
                  'Q': result[0][25],
                  'R': result[0][26],
                  'S': result[0][27],
                  'T': result[0][28],
                  'U': result[0][29],
                  'V': result[0][30],
                  'W': result[0][31],
                  'X': result[0][32],
                  'Y': result[0][33],
                  'Z': result[0][34],
                  }
    # Sorting based on top prediction
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (30, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()