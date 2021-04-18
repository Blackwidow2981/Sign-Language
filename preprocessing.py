import numpy as np
import cv2
import os
import csv
from img_processing import *

if not os.path.exists("data1"):
    os.makedirs("data1")
if not os.path.exists("data1/train"):
    os.makedirs("data1/train")
if not os.path.exists("data1/test"):
    os.makedirs("data1/test")

path="data1"
subpath="train"

label=0
count=0
count1=0
count2=0

for(dirpath,dirnames,filenames) in os.walk("data/"+subpath):
   for dirname in dirnames:
       print(dirname) 
       for(dpath,dnames,files) in os.walk("data/"+subpath+"/"+dirname):
       	    if not os.path.exists(path+"/train/"+dirname):
                os.makedirs(path+"/train/"+dirname)
            if not os.path.exists(path+"/test/"+dirname):
                os.makedirs(path+"/test/"+dirname)
            num=0.75*len(files) 
            i=0
            #print(len(files))
            for file in files:
                count=count+1
                actual_path="data/"+subpath+"/"+dirname+"/"+file
                actual_path1=path+"/"+"train/"+dirname+"/"+file
                actual_path2=path+"/"+"test/"+dirname+"/"+file
                image= cv2.imread(actual_path,0)
                bw_filter= preprocess(actual_path)
                if i<num:
                    count1 = count1+1
                    cv2.imwrite(actual_path1 , bw_filter)
                else:
                    count2 = count2+ 1
                    cv2.imwrite(actual_path2 , bw_filter)
                i=i+1
       label= label+1
print(count)
print(count1)
print(count2)
