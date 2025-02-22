import numpy as np
from sklearn import preprocessing, neighbors
import sklearn.model_selection
import pandas as pd
import os
import random
import cv2

image_TRAIN=[]
label_TRAIN=[]
for i in os.listdir("KNN/Parasitized"):
    path=os.path.join("KNN/Parasitized",i)
    try:
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(128,128))
        img=img.flatten()
        img=img/255
        image_TRAIN.append(img)
        label_TRAIN.append(0)
    except:
        print('Image Unreadable')

for i in os.listdir("KNN/Uninfected"):
    path=os.path.join("KNN/Uninfected",i)
    try:
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(128,128))
        img=img/255
        img=img.flatten()
        image_TRAIN.append(img)
        label_TRAIN.append(1)
    except:
        print('Image Unreadable')

trainx,testx,trainy,testy=sklearn.model_selection.train_test_split(image_TRAIN,label_TRAIN,test_size=0.2)

trainx=np.array(trainx)
# trainx=trainx.reshape(-1,1)

testx=np.array(testx)
# testx=testx.reshape(-1,1)

trainy=np.array(trainy)
testy=np.array(testy)

print(trainx.shape,testx.shape)
classifier=neighbors.KNeighborsClassifier()
classifier.fit(trainx,trainy)

print(trainx.shape,trainy.shape,testx.shape,testy.shape)

y_predict=classifier.predict(testx)

accuracy=classifier.score(testx,y_predict)
print(accuracy)