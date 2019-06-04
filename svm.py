import cv2
import numpy as np
import os

def preprocessing(character):
	#character = cv2.imread('word 80.png')
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100,255,cv2.THRESH_BINARY_INV)
    thresh = cv2.resize(thresh , (40 , 40))

    intensities = []

    for i in range(0, 40, 10):
        for j in range(0, 40, 10):
            roi = thresh[i : i + 10, j : j + 10];
            intensities.append(findAverageIntensity(roi))

    #print(intensities)
    return intensities

#Calculate the average intensity of the region
def findAverageIntensity(region):
    sum = 0
    for i in range(len(region)):
        for j in range(len(region[i])):
            if region[i][j] == 255:
               sum = sum + 1

    return sum


x_train = []
y_train = []

for i in range(8):
	files = os.listdir("./fyp/writer" + str(i+1))
	files = [file for file in files if ".png" in file]

	for file in files:
		img = cv2.imread("./fyp/writer" + str(i+1) + "/" + file)
		#print(img.shape)
		# img = cv2.resize(img,(32,64))
		# img = (img - np.mean(img))/np.std(img)
		img = preprocessing(img)
		#img = np.ndarray.flatten(img)
		x_train.append(img)
		y_train.append(i)

print(np.array(x_train).shape,len(y_train))

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.25,random_state=4)

#x_train 
clf = svm.SVC(kernel='sigmoid',gamma='scale',decision_function_shape='ovr')
clf.fit(x_train,y_train)	


y_pred = clf.predict(x_train)	
print(accuracy_score(y_pred,y_train))

print(confusion_matrix(y_train,y_pred))

y_pred = clf.predict(x_test)	
print(accuracy_score(y_pred,y_test))

print(confusion_matrix(y_test,y_pred))
		
