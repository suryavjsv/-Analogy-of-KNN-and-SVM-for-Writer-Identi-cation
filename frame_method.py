import cv2
import numpy as np

def main():
    character = cv2.imread('word 75.png')
    gray = cv2.cvtColor(character, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100,255,cv2.THRESH_BINARY_INV)
    thresh = cv2.resize(thresh , (40 , 40))

    intensities = []

    for i in range(0, 40, 10):
        for j in range(0, 40, 10):
            roi = thresh[i : i + 10, j : j + 10];
            intensities.append(findAverageIntensity(roi))

    print(intensities)

#Calculate the average intensity of the region
def findAverageIntensity(region):
    sum = 0
    for i in range(len(region)):
        for j in range(len(region[i])):
            if region[i][j] == 255:
               sum = sum + 1

    return sum


if __name__ == '__main__':
    main()
