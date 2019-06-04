import cv2
import numpy as np

def main():
    image = cv2.imread('dinesh.png')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray, 100,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((1,200), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)
    ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    i = len(sorted_ctrs) - 1

    for _, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        if h < 10:
	    i -= 1
            continue
        roi = gray[y-1 : y+h+1, x : x+w]
        processLine(roi, i)
        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
        i -= 1

    cv2.imshow('marked areas',image)
    cv2.waitKey(0)

def processLine(line, lineNum):
    print lineNum
    ret,thresh = cv2.threshold(line,127,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((line.shape[0], 10), np.uint8) #word:10
    img_dilation = cv2.dilate(thresh, kernel, iterations=2) #word:2
    ctrs, _ = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
    i = 0
    for _, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = line[y-1 : y+h+1, x : x+w]
        cv2.rectangle(line,(x,y),( x + w, y + h ),(90,0,255),1)
        cv2.imwrite('word ' + str(lineNum) + str(i) + ".png", roi)
        i += 1
          
    cv2.imshow('line ' + str(lineNum), line)

if __name__ == '__main__':
    main()
