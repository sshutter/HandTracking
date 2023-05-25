import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

################################
brushThickness = 20
eraserThickness = 50
################################

# Headers
folderPath = 'HandTracking/Headers'
myList = os.listdir(folderPath)
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[7]
drawColor = (100, 100, 100)

# Hand tracking
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    
while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    
    # 2.Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
    
        # 3.Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
    
        # 4.If selection mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # Checking for the click
            if y1 < 125:
                if 60 < x1 < 170:
                    header = overlayList[7]
                    drawColor = (100, 100, 100)
                elif 210 < x1 < 320:
                    header = overlayList[5]
                    drawColor = (0, 0, 255)
                elif 360 < x1 < 470:
                    header = overlayList[6]
                    drawColor = (230, 224, 91)
                elif 510 < x1 < 620:
                    header = overlayList[1]
                    drawColor = (173, 74, 0)
                elif 650 < x1 < 760:
                    header = overlayList[2]
                    drawColor = (98, 191, 3)
                elif 800 < x1 < 910:
                    header = overlayList[4]
                    drawColor = (195, 103, 254)
                elif 950 < x1 < 1060:
                    header = overlayList[3]
                    drawColor = (255, 83, 140)
                elif 1100 < x1 < 1220:
                    header = overlayList[0]
                    drawColor = (0, 0, 0)
                    
            if drawColor == (0, 0, 0):
                cv2.circle(img, (x1, y1), 20, (255, 255, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 20, (255, 255, 255), cv2.FILLED)
            else:
                cv2.circle(img, (x1, y1), 20, drawColor, cv2.FILLED)
                cv2.circle(img, (x2, y2), 20, drawColor, cv2.FILLED)
    
        # 5.If drawing mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 20, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
                
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), (255, 255, 255), eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                
            xp, yp = x1, y1
            
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)
    
    # setting the header image
    img[0:125, 0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow('Image', img)
    # cv2.imshow('Canvas', imgCanvas)
    cv2.waitKey(1)