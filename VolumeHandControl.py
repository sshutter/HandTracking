import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math
import applescript

####################################
wCam, hCam = 640, 480
####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) > 0:
        
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        length = math.hypot(x2-x1, y2-y1)
        # print(length)
        
        cv2.circle(img, (x1,y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)
        
        if length < 30:
            cv2.circle(img, (cx,cy), 15, (0, 255, 0), cv2.FILLED)
            
        # Hand Range 30-250
        # Volume Range 1 - 100
        minVolume = 1
        maxVolume = 100
        vol = np.interp(length, [30,200], [minVolume, maxVolume])
        volBar = np.interp(length, [30,200], [400, 150])
        volPer = np.interp(length, [30,200], [0, 100])
        print(int(length), vol)
        target_volume = vol
        applescript.AppleScript("set volume output volume {}".format(target_volume)).run()
        
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (50, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
