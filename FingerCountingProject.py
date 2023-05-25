import cv2
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)

folderPath = 'HandTracking/FingerImages'
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

pTime = 0
detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        fingers = []
        #Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers                
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalFingers = fingers.count(1)
        if totalFingers == 0:
            h, w, c = overlayList[5].shape
            img[0:h, 0:w] = overlayList[5]
        elif totalFingers == 1:
            h, w, c = overlayList[4].shape
            img[0:h, 0:w] = overlayList[4]
        elif totalFingers == 2:
            h, w, c = overlayList[2].shape
            img[0:h, 0:w] = overlayList[2]
        elif totalFingers == 3:
            h, w, c = overlayList[3].shape
            img[0:h, 0:w] = overlayList[3]
        elif totalFingers == 4:
            h, w, c = overlayList[0].shape
            img[0:h, 0:w] = overlayList[0]
        else:
            h, w, c = overlayList[1].shape
            img[0:h, 0:w] = overlayList[1]
            
        cv2.rectangle(img, (20, 425), (170, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 600), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, f' FPS: {int(fps)}', (1000, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('frame', img)
    cv2.waitKey(1)