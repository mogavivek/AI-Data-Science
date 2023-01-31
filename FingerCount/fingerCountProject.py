import cv2
import time
import os
import handtracking_module as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

# Here we are using just 6 senarieo
tipIds = [4,8,12,16,20]     # Inside is the finger number

# To store the image we are using the os
folderPath = "C:/Users/vivek/PycharmProjects/Vivekcode/Computer-Vision/FingerCount/fingers"
myList = os.listdir(folderPath)
#print(myList)
overlayList = []
# In below code the imPath is 1.jpg, 2.jpg means it will run 6 times because we have 6 images
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    print(f'{folderPath}/{imPath}')
    overlayList.append(image)       # We are saving our images

print(len(overlayList))


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    if len(lmList) != 0:
        # Now we know that 8 is above 6 means finger open, hence lmList[finger point position][y-axis]
        fingers = []
        # This is only for the thumb and in x-axis
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)
        # means below code run like that, I put 1 means in tuple shows [1,1,1,1,1] then it will show 5 open fingers and
        # if it is [0,1,1,0,0] then it will show 2
        totalFingers = fingers.count(1)
        print(totalFingers)
        # We are slicing the and giving the width and height of the image img[width, height]
        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers]
        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime


    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
