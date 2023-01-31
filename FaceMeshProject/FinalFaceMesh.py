import cv2
import time
import FaceMeshModule as fmm

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('C:/Users/vivek/PycharmProjects/Vivekcode/Computer-Vision/PoseEstimation/Confe.mp4')
pTime = 0
detector = fmm.FaceMeshDetector()

while (cap.isOpened()):
    success, img = cap.read()
    img, faces = detector.FindFaceMesh(img)
    if len(faces) != 0:
        # Total we have 468 points on face
        print(len(faces))


    if success == True:
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
