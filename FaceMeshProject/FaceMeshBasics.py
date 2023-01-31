import cv2
import time
import mediapipe as mp

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('C:/Users/vivek/PycharmProjects/Vivekcode/Computer-Vision/PoseEstimation/Confe.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
# The below fashMesh only follow the RGB img / video
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) # Press ctl and select the FaceMesh to see the properties
# We are changing the soze of mesh
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while (cap.isOpened()):
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        # We have to loop because we have more then one face
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)

            # We are getting the information about where is the lips / nose by using below code of line
            for id, lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                # we multiply above value to normalize value and covert to pixel value
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)


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