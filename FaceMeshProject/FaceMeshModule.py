import cv2
import time
import mediapipe as mp

# remove False from line 56 to show the face mesh


class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        # The below fashMesh only follow the RGB img / video
        # Press ctl and select the FaceMesh to see the properties
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        # We are changing the size of mesh
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def FindFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []  # This tuple is for more then one face
        if self.results.multi_face_landmarks:
            # We have to loop because we have more then one face
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)

                # We have to store every landmarks
                face = []       # This tuple store one by one value of faces
                # We are getting the information about where is the lips / nose by using below code of line
                for id, lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    # we multiply above value to normalize value and covert to pixel value
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.4, (0, 255, 0), 1)
                    #print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('C:/Users/vivek/PycharmProjects/Vivekcode/Computer-Vision/PoseEstimation/Confe.mp4')
    pTime = 0
    detector = FaceMeshDetector()

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


if __name__ == "__main__":
    main()