import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # We have to input Hands(static_image_mode, max_num_hands, min_detection_confidence (in %), min_tracking_confidence(in %))
        # I am using the given parameters, hecne I am not writing anything
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.detectionCon, self.trackCon)
        # This is useful to draw the dots in hand means at our joints
        self.mpDraw = mp.solutions.drawing_utils
        # Here we are using just 6 senarieo
        self.tipIds = [4,8,12,16,20]     # Inside is the finger number

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # To give the color
        self.results = self.hands.process(imgRGB) # Processes an RGB image and returns the hand landmarks and handedness of each detected hand
        # print(results.multi_hand_landmarks) # multi_hand_landmarks: it shows the detection and coordinates of hand
        # Now we have to detect if there is multiple hands or not


        if self.results.multi_hand_landmarks:
             for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # mpHands.HAND_CONNECTIONS: this function will useful to connect the dots with line (green color used inbuilt)
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.lmlist = []
        if self.results.multi_hand_landmarks:   # If hand is detect
            myhand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape     # this will give height and width of image
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print("\n", id, cx, cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    # This below line of code show the first position of point and total we have 20
                    # circle(image, points, thickness, color, filled otherwise it will not show
                    cv2.circle(img, (cx, cy), 8, (255,0,255), cv2.FILLED)
        return self.lmlist

    def fingersUp(self):
        # Now we know that 8 is above 6 means finger open, hence lmList[finger point position][y-axis]
        fingers = []
        # This is only for the thumb and in x-axis
        if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    # Here p means previous and c means current
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)     # most important is to write img
        lmlist = detector.findPosition(img)     # if write here, (img, drwa=False) then the pink points will go
        if len(lmlist) != 0:
            print(lmlist[4])


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # putText(image, fps in int, frame location, write text, scale, color, thickness of font)
        cv2.putText(img, str(int(fps)), (10, 78), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # The function imshow displays an image in the specified window
        cv2.imshow("Image", img)
        cv2.waitKey(3)

if __name__ == "__main__":
    main()