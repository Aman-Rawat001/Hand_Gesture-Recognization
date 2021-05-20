import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLns in results.multi_hand_landmarks:
            # find index number id, cx, cy(pos of fingers).
            for (id, ln) in enumerate(handLns.landmark):
                # print(id, ln)
                h, w, c = img.shape
                cx, cy = int(ln.x*w), int(ln.y*h)
                print(id, cx, cy)
                if id == 0: # id means we 21 points ex: id==4 is thumb tip etc.
                    # drawing a filled circle in position o in landmark
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLns, mpHands.HAND_CONNECTIONS)

    # for fps calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # it shows the fps on our screen
    cv2.putText(img, "fps:"+str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
