import cv2
import math

# Try to import MediaPipe; if unavailable we provide a dummy detector.
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("Warning: MediaPipe not found. Hand detection will be disabled.")

if HAS_MEDIAPIPE:
    class handDetector:
        def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
            self.mode = mode
            self.maxHands = maxHands
            self.detectionCon = detectionCon
            self.trackCon = trackCon
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(
                self.mode,
                self.maxHands,
                min_detection_confidence=self.detectionCon,
                min_tracking_confidence=self.trackCon,
            )
            self.mpDraw = mp.solutions.drawing_utils
            self.mpFaceDetection = mp.solutions.face_detection
            self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)
            self.tipIds = [4, 8, 12, 16, 20]

        def findHands(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            if self.results.multi_hand_landmarks and draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            return img

        def findPosition(self, img, handNo=0, draw=True):
            self.lmList = []
            if self.results.multi_hand_landmarks:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return self.lmList

        def fingersUp(self, lmList, handNo=0):
            if not lmList:
                return []
            fingers = []
            # Thumb (simple x‑check for right hand)
            if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # Other four fingers
            for id in range(1, 5):
                if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers

        def findFaces(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.resultsFace = self.faceDetection.process(imgRGB)
            bboxs = []
            if self.resultsFace.detections:
                for detection in self.resultsFace.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    bbox = (
                        int(bboxC.xmin * iw),
                        int(bboxC.ymin * ih),
                        int(bboxC.width * iw),
                        int(bboxC.height * ih),
                    )
                    bboxs.append(bbox)
                    if draw:
                        img = self.fancyDraw(img, bbox)
                        cv2.rectangle(img, bbox, (255, 0, 255), 2)
                        cv2.putText(
                            img,
                            f"{int(detection.score[0] * 100)}%",
                            (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN,
                            2,
                            (255, 0, 255),
                            2,
                        )
            return bboxs

        def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
            x, y, w, h = bbox
            x1, y1 = x + w, y + h
            cv2.rectangle(img, bbox, (255, 0, 255), rt)
            # Top‑Left
            cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
            cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
            # Top‑Right
            cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
            cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
            # Bottom‑Left
            cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
            cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
            # Bottom‑Right
            cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
            cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
            return img
else:
    # Dummy fallback – provides the same API but does nothing.
    class handDetector:
        def __init__(self, *_, **__):
            pass
        def findHands(self, img, draw=True):
            return img
        def findPosition(self, img, handNo=0, draw=True):
            return []
        def fingersUp(self, lmList, handNo=0):
            return []
        def findFaces(self, img, draw=True):
            return []
        def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
            return img