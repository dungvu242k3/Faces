"""import pickle

import cv2
import torch
from detector import FaceDetector
from embeding import FaceEmbedder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = FaceDetector(device)
embedder = FaceEmbedder(device)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Add Face", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Add Face", 800, 600)

face_captured = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detector.detect_faces(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Add Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if boxes is not None and len(boxes) > 0:
            face_captured = detector.extract_faces(frame, boxes)[0]
            break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

if face_captured is None:
    exit()

name = input("Nhập tên: ")
embedding = embedder.get_embedding(face_captured)

try:
    with open("database.pkl", "rb") as f:
        db = pickle.load(f)
except FileNotFoundError:
    db = {}

db[name] = embedding.cpu()

with open("database.pkl", "wb") as f:
    pickle.dump(db, f)

print(f"Đã thêm: {name}")
"""

import cv2

#640 480
#1280 720 

class Camera:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

