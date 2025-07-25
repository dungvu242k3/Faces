import os
import sys

import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR= os.path.join(BASE_DIR, "Silent-Face-Anti-Spoofing-master",'src')
sys.path.append(dir)

from modules.src.anti_spoof_predict import AntiSpoofPredict


class FaceDetector:
    def __init__(self, device='cpu'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

        model_path = "models/mini_fasnetv2.pth" 
        self.anti_spoof = AntiSpoofPredict()

    def detect_faces(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = self.mtcnn.detect(img)
        return boxes

    def extract_faces(self, frame, boxes, size=160):
        faces = []
        h, w, _ = frame.shape

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (size, size))
            pil_face = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
            result = self.anti_spoof.predict(pil_face)
            if result["label"] == 0:  
                print(f"[SPOOF] confidence={result['confidence']:.2f} → bỏ qua")
                continue
            print(f"[REAL] confidence={result['confidence']:.2f}")

            faces.append(pil_face)

        return faces
