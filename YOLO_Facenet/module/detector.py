import cv2
from ultralytics import YOLO


class FaceDetector:
    def __init__(self, model_path='yolov8m.pt', device='cuda'):
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)

    def detect(self, frame, conf=0.5):
        results = self.model.predict(source=frame, conf=conf, verbose=False, device=self.device)[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes


    def extract_faces(self, frame, boxes, target_size=(160, 160)):
        faces = []
        for (x1, y1, x2, y2) in boxes:
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                faces.append(None)
                continue
            face = cv2.resize(face, target_size)
            faces.append(face)
        return faces
