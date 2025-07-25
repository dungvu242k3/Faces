import cv2
import numpy as np
import torch

from module.camera import Camera
from module.database import FaceDatabase
from module.detector import FaceDetector
from module.embedding import FaceEmbedder
from module.livenessnet import LivenessNet
from module.processing_images import load_faces_from_folder
from module.recognition import FaceRecognizer


def save_face_to_db(name, face_img, db, embedder, recognizer):
    if face_img is None or face_img.size == 0:
        return
    embedding = embedder.get_embedding(face_img)
    db.add_face(name, embedding, filename=None)
    db.save_database()
    recognizer.db = db.get_all()
    print(f"Đã lưu khuôn mặt cho: {name}")

def preprocess_for_liveness(face_img):
    face = cv2.resize(face_img, (32, 32))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0)
    return face

device = "cuda" if torch.cuda.is_available() else "cpu"
liveness = LivenessNet(width=32, height=32, depth=3, classes=2).to(device)
liveness.load_weights("best_model.pth")
liveness.eval()

def main():
    cam = Camera()
    detector = FaceDetector()
    embedder = FaceEmbedder()
    db = FaceDatabase()
    load_faces_from_folder("train", detector, embedder, db)
    recognizer = FaceRecognizer(db.get_all())

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        boxes = detector.detect(frame)
        unknown_faces = []

        for (x1, y1, x2, y2) in boxes:
            face_img = frame[y1:y2, x1:x2]
            if face_img is None or face_img.size == 0:
                continue

            tensor = preprocess_for_liveness(face_img).to(device)
            with torch.no_grad():
                pred = torch.argmax(liveness(tensor), dim=1).item()

            if pred == 0:
                label = "Fake"
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                continue

            embedding = embedder.get_embedding(face_img)
            name, dist = recognizer.recognize(embedding)

            if name == "Unknown":
                unknown_faces.append((face_img, (x1, y1, x2, y2)))

            color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
            label = name
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s') and boxes:
            name = input("Nhập tên: ").strip()
            if name:
                x1, y1, x2, y2 = boxes[0]
                face_img = frame[y1:y2, x1:x2]
                save_face_to_db(name, face_img, db, embedder, recognizer)
        elif key == ord('a') and unknown_faces:
            name = input("Nhập tên cho người 'Unknown': ").strip()
            if name:
                face_img, _ = unknown_faces[0]
                save_face_to_db(name, face_img, db, embedder, recognizer)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
