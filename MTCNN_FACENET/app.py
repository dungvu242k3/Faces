import cv2
import numpy as np
import torch

from modules.camera import Camera
from modules.database import FaceDatabase
from modules.detector import FaceDetector
from modules.embeding import FaceEmbedder
from modules.processing_images import load_faces_from_folder
from modules.recognition import FaceRecognizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = FaceDetector(device)
    embedder = FaceEmbedder(device)
    database = FaceDatabase("database/database.pkl")
    load_faces_from_folder("train", detector, embedder, database, device=device)
    recognizer = FaceRecognizer(database, device=device)
    camera = Camera()

    

    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break

        boxes = detector.detect_faces(frame)
        faces = detector.extract_faces(frame, boxes) if boxes is not None else []

        unknown_faces = []

        if faces:
            for box, face_img in zip(boxes, faces):
                x1, y1, x2, y2 = map(int, box)
                is_real = True

                if not is_real:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "FAKE", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue

                embedding = embedder.get_embedding(face_img)
                name, dist, _ = recognizer.recognize(embedding)

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                label = name
                if name == "Unknown":
                    label = f"Unknown{len(unknown_faces) + 1}"
                    unknown_faces.append((embedding, face_img))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("a") and unknown_faces:
            print("/n==> Enter names for unknown faces:")
            for idx, (embedding, face_img) in enumerate(unknown_faces):
                window_name = f"Unknown Face {idx + 1}"
                cv2.imshow(window_name, cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
                name = input(f"Enter name for {window_name}: ").strip()
                if name:
                    database.add_face(name, embedding.cpu())
                    print(f" Added {name} to database.")
                cv2.destroyWindow(window_name)

            database.save_database()
            recognizer = FaceRecognizer(database, device=device)
            unknown_faces.clear()

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
