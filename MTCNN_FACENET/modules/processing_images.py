import os

import cv2
from PIL import Image


def load_faces_from_folder(folder_path, detector, embedder, database, device="cpu"):
    existing_files = database.get_filenames()

    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            full_path = os.path.join(person_folder, filename)

            if full_path in existing_files:
                continue  
            image = cv2.imread(full_path)
            boxes = detector.detect_faces(image)

            if boxes is not None and len(boxes) > 0:
                faces = detector.extract_faces(image, boxes)
                if faces:
                    embedding = embedder.get_embedding(faces[0])
                    database.add_face(person_name, embedding.cpu(), filename=full_path)
                    print(f"Added {person_name} - {filename}")

    database.save_database()

