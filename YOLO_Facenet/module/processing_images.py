import os

import cv2


def load_faces_from_folder(folder_path, detector, embedder, database, device='cpu'):
    existing_files = set(database.get_filenames())  # Ảnh đã xử lý

    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for file_name in os.listdir(person_folder):
            full_path = os.path.join(person_folder, file_name)
            if full_path in existing_files:
                continue  # ❌ bỏ qua ảnh đã xử lý

            image = cv2.imread(full_path)
            if image is None:
                continue

            boxes = detector.detect(image)
            if not boxes:
                continue

            face_img = detector.extract_faces(image, boxes[:1])[0]
            if face_img is None:
                continue

            embedding = embedder.get_embedding(face_img)
            database.add_face(person_name, embedding, filename=full_path)
            print(f"✅ Đã thêm mới: {full_path}")
