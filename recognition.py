import argparse
import pickle
import time
from collections import Counter
from pathlib import Path

import cv2
import face_recognition
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
FONT_PATH = "arial.ttf"

def capture_faces(name: str, save_dir: Path = Path("training")):
    person_dir = save_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    existing_images = list(person_dir.glob(f"{name}_*.jpg"))
    existing_count = len(existing_images)

    cap = cv2.VideoCapture(0)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            new_index = existing_count + count + 1
            filename = f"{name}_{new_index:02d}.jpg"
            cv2.imwrite(str(person_dir / filename), frame)
            print(f"luu anh : {filename}")
            count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []
    encoded_files = set()

    if encodings_location.exists():
        with encodings_location.open("rb") as f:
            existing = pickle.load(f)
            names = existing.get("names", [])
            encodings = existing.get("encodings", [])
            encoded_files = set(existing.get("files", []))

    new_files = []
    total_skipped = 0

    for filepath in Path("training").glob("*/*.jpg"):
        filepath_str = str(filepath.resolve())
        if filepath_str in encoded_files:
            continue

        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            distances = face_recognition.face_distance(encodings, encoding)
            if any(d < 0.45 for d in distances):  
                total_skipped += 1
                continue

            names.append(name)
            encodings.append(encoding)
            new_files.append(filepath_str)

        print(f"Đã encode ảnh mới: {filepath_str}")

    encoded_files.update(new_files)

    name_encodings = {
        "names": names,
        "encodings": encodings,
        "files": list(encoded_files),
    }

    encodings_location.parent.mkdir(parents=True, exist_ok=True)
    with encodings_location.open("wb") as f:
        pickle.dump(name_encodings, f)

    print(f"Đã lưu {len(new_files)} ảnh mới (bỏ qua {total_skipped} trùng lặp). Tổng: {len(encodings)} ảnh.")


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=0.6
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box

    try:
        font = ImageFont.truetype(FONT_PATH, size=25)
    except OSError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((left, top), name, font=font)
    text_left, text_top, text_right, text_bottom = text_bbox

    text_height = text_bottom - text_top
    adjusted_text_top = max(top - text_height - 4, 0)
    text_bbox = (text_left, adjusted_text_top, text_right, adjusted_text_top + text_height)

    draw.rectangle(text_bbox, fill="black")
    draw.text((text_left, adjusted_text_top), name, fill="white", font=font)
    draw.rectangle(((left, top), (right, bottom)), outline="green", width=2)

def recognition_unknown_face(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    if not Path(image_location).exists():
        print(f" File ảnh không tồn tại: {image_location}")
        return

    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()

def recognize_from_webcam(encodings_location: Path = DEFAULT_ENCODINGS_PATH, tolerance: float = 0.6):
    if not encodings_location.exists():
        print("False")
        return

    with encodings_location.open("rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance)
            if True in matches:
                matched_idxs = [i for i, m in enumerate(matches) if m]
                counts = Counter(known_names[i] for i in matched_idxs)
                name = counts.most_common(1)[0][0]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    

def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognition_unknown_face(image_location=str(filepath.absolute()), model=model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition CLI")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--realtime_v2", action="store_true")
    parser.add_argument("-m", choices=["hog", "cnn"], default="hog")
    parser.add_argument("-f", type=str)

    args = parser.parse_args()

    if args.capture:
        if not args.name:
            print("Cung cấp tên bằng cách dùng --name")
        else:
            capture_faces(name=args.name)
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    
    if args.realtime_v2:
        recognize_from_webcam(encodings_location=DEFAULT_ENCODINGS_PATH, tolerance=0.6)
    