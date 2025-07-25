def recognize_from_webcam_yolo(encodings_location: Path = DEFAULT_ENCODINGS_PATH, tolerance: float = 0.6):
    if not encodings_location.exists():
        print("Không tìm thấy encodings.pkl. Vui lòng train trước.")
        return

    with encodings_location.open("rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n.pt") 

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cropped = frame[y1:y2, x1:x2]
            rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_cropped)

            for (top, right, bottom, left) in face_locations:

                face_image = rgb_cropped[top:bottom, left:right]
                encs = face_recognition.face_encodings(rgb_cropped, [(top, right, bottom, left)])

                if not encs:
                    continue

                name = "Unknown"
                matches = face_recognition.compare_faces(known_encodings, encs[0], tolerance)
                if True in matches:
                    idxs = [i for i, m in enumerate(matches) if m]
                    counts = Counter(known_names[i] for i in idxs)
                    name = counts.most_common(1)[0][0]

                abs_left = x1 + left
                abs_top = y1 + top
                abs_right = x1 + right
                abs_bottom = y1 + bottom

                cv2.rectangle(frame, (abs_left, abs_top), (abs_right, abs_bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (abs_left, abs_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Webcam Face Recognition with YOLO refined boxes", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    



