import atexit
import csv
import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import torch

from module.camera import Camera
from module.database import FaceDatabase
from module.detector import FaceDetector
from module.embedding import FaceEmbedder
from module.livenessnet import LivenessNet
from module.processing_images import load_faces_from_folder
from module.recognition import FaceRecognizer

st.set_page_config(page_title="Chấm công khuôn mặt", layout="centered")
st.title("Chấm Công Khuôn Mặt")
device = "cuda" if torch.cuda.is_available() else "cpu"

database = FaceDatabase("database/database_yolo.pkl")
detector = FaceDetector(device=device)
embedder = FaceEmbedder(device=device)
recognizer = FaceRecognizer(database.get_all())
liveness = LivenessNet(32, 32, 3, 2).to(device)
liveness.load_state_dict(torch.load("best_model.pth", map_location=device)["model_state_dict"])
liveness.eval()
camera = Camera()
frame_win = st.image([])

main_page = st.Page("main_page.py", title="Chấm công khuôn mặt")
page_2 = st.Page("page_2.py",title = "STATUS")
page_3 = st.page("page_3.py",title = "Cập nhập dữ liệu")
pg = st.navigation([main_page,page_2,page_3])
pg.run()




if 'new_name' not in st.session_state:
    st.session_state.new_name = ""
if 'add_face' not in st.session_state:
    st.session_state.add_face = False
if 'last_embedding' not in st.session_state:
    st.session_state.last_embedding = None

def preprocess_for_liveness(face_img):
    face = cv2.resize(face_img, (32, 32))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.tensor(face).permute(2, 0, 1).unsqueeze(0)

def log_checkin(name):
    os.makedirs("logs", exist_ok=True)
    with open("logs/checkin_log.csv", "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name])

def start_add():
    st.session_state.add_face = True

def checkin_action():
    frame = camera.get_frame()
    if frame is None:
        st.error("Không lấy được ảnh từ camera")
        return

    frame_win.image(frame[:, :, ::-1])
    boxes = detector.detect(frame)
    if not boxes:
        st.warning("Không tìm thấy khuôn mặt")
        return

    face_img = detector.extract_faces(frame, boxes[:1])[0]
    if face_img is None:
        st.warning("Không trích xuất được khuôn mặt")
        return

    tensor = preprocess_for_liveness(face_img).to(device)
    with torch.no_grad():
        is_real = torch.argmax(liveness(tensor), dim=1).item()

    st.image(face_img[:, :, ::-1], width=180)  
    if is_real == 0:
        st.error("Phát hiện khuôn mặt giả mạo!")
        return

    embedding = embedder.get_embedding(face_img)
    name, dist = recognizer.recognize(embedding)

    if name == "Unknown":
        st.warning("Khuôn mặt chưa có trong hệ thống")
        st.session_state.last_embedding = embedding
        st.button("Đăng ký người mới", on_click=start_add)
    else:
        st.success(f"{name} đã chấm công lúc {datetime.now().strftime('%H:%M:%S')}")
        log_checkin(name)

if st.button(" Chấm công"):
    checkin_action()

if st.session_state.add_face:
    st.session_state.new_name = st.text_input("Nhập tên người mới:", value=st.session_state.new_name)
    if st.session_state.new_name:
        if st.button("Xác nhận đăng ký"):
            if st.session_state.last_embedding is not None:
                database.add_face(st.session_state.new_name, st.session_state.last_embedding)
                database.save_database()
                recognizer = FaceRecognizer(database.get_all())
                st.success(f"Đã đăng ký và chấm công {st.session_state.new_name}")
                log_checkin(st.session_state.new_name)
                st.session_state.add_face = False
                st.session_state.new_name = ""
                st.session_state.last_embedding = None
            else:
                st.error("Không có dữ liệu embedding. Vui lòng chấm công lại trước khi đăng ký.")

if st.button("Cập nhật DB từ thư mục ảnh"):
    load_faces_from_folder("train", detector, embedder, database, device=device)
    st.success("Đã cập nhật database từ ảnh")

atexit.register(lambda: camera.release())
