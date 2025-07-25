import os
import shutil

# Thư mục ảnh gốc
SOURCE_IMAGE_DIR = r'C:\Users\dungv\Hugging_Face\YOLO + Facenet\test'

# Thư mục ảnh đích (YOLOv8 structure)
DEST_IMAGE_DIR = r'C:\Users\dungv\Hugging_Face\YOLO + Facenet\images\test'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(DEST_IMAGE_DIR, exist_ok=True)

# Lặp qua tất cả các file ảnh trong thư mục gốc
for filename in os.listdir(SOURCE_IMAGE_DIR):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        src_path = os.path.join(SOURCE_IMAGE_DIR, filename)
        dst_path = os.path.join(DEST_IMAGE_DIR, filename)
        shutil.copy(src_path, dst_path)

print("✅ Đã copy toàn bộ ảnh sang thư mục 'yolo_dataset/images/train'.")
