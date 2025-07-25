import os

# Đường dẫn đến thư mục chứa các file cần đổi tên
folder_path = 'C:/Users/dungv/Hugging_Face/MTCNN_FACENET/train/Elon Musk' 


prefix = 'Elon Musk'  # 👉 Tiền tố tên file mong muốn

# Lấy và sắp xếp danh sách file (theo tên)
files = sorted(os.listdir(folder_path))

# Duyệt và đổi tên file
for index, filename in enumerate(files[:100]):
    # Lấy phần mở rộng (vd: .jpg, .png)
    ext = os.path.splitext(filename)[1]
    # Tạo tên mới: tenfile_1.jpg, tenfile_2.jpg, ...
    new_name = f"{prefix}_{index + 1}{ext}"
    # Tạo đường dẫn đầy đủ
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)

print("✅ Đổi tên xong.")
