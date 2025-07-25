import os
import xml.etree.ElementTree as ET

# Thư mục chứa file XML
xml_folder = r"C:\Users\dungv\Hugging_Face\YOLO + Facenet\test"
# Thư mục lưu file TXT theo định dạng YOLO
output_folder = r"C:\Users\dungv\Hugging_Face\YOLO + Facenet\labels\test"
os.makedirs(output_folder, exist_ok=True)

# Duyệt tất cả file XML trong thư mục
for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(xml_folder, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Kích thước ảnh
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    txt_lines = []
    for obj in root.findall("object"):
        # Dù là with_mask hay without_mask -> class_id = 0
        class_id = 0

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        # YOLO line: <class_id> <x_center> <y_center> <width> <height>
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        txt_lines.append(line)

    # Tên file TXT giống file XML nhưng đổi đuôi
    txt_filename = os.path.splitext(xml_file)[0] + ".txt"
    txt_path = os.path.join(output_folder, txt_filename)

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))

print("✅ Chuyển đổi xong! Các file TXT đã lưu vào thư mục:", output_folder)
