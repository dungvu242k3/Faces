    # Face_Recognition
🧠 Face Recognition System - Hệ thống nhận diện khuôn mặt thời gian thực
📌 Giới thiệu dự án
Trong thời đại công nghệ 4.0, trí tuệ nhân tạo (AI) ngày càng đóng vai trò quan trọng trong việc tự động hóa và nâng cao hiệu quả các hệ thống an ninh, giám sát và quản lý. Dự án này được xây dựng nhằm phát triển một hệ thống nhận diện khuôn mặt thông minh, hoạt động theo thời gian thực qua webcam, ứng dụng công nghệ thị giác máy tính hiện đại như OpenCV, YOLOv8 và thư viện face_recognition.

Hệ thống có thể làm việc với cơ sở dữ liệu gồm hàng triệu ảnh khuôn mặt của nhiều người khác nhau được lưu trữ trong thư mục training. Đồng thời, người dùng có thể dễ dàng thêm ảnh khuôn mặt mới thông qua webcam, lưu ảnh và mã hóa để cập nhật vào cơ sở dữ liệu nhận diện. Khi chạy nhận diện, hệ thống sẽ phát hiện các khuôn mặt trong video webcam, vẽ bounding box và hiển thị tên người được nhận diện chính xác.

🧠 Mục tiêu và tính năng chính
Phát hiện và nhận diện khuôn mặt thời gian thực: Hệ thống mở webcam, quét và nhận diện khuôn mặt người dùng, vẽ khung viền (bounding box) cùng tên người trên màn hình.
Cơ sở dữ liệu khuôn mặt quy mô lớn: Xử lý và lưu trữ embedding từ hàng triệu ảnh khuôn mặt trong thư mục training/.
Thêm dữ liệu mới qua webcam: Người dùng nhập tên, hệ thống tự động chụp ảnh, lưu ảnh và cập nhật embedding.
Kiểm thử với ảnh tĩnh: Dùng ảnh trong thư mục validation/ để kiểm tra và đánh giá độ chính xác mô hình.
Hiệu suất cao, dễ sử dụng: Hỗ trợ nhận diện đa khuôn mặt cùng lúc với tốc độ mượt mà.
🗃️ Cấu trúc thư mục dự án
face_recognition_project/
├── output/                 # Lưu embedding các khuôn mặt đã mã hóa
├── training/               # Chứa ảnh gốc của hàng triệu người (ảnh training)
├── validation/             # Ảnh dùng để test, đánh giá mô hình
├── detector.py             # Module phát hiện khuôn mặt (YOLOv8 hoặc tương tự)
├── requirements.txt        # Thư viện cần thiết


---

## ⚙️ Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường

```bash
git clone https://github.com/dungvu242k3/Face_Recognition.git
cd Face_Recognition

# Tạo môi trường ảo (tuỳ chọn)
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
2. Tạo database embedding từ ảnh trong thư mục training
python detectors.py --train -m ["hog" or "cnn"]
Script này sẽ duyệt tất cả ảnh trong training/, tạo embedding khuôn mặt và lưu vào thư mục output/ để phục vụ việc nhận diện.

3. Thêm khuôn mặt mới qua webcam
python detector.py --capture --name "Nguyen Van A"
Hệ thống mở webcam, tự động chụp ảnh khuôn mặt người dùng.
Ảnh sẽ được lưu và embedding được cập nhật trong output/.
Khuôn mặt mới được bổ sung vào cơ sở dữ liệu để nhận diện chính xác sau này.

4. Nhận diện khuôn mặt thời gian thực qua webcam
python detector.py --realtime_yolo
Hệ thống phát hiện và nhận diện khuôn mặt trong khung hình webcam.
Vẽ bounding box và hiển thị tên người đã được nhận diện.
Hỗ trợ nhận diện đa khuôn mặt cùng lúc.
5. Kiểm thử mô hình với ảnh validation
Sử dụng ảnh trong thư mục validation/ để test độ chính xác của mô hình nhận diện.

📊 Kết quả và hiệu quả
Hệ thống nhận diện chính xác khuôn mặt trong nhiều điều kiện ánh sáng và góc nhìn khác nhau.
Tốc độ nhận diện nhanh, đạt từ 15–25 FPS tùy cấu hình máy.
Quản lý và mở rộng dữ liệu khuôn mặt dễ dàng thông qua webcam và bộ công cụ đi kèm.
Giao diện thân thiện, phù hợp cho các ứng dụng thực tế như kiểm soát ra vào, điểm danh tự động, xác thực người dùng.

📈 Hướng phát triển
Triển khai dịch vụ nhận diện khuôn mặt qua API RESTful (FastAPI/Flask).
Sử dụng cơ sở dữ liệu vector để tối ưu truy vấn embedding (FAISS, Milvus).
Tối ưu mô hình cho các thiết bị biên (edge devices) như Raspberry Pi, Jetson Nano.
Phát triển giao diện người dùng trực quan (web hoặc desktop).
Mở rộng thêm tính năng phân tích biểu cảm khuôn mặt, cảnh báo an ninh.

so sánh các model :
I.Face Detective:
1.MTCNN : gồm 3 mạng CNN (P,R,O - net) cấu trúc khác nhau vai trò khác nhau, nhẹ,có landmark(tai,mũi,miệng...), phát hiện góc nghiêng occlusion kém
P - net (12x12)
R - net (24x24)
O - net (48x48)
2.retinaface : Accuracy hàng đầu, tốt trong điều kiện khó như góc nghiêng, độ sáng yếu, mặt nhỏ, mặt che; phát hiện landmark 5 điểm + mesh 3D.
chậm trên cpu
3.blazeface :siêu nhẹ tối ưu cho mobile gpu fps cao
acc thấp hơn retinaface phù hợp vs ar và realime k khuyên dùng khi cần acc cao

II.Face recognition
1.Facenet : sử dụng triple loss embedding 128-dim,chọn nếu cần nhẹ  
2.vggface : network lớn (145tr tham số), xử lý full‑frontal faces rất tốt,tốt nếu sử lý ảnh rõ,không phù hợp cho các ứng dụng đòi hỏi độ chính xác rất cao trong các bài toán khó hơn,cấu trúc lớn khiến model có thể nặng và khó triển khai trong các môi trường có tài nguyên hạn chế.
3.arcface : Sử dụng Additive Angular Margin Loss,512 dim embedding(có thể thay đổi),tính linh hoạt cao, hoạt động tốt trên cả khuôn mặt chính diện và không chính diện,nhẹ hơn vggface,cần tài nguyên tính toán khá mạnh để triển khai và huấn luyện.
4.cosface : Cosine Margin Loss (La Loss),đơn giản và hiệu quả, đặc biệt trong các tình huống cần độ chính xác cao mà không đòi hỏi quá nhiều tài nguyên tính toán,dễ triển khai và cho kết quả khá ổn định, 

Model	Accuracy (LFW)	Accuracy (YTF)	Accuracy (MegaFace)	Embedding	Loss Function	
FaceNet	99.63%	95.1%	Không có dữ liệu	128-dim	Triplet Loss	
VGGFace	~98.95%	~97.3%	Không có dữ liệu	Không rõ	Softmax Loss (Cross-entropy)	
ArcFace	99.83%	98.02%	96.98%	512-dim	Additive Angular Margin Loss	Kết quả state-of-the-art, cấu trúc nhẹ hơn VGGFace
CosFace	~99.5%+	Không có dữ liệu	Không có dữ liệu	512-dim	Cosine Margin Loss (La Loss)	







