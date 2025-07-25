import pickle
from pathlib import Path

embedding_path = Path("database.pkl")

with embedding_path.open("rb") as f:
    data = pickle.load(f)

print("Tổng số khuôn mặt đã mã hóa:", len(data["embedding"]))
print("Danh sách tên:", set(data["names"]))

for i, (name, encoding) in enumerate(zip(data["names"], data["embedding"])):
    print(f"[{i}] Tên: {name}, vector mã hóa (rút gọn): {encoding[:5]}...")
