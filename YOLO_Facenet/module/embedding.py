import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1


class FaceEmbedder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def preprocess(self, face_img):
        face = cv2.resize(face_img, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) / 0.5
        face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return face

    def get_embedding(self, face_img):
        face_tensor = self.preprocess(face_img)
        with torch.no_grad():
            embedding = self.model(face_tensor).cpu().numpy()[0]
        return embedding / np.linalg.norm(embedding)
