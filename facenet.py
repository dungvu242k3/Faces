import argparse
import os
import pickle
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True,device=device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
while cap.isOpened() :
    ret,frame = cap.read()
    if ret :
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None :
            for box in boxes :
                bbox = list(map(int,box.tolist()))
                frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    cv2.imshow("face_detective", frame)
    if cv2.waitKey(1)&0xFF == 27 :
        break
cap.release()
cv2.destroyAllWindows()
    
                