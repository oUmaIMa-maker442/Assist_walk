# test_yolo_direct.py
import sys
sys.path.insert(0, 'assistwalk_vision')
import cv2
from src.step3_yolo_detection import YOLODetector

yolo = YOLODetector(model_name='yolov8n.pt')

# Capture directe sans preprocessing
cap = cv2.VideoCapture("http://10.1.151.79:8080/video")
for _ in range(5):
    cap.read()
ret, frame = cap.read()
cap.release()

# Test YOLO sur image originale
results = yolo.detect(frame)
print("Objets détectés :", results)