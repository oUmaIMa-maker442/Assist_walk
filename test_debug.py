# test_debug.py
import cv2
import sys
import time
sys.path.insert(0, 'text_module')
sys.path.insert(0, 'assistwalk_vision')

from src.step3_yolo_detection import YOLODetector
from distance_estimator import estimate_distance

yolo = YOLODetector(model_name='yolov8n.pt')

cap = cv2.VideoCapture("http://10.1.151.79:8080/video")
time.sleep(2)
for _ in range(10):
    ret, frame = cap.read()
cap.release()

# ✅ Rotation
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("debug_frame.jpg", frame)
print(f"Frame shape après rotation : {frame.shape}")

results = yolo.detect(frame)
print(f"YOLO détecte : {results}")

for obj in results:
    bbox = [int(b) for b in obj['bbox']]
    dist_text, level = estimate_distance(bbox, frame.shape[1], frame.shape[0])
    print(f"→ {obj['class']} | dist={dist_text}")