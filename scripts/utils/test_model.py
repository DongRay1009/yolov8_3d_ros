#!/usr/bin/env python3
import cv2
print(f"OpenCV version: {cv2.__version__}")
try:
    net = cv2.dnn.readNetFromONNX("/home/msi/yolo_3d_ws/src/yolov8_3d/models/yolov8m.onnx")
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {e}")