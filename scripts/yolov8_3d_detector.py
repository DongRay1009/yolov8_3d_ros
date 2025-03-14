#!/usr/bin/env python

import rospy
from yolov8_3d.srv import Detect3D
from yolov8_3d.msg import DetectionResult
# ⚠️ 需要调整: 确保导入路径正确，可能需要修改为相对导入或绝对路径
from yolov8_detector import YoloV8Detector

class YoloV83DDetectorNode:
    def __init__(self):
        rospy.init_node('yolov8_3d_detector', anonymous=True)
        # ⚠️ 需要调整: 根据您的设备添加初始化参数，如模型路径、设备选择(CPU/GPU)、置信度阈值等
        self.detector = YoloV8Detector()
        self.service = rospy.Service('detect_3d', Detect3D, self.handle_detection)
        rospy.loginfo("YOLOv8 3D Detector Node Initialized")

    def handle_detection(self, req):
        image = req.image
        # ⚠️ 需要调整: 可能需要从ROS图像消息转换为OpenCV格式
        results = self.detector.detect(image)
        response = DetectionResult()
        response.detections = results
        return response

if __name__ == '__main__':
    try:
        detector_node = YoloV83DDetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass