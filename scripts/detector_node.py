#!/usr/bin/env python3
# filepath: /home/msi/yolo_3d_ws/src/yolov8_3d/scripts/detector_node.py

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO  # 导入ultralytics包来使用PyTorch模型

class YoloV8Detector:
    def __init__(self, model_path):
        """初始化YOLOv8检测器使用PyTorch模型"""
        try:
            self.model = YOLO(model_path)  # 加载PyTorch模型
            self.confidence_threshold = 0.25
            rospy.loginfo(f"成功加载YOLOv8 PyTorch模型: {model_path}")
        except Exception as e:
            rospy.logerr(f"加载模型失败: {e}")
            raise e
        
    def detect(self, image):
        """使用YOLOv8 PyTorch模型进行目标检测"""
        # 使用ultralytics API预测
        results = self.model(image, verbose=False)
        
        # 获取检测结果
        detections = []
        
        for result in results:
            # 如果有检测到的目标
            if result.boxes is not None and len(result.boxes) > 0:
                # 获取边界框、置信度和类别
                boxes = result.boxes.xyxy.cpu().numpy()    # 以(x1,y1,x2,y2)格式获取边界框
                confs = result.boxes.conf.cpu().numpy()    # 获取置信度
                clss = result.boxes.cls.cpu().numpy()      # 获取类别索引
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                    # 过滤低置信度检测
                    if conf < self.confidence_threshold:
                        continue
                        
                    # 获取边界框坐标(左上和右下)
                    x1, y1, x2, y2 = box
                    
                    # 计算中心点和宽高
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # 记录检测信息
                    rospy.loginfo(f"检测到物体: 中心点=({center_x},{center_y}), " 
                                 f"宽高=({width},{height}), 类别={int(cls)}, 置信度={conf:.3f}")
                    
                    detections.append({
                        'x': center_x,
                        'y': center_y,
                        'width': width,
                        'height': height,
                        'class_id': int(cls),
                        'confidence': float(conf)
                    })
            
        return detections

class DetectorNode:
    def __init__(self):
        rospy.init_node('detector_node_py')
        model_path = rospy.get_param('~model_path', '/home/msi/yolo_3d_ws/src/yolov8_3d/models/yolov8m.pt')
        input_topic = rospy.get_param('~input_topic', '/rgb/image_raw')
        
        self.bridge = CvBridge()
        self.detector = YoloV8Detector(model_path)
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        self.detection_pub = rospy.Publisher('detection_image', Image, queue_size=1)
        
        # COCO数据集类别名称
        self.coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                     "hair drier", "toothbrush"]
        
        rospy.loginfo("YOLOv8 Detector Node Initialized")
        
    def image_callback(self, msg):
        try:
            # 转换ROS图像到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 执行检测
            detections = self.detector.detect(cv_image)
            
            # 显示结果前复制图像以防止修改原始数据
            display_image = cv_image.copy()
            
            rospy.loginfo(f"检测到 {len(detections)} 个物体")
            
            # 在图像上绘制检测结果
            for det in detections:
                x, y = det['x'], det['y']
                w, h = det['width'], det['height']
                class_id = det['class_id']
                conf = det['confidence']
                
                # 获取类别名称
                class_name = self.coco_names[class_id] if class_id < len(self.coco_names) else f"类别{class_id}"
                
                # 在图像上绘制边界框
                left = int(x - w/2)
                top = int(y - h/2)
                right = int(x + w/2)
                bottom = int(y + h/2)
                
                # 确保坐标在图像范围内
                left = max(0, left)
                top = max(0, top)
                right = min(display_image.shape[1]-1, right)
                bottom = min(display_image.shape[0]-1, bottom)
                
                # 绘制边界框和标签
                cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # 添加类别标签和置信度
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(display_image, label, (left, top-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            # 显示图像
            cv2.imshow("YOLOv8 Detections", display_image)
            cv2.waitKey(1)
                
            # 发布结果
            self.detection_pub.publish(self.bridge.cv2_to_imgmsg(display_image, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    try:
        node = DetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()