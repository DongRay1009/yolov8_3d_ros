#!/usr/bin/env python3
# filepath: d:\GitHub repo\yolov8_3d_ros\scripts\detector_node.py

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO  # Import ultralytics package to use PyTorch model

class YoloV8Detector:
    def __init__(self, model_path):
        """Initialize YOLOv8 detector with PyTorch model"""
        try:
            self.model = YOLO(model_path)  # Load PyTorch model
            self.confidence_threshold = 0.25
            rospy.loginfo(f"Successfully loaded YOLOv8 PyTorch model: {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            raise e
        
    def detect(self, image):
        """Perform object detection using YOLOv8 PyTorch model"""
        # Use ultralytics API for prediction
        results = self.model(image, verbose=False)
        
        # Get detection results
        detections = []
        
        for result in results:
            # If objects are detected
            if result.boxes is not None and len(result.boxes) > 0:
                # Get bounding boxes, confidence scores and classes
                boxes = result.boxes.xyxy.cpu().numpy()    # Get bounding boxes in (x1,y1,x2,y2) format
                confs = result.boxes.conf.cpu().numpy()    # Get confidence scores
                clss = result.boxes.cls.cpu().numpy()      # Get class indices
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                    # Filter low confidence detections
                    if conf < self.confidence_threshold:
                        continue
                        
                    # Get bounding box coordinates (top-left and bottom-right)
                    x1, y1, x2, y2 = box
                    
                    # Calculate center point and dimensions
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Log detection information
                    rospy.loginfo(f"Detected object: center=({center_x},{center_y}), " 
                                 f"dimensions=({width},{height}), class={int(cls)}, confidence={conf:.3f}")
                    
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
        
        # COCO dataset class names
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
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Perform detection
            detections = self.detector.detect(cv_image)
            
            # Copy image before displaying results to avoid modifying original data
            display_image = cv_image.copy()
            
            rospy.loginfo(f"Detected {len(detections)} objects")
            
            # Draw detection results on the image
            for det in detections:
                x, y = det['x'], det['y']
                w, h = det['width'], det['height']
                class_id = det['class_id']
                conf = det['confidence']
                
                # Get class name
                class_name = self.coco_names[class_id] if class_id < len(self.coco_names) else f"class{class_id}"
                
                # Draw bounding box on the image
                left = int(x - w/2)
                top = int(y - h/2)
                right = int(x + w/2)
                bottom = int(y + h/2)
                
                # Ensure coordinates are within image bounds
                left = max(0, left)
                top = max(0, top)
                right = min(display_image.shape[1]-1, right)
                bottom = min(display_image.shape[0]-1, bottom)
                
                # Draw bounding box and label
                cv2.rectangle(display_image, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add class label and confidence
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(display_image, label, (left, top-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            # Display image
            cv2.imshow("YOLOv8 Detections", display_image)
            cv2.waitKey(1)
                
            # Publish results
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