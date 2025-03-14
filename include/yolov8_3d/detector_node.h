#ifndef YOLOV8_3D_DETECTOR_NODE_H
#define YOLOV8_3D_DETECTOR_NODE_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <yolov8_3d/yolov8_detector.h>

class DetectorNode {
public:
    DetectorNode(ros::NodeHandle& nh);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Publisher detection_pub_;
    YoloV8Detector detector_;
};

#endif // YOLOV8_3D_DETECTOR_NODE_H