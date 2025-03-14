#ifndef YOLOV8_DETECTOR_H
#define YOLOV8_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>  // 添加DNN模块
#include <opencv2/imgproc.hpp>  // 添加图像处理模块
#include <opencv2/highgui.hpp>  // 添加GUI模块，用于imshow等

// 结构体定义
struct Detection {
    float x;         // 边界框中心x坐标
    float y;         // 边界框中心y坐标
    float width;     // 边界框宽度
    float height;    // 边界框高度
    int class_id;    // 类别ID
    float confidence;  // 置信度
};

class YoloV8Detector {
public:
    YoloV8Detector(const std::string& model_path);
    ~YoloV8Detector() = default;

    // 修改返回类型为 Detection 数组
    std::vector<Detection> detect(const cv::Mat& image);

private:
    std::string model_path_;
    float confidence_threshold_ = 0.5;  // 添加默认值
    float nms_threshold_ = 0.4;         // 添加默认值
    cv::dnn::Net net;                   // 添加网络模型对象
};

#endif // YOLOV8_DETECTOR_H