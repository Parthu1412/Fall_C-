// Interface for the fall detection service.
// Declares FallDetectionService, which accepts a raw frame, runs YOLO-pose
// inference, evaluates body angles, and returns a fall verdict with annotated
// frame and keypoints for downstream buffering and video clip assembly.

#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

namespace app::core::inferences {
class PoseInference;
}

namespace app::core::services {

class FallDetectionService {
public:
    explicit FallDetectionService(bool annotate_image = true);
    ~FallDetectionService();

    /** @param redis_style_detections Optional poses from Redis (per person: 17×[x,y,conf]). */
    std::pair<bool, cv::Mat> detect_fall(const cv::Mat& frame,
                                         const std::vector<std::vector<std::vector<float>>>& redis_style_detections);

private:
    bool annotate_image_ = true;
    std::unique_ptr<app::core::inferences::PoseInference> pose_;
};

}  // namespace app::core::services
