// Interface for core helper utilities.
// Declares geometric helpers for body angle computation and keypoint
// coordinate validation shared across fall_eval and skeleton drawing code.

#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

namespace app {
namespace core {

// Calculates the acute angle (0-90 degrees) between two 2D vectors
std::optional<float> calculate_angle(const cv::Point2f& vector1, const cv::Point2f& vector2);

// Validates if a coordinate falls within the frame bounds
bool is_valid_coordinate(int x, int y, int frame_width, int frame_height);

// Draws skeletal connections and keypoints on the OpenCV Mat frame
// keypoints expects a vector of [x, y, confidence] arrays
void draw_keypoints(cv::Mat& frame, const std::vector<std::vector<float>>& keypoints,
                    int frame_width, int frame_height);

}  // namespace core
}  // namespace app