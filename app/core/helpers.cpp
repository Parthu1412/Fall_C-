// Core helper utilities — shared geometric and keypoint helper functions used
// by fall evaluation and skeleton annotation. Includes body angle calculation
// (torso and leg segment vectors vs. vertical axis) and keypoint visibility
// validation, matching the Python core/helpers.py logic exactly.

#include "helpers.hpp"

#include <cmath>
#include <iostream>

#include "../config.hpp"  // For the confidence threshold singleton

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace app {
namespace core {

std::optional<float> calculate_angle(const cv::Point2f& vector1, const cv::Point2f& vector2)
{
    // OPTIMIZATION: Raw math is much faster than cv::norm() for 2D points
    float norm1 = std::sqrt(vector1.x * vector1.x + vector1.y * vector1.y);
    float norm2 = std::sqrt(vector2.x * vector2.x + vector2.y * vector2.y);

    if (norm1 < 1e-6f || norm2 < 1e-6f)
    {
        return std::nullopt;
    }

    // OPTIMIZATION: Calculate dot product directly without creating intermediate unit vectors
    float dot_product = (vector1.x * vector2.x + vector1.y * vector2.y) / (norm1 * norm2);
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));

    float angle_rad = std::acos(dot_product);
    float angle_deg = angle_rad * (180.0f / M_PI);

    return (angle_deg > 90.0f) ? (180.0f - angle_deg) : angle_deg;
}

bool is_valid_coordinate(int x, int y, int frame_width, int frame_height)
{
    if (x < 0 || x >= frame_width || y < 0 || y >= frame_height)
    {
        return false;
    }
    if (x == 0 && y == 0)
    {
        return false;
    }

    const int corner_tolerance = 1;
    bool is_at_corner =
        (x < corner_tolerance && y < corner_tolerance) ||
        (x >= frame_width - corner_tolerance && y >= frame_height - corner_tolerance) ||
        (x < corner_tolerance && y >= frame_height - corner_tolerance) ||
        (x >= frame_width - corner_tolerance && y < corner_tolerance);
    if (is_at_corner)
    {
        return false;
    }

    if (std::abs(x) > 1000000 || std::abs(y) > 1000000)
    {
        return false;
    }
    return true;
}

void draw_keypoints(cv::Mat& frame, const std::vector<std::vector<float>>& keypoints,
                    int frame_width, int frame_height)
{
    if (keypoints.empty())
        return;

    // Static vector is good, it initializes only once
    static const std::vector<std::pair<int, int>> connections = {
        {5, 6}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}};

    // OPTIMIZATION: Stack-allocated array instead of Heap-allocated std::set
    // YOLOv8 pose models output exactly 17 keypoints (indices 0 to 16)
    bool drawn_points[17] = {false};

    // OPTIMIZATION: Fetch the threshold once, outside the loop
    static float conf_threshold = app::config::AppConfig::getInstance().keypoints_conf_threshold;

    for (const auto& conn : connections)
    {
        int start = conn.first;
        int end = conn.second;

        // Safety check to prevent array out-of-bounds just in case
        if (start >= 17 || end >= 17 || start >= static_cast<int>(keypoints.size()) ||
            end >= static_cast<int>(keypoints.size()))
        {
            continue;
        }

        if (keypoints[start][2] < conf_threshold || keypoints[end][2] < conf_threshold)
        {
            continue;
        }

        int x1 = static_cast<int>(keypoints[start][0]);
        int y1 = static_cast<int>(keypoints[start][1]);
        int x2 = static_cast<int>(keypoints[end][0]);
        int y2 = static_cast<int>(keypoints[end][1]);

        if (!std::isfinite(keypoints[start][0]) || !std::isfinite(keypoints[start][1]) ||
            !std::isfinite(keypoints[end][0]) || !std::isfinite(keypoints[end][1]))
        {
            continue;
        }

        if (!is_valid_coordinate(x1, y1, frame_width, frame_height) ||
            !is_valid_coordinate(x2, y2, frame_width, frame_height))
        {
            continue;
        }

        // Fast array lookup instead of std::set::find()
        if (!drawn_points[start])
        {
            cv::circle(frame, cv::Point(x1, y1), 2, cv::Scalar(0, 255, 0), -1);
            drawn_points[start] = true;
        }
        if (!drawn_points[end])
        {
            cv::circle(frame, cv::Point(x2, y2), 2, cv::Scalar(0, 255, 0), -1);
            drawn_points[end] = true;
        }

        cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255), 1);
    }
}

}  // namespace core
}  // namespace app