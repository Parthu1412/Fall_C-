#include "fall.hpp"
#include "../inferences/pose.hpp"
#include "../helpers.hpp"
#include "fall_eval.hpp"
#include "fall_log.hpp"
#include "../../config.hpp"
#include "../../utils/logger.hpp"
#include <memory>

namespace app::core::services {

FallDetectionService::FallDetectionService(bool annotate_image)
    : annotate_image_(annotate_image)
{
    auto& cfg = app::config::AppConfig::getInstance();
    pose_ = std::make_unique<app::core::inferences::PoseInference>(cfg.model_path);
}

FallDetectionService::~FallDetectionService() = default;

std::pair<bool, cv::Mat> FallDetectionService::detect_fall(
    const cv::Mat& frame,
    const std::vector<std::vector<std::vector<float>>>& redis_style_detections)
{
    if (frame.empty()) {
        app::utils::Logger::error("[FallDetectionService] Invalid input frame");
        return {false, cv::Mat()};
    }

    auto& cfg = app::config::AppConfig::getInstance();
    try {
        std::vector<std::vector<std::vector<float>>> keypoints_list =
            pose_->detect(frame, redis_style_detections);

        cv::Mat output_frame = frame.clone();
        bool fall_detected = false;
        const float torso_th = static_cast<float>(cfg.torso_angle);
        const float leg_th = static_cast<float>(cfg.leg_angle);
        const float conf_th = cfg.keypoints_conf_threshold;

        for (const auto& keypoints : keypoints_list) {
            auto eval = evaluate_fall_keypoints_python(keypoints, torso_th, leg_th, conf_th);
            if (!eval.is_fall) continue;

            fall_detected = true;
            if (annotate_image_) {
                app::core::draw_keypoints(output_frame, keypoints, frame.cols, frame.rows);
            }
            app::utils::Logger::info(format_fall_detected_log(
                eval.left_leg_deg, eval.right_leg_deg, eval.torso_deg));
        }

        return {fall_detected, output_frame};
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[FallDetectionService] Error: ") + e.what());
        return {false, cv::Mat()};
    }
}

}  // namespace app::core::services
