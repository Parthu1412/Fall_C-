#include "fall_eval.hpp"
#include "../helpers.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <tuple>

namespace app::core::services {

namespace {

constexpr int kIdxLShoulder = 5;
constexpr int kIdxRShoulder = 6;
constexpr int kIdxLHip = 11;
constexpr int kIdxRHip = 12;
constexpr int kIdxLKnee = 13;
constexpr int kIdxRKnee = 14;
constexpr int kIdxLAnkle = 15;
constexpr int kIdxRAnkle = 16;

constexpr int kRequiredIndices[] = {
    kIdxLShoulder, kIdxRShoulder, kIdxLHip, kIdxRHip, kIdxLKnee, kIdxRKnee, kIdxLAnkle, kIdxRAnkle,
};

bool joint_ok_for_angles(const std::vector<std::vector<float>>& kpts,
                         int idx,
                         float keypoints_conf_threshold) {
    if (idx < 0 || static_cast<size_t>(idx) >= kpts.size() || kpts[idx].size() < 3)
        return false;
    float x = kpts[idx][0];
    float y = kpts[idx][1];
    float c = kpts[idx][2];
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(c))
        return false;
    if (std::fabs(x) > 1e6f || std::fabs(y) > 1e6f)
        return false;
    return c >= keypoints_conf_threshold;
}

bool all_eight_keypoints_ok(const std::vector<std::vector<float>>& kpts, float conf_thresh) {
    for (int idx : kRequiredIndices) {
        if (!joint_ok_for_angles(kpts, idx, conf_thresh))
            return false;
    }
    return true;
}

std::optional<std::tuple<float, float, float>> compute_angles_deg(
    const std::vector<std::vector<float>>& kpts,
    float keypoints_conf_threshold)
{
    if (kpts.size() < 17 || !all_eight_keypoints_ok(kpts, keypoints_conf_threshold))
        return std::nullopt;

    cv::Point2f shoulder_mid(
        (kpts[kIdxLShoulder][0] + kpts[kIdxRShoulder][0]) * 0.5f,
        (kpts[kIdxLShoulder][1] + kpts[kIdxRShoulder][1]) * 0.5f);
    cv::Point2f hip_mid(
        (kpts[kIdxLHip][0] + kpts[kIdxRHip][0]) * 0.5f,
        (kpts[kIdxLHip][1] + kpts[kIdxRHip][1]) * 0.5f);

    cv::Point2f torso_vec = shoulder_mid - hip_mid;
    cv::Point2f left_leg_vec(
        kpts[kIdxLAnkle][0] - kpts[kIdxLKnee][0],
        kpts[kIdxLAnkle][1] - kpts[kIdxLKnee][1]);
    cv::Point2f right_leg_vec(
        kpts[kIdxRAnkle][0] - kpts[kIdxRKnee][0],
        kpts[kIdxRAnkle][1] - kpts[kIdxRKnee][1]);

    const cv::Point2f vertical(0.f, -1.f);

    auto torso_angle = app::core::calculate_angle(torso_vec, vertical);
    auto left_leg_angle = app::core::calculate_angle(left_leg_vec, vertical);
    auto right_leg_angle = app::core::calculate_angle(right_leg_vec, vertical);

    if (!torso_angle.has_value() || !left_leg_angle.has_value() || !right_leg_angle.has_value())
        return std::nullopt;

    return std::tuple(*torso_angle, *left_leg_angle, *right_leg_angle);
}

}  // namespace

FallAngleResult evaluate_fall_keypoints_python(
    const std::vector<std::vector<float>>& kpts,
    float torso_threshold_deg,
    float leg_threshold_deg,
    float keypoints_conf_threshold)
{
    FallAngleResult r{};
    auto ang = compute_angles_deg(kpts, keypoints_conf_threshold);
    if (!ang.has_value())
        return r;
    r.torso_deg = std::get<0>(*ang);
    r.left_leg_deg = std::get<1>(*ang);
    r.right_leg_deg = std::get<2>(*ang);
    bool torso_fallen = (r.torso_deg > torso_threshold_deg);
    bool legs_fallen = (r.left_leg_deg > leg_threshold_deg || r.right_leg_deg > leg_threshold_deg);
    r.is_fall = torso_fallen && legs_fallen;
    return r;
}

bool is_fall_from_keypoints_python(
    const std::vector<std::vector<float>>& kpts,
    float torso_threshold_deg,
    float leg_threshold_deg,
    float keypoints_conf_threshold)
{
    return evaluate_fall_keypoints_python(kpts, torso_threshold_deg, leg_threshold_deg, keypoints_conf_threshold)
        .is_fall;
}

}  // namespace app::core::services
