// Interface for fall angle evaluation.
// Declares FallAngleResult and evaluate_fall_angles(), which inspects 17 YOLO-pose
// keypoints (shoulders, hips, knees, ankles) to determine if a person has fallen
// based on configurable torso and leg angle thresholds.

#pragma once

#include <vector>

namespace app::core::services {
// Parity with fall-detection:
// - helpers.calculate_body_angles: all 8 joints must pass confidence / finite / |coord|<=1e6 before
// angles exist
// - fall._is_fall: torso > torso_threshold AND (left_leg > leg_threshold OR right_leg >
// leg_threshold) Angles use the same vector math as Python: angle between (segment vector) and
// vertical (0, -1), acute 0–90°.

struct FallAngleResult {
    bool is_fall = false;
    float left_leg_deg = 0.f;
    float right_leg_deg = 0.f;
    float torso_deg = 0.f;
};

FallAngleResult evaluate_fall_keypoints_python(const std::vector<std::vector<float>>& kpts,
                                               float torso_threshold_deg, float leg_threshold_deg,
                                               float keypoints_conf_threshold);

bool is_fall_from_keypoints_python(const std::vector<std::vector<float>>& kpts,
                                   float torso_threshold_deg, float leg_threshold_deg,
                                   float keypoints_conf_threshold);

}  // namespace app::core::services
