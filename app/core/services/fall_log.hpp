#pragma once

#include <iomanip>
#include <sstream>
#include <string>

namespace app::core::services {

// Matches Python: fall.py → logger.info("Fall detected", extra={...})
// Since C++ Logger only accepts a string, we inline the same fields as key=value.
inline std::string format_fall_detected_log(float left_leg_deg, float right_leg_deg, float torso_deg) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    // Keep key names identical to Python: torso_angle, left_leg_angle, right_leg_angle
    oss << "Fall detected"
        << " | torso_angle=" << torso_deg << "°"
        << " | left_leg_angle=" << left_leg_deg << "°"
        << " | right_leg_angle=" << right_leg_deg << "°";
    return oss.str();
}

// Same key names as Python logs; use at DEBUG while tuning thresholds (no "Fall detected" prefix).
inline std::string format_pose_angles_log(float left_leg_deg, float right_leg_deg, float torso_deg) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "left_leg_angle=" << left_leg_deg << "° | right_leg_angle=" << right_leg_deg
        << "° | torso_angle=" << torso_deg << "°";
    return oss.str();
}

}  // namespace app::core::services
