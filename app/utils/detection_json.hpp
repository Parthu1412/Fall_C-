// Detection JSON helpers — serialises and deserialises keypoint data between
// the 3D vector format used internally (persons × keypoints × [x,y,vis]) and
// the nlohmann::json representation carried in ZMQ inter-process messages.

#pragma once

#include <nlohmann/json.hpp>
#include <vector>

namespace app::utils {

inline void parse_detections_keypoints(const nlohmann::json& j,
                                       std::vector<std::vector<std::vector<float>>>& out)
{
    out.clear();
    if (!j.is_object() || !j.contains("detections") || !j["detections"].is_array())
        return;
    for (const auto& det : j["detections"])
    {
        if (!det.contains("keypoints"))
            continue;
        std::vector<std::vector<float>> person;
        for (const auto& row : det["keypoints"])
        {
            std::vector<float> v;
            for (const auto& x : row)
            {
                if (x.is_number())
                    v.push_back(static_cast<float>(x.get<double>()));
            }
            person.push_back(v);
        }
        if (person.size() >= 17)
            out.push_back(std::move(person));
    }
}

}  // namespace app::utils
