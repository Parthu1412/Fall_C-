#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace app {
namespace utils {

struct FallMessage {
    int store_id;
    int moksa_camera_id;
    std::string s3_uri;
    std::string video_uri;
    std::string trace_id;
    std::string timestamp;

    // This single line generates the equivalent of Python's to_dict() and json parsing
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(FallMessage, 
                                   store_id, 
                                   moksa_camera_id, 
                                   s3_uri, 
                                   video_uri, 
                                   trace_id, 
                                   timestamp)
};

} // namespace utils
} // namespace app