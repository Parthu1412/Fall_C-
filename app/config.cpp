#include "config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <mutex>
#include <stdexcept>

namespace app {
namespace config {

namespace {

std::string trim(std::string s) {
    auto not_space = [](int ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

void load_dotenv_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) return;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (!val.empty() && (val.front() == '"' || val.front() == '\'')) {
            char q = val.front();
            if (val.size() >= 2 && val.back() == q)
                val = val.substr(1, val.size() - 2);
        }
        if (::getenv(key.c_str()) == nullptr) {
#ifdef _WIN32
            _putenv_s(key.c_str(), val.c_str());
#else
            ::setenv(key.c_str(), val.c_str(), 0);
#endif
        }
    }
}

int getenv_int(const char* k, int def) {
    const char* v = std::getenv(k);
    if (!v || !*v) return def;
    try {
        return std::stoi(v);
    } catch (...) {
        return def;
    }
}

float getenv_float(const char* k, float def) {
    const char* v = std::getenv(k);
    if (!v || !*v) return def;
    try {
        return std::stof(v);
    } catch (...) {
        return def;
    }
}

std::string getenv_str(const char* k, const std::string& def = {}) {
    const char* v = std::getenv(k);
    return v ? std::string(v) : def;
}

bool getenv_bool(const char* k, bool def) {
    const char* v = std::getenv(k);
    if (!v) return def;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s == "1" || s == "true" || s == "yes";
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

} // namespace

AppConfig& AppConfig::getInstance() {
    static AppConfig inst;
    static std::once_flag once;
    std::call_once(once, [](AppConfig* self) { self->load(); }, &inst);
    return inst;
}

void AppConfig::load() {
    load_dotenv_file(".env");

    total_cameras = getenv_int("TOTAL_CAMERAS", 0);
    num_workers = getenv_int("NUM_WORKERS", 1);
    client_type = getenv_str("CLIENT_TYPE", "rtsp");
    model_path = getenv_str("MODEL_PATH", "yolov8m-pose-fp16.torchscript");
    torso_angle = getenv_int("TORSO_ANGLE", 45);
    leg_angle = getenv_int("LEG_ANGLE", 75);
    keypoints_conf_threshold = getenv_float("KEYPOINTS_CONF_THRESHOLD", 0.7f);
    fps = getenv_int("FPS", 15);
    buffer_size = getenv_int("BUFFER_SIZE", 50);

    video_buffer_size = getenv_int("VIDEO_BUFFER_SIZE", 75);
    frames_before_detection = getenv_int("FRAMES_BEFORE_DETECTION", 30);
    frames_after_detection = getenv_int("FRAMES_AFTER_DETECTION", 45);
    fall_sample_interval = getenv_int("FALL_SAMPLE_INTERVAL", 15);

    redis_host = getenv_str("REDIS_HOST", "localhost");
    redis_port = getenv_int("REDIS_PORT", 6379);
    redis_password = getenv_str("REDIS_PASSWORD", "");
    redis_expiry = getenv_int("REDIS_EXPIRY", 200);
    try {
        longint_max = static_cast<int64_t>(
            std::stoll(getenv_str("LONGINT_MAX", "9223372036854775807")));
    } catch (...) {
        longint_max = 9223372036854775807LL;
    }

    aws_bucket = getenv_str("AWS_BUCKET", "");
    aws_region = getenv_str("AWS_REGION", "");
    aws_object_name = getenv_str("AWS_OBJECT_NAME", "");

    {
        std::string bs = getenv_str("KAFKA_BOOTSTRAP_SERVERS", "");
        kafka_bootstrap_servers = split_csv(bs);
    }
    kafka_client_id = getenv_str("KAFKA_CLIENT_ID", "");
    kafka_topic = getenv_str("KAFKA_TOPIC", "");
    kafka_aws_region = getenv_str("KAFKA_AWS_REGION", "");

    rabbitmq_user = getenv_str("RABBITMQ_USER", "");
    rabbitmq_host = getenv_str("RABBITMQ_HOST", "");
    rabbitmq_port = getenv_int("RABBITMQ_PORT", 5671);
    rabbitmq_pass = getenv_str("RABBITMQ_PASS", "");
    rabbitmq_use_ssl = getenv_bool("RABBITMQ_USE_SSL", true);

    {
        std::string ug = getenv_str("USE_GENERIC_QUEUE", "");
        std::string lower = ug;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        use_generic_queue = (!ug.empty() && lower != "0" && lower != "false" && lower != "no");
    }
    generic_queue_name = getenv_str("GENERIC_QUEUE_NAME", "");

    video_sender_port = getenv_int("VIDEO_SENDER_PORT", 5560);
    fall_inference_port = getenv_int("FALL_INFERENCE_PORT", 5559);

    api_base_url = getenv_str("API_BASE_URL", "");
    antmedia_base_url = getenv_str("ANTMEDIA_BASE_URL", "");
    api_email = getenv_str("API_EMAIL", "");
    api_password = getenv_str("API_PASSWORD", "");
    api_env = getenv_str("API_ENV", "default");

    max_retries = getenv_int("MAX_RETRIES", 3);
}

std::unordered_map<int, CameraConfig> AppConfig::load_camera_configs() const {
    std::unordered_map<int, CameraConfig> out;
    if (total_cameras <= 0) return out;

    for (int i = 1; i <= total_cameras; ++i) {
        std::string suffix = "_" + std::to_string(i);
        CameraConfig c;
        c.id = getenv_int(("CAMERA_ID" + suffix).c_str(), i);
        c.url = getenv_str(("CAMERA_URL" + suffix).c_str(), "");
        c.websocket_url = getenv_str(("WEBSOCKET_URL" + suffix).c_str(), "");
        c.store_id = getenv_int(("STORE_ID" + suffix).c_str(), 0);
        c.client_type = client_type;
        if (c.url.empty()) {
            // Match Python: log warning and skip cameras with no URL
            continue;
        }
        // Match Python CameraConfig.__post_init__ validation
        static const std::vector<std::string> valid_types = {"webrtc", "rtsp", "redis", "video"};
        bool valid_type = std::find(valid_types.begin(), valid_types.end(), c.client_type) != valid_types.end();
        if (!valid_type)
            throw std::runtime_error("Invalid client type '" + c.client_type + "' for camera " + std::to_string(c.id));
        if (!c.store_id)
            throw std::runtime_error("Store ID cannot be empty for camera " + std::to_string(c.id));
        if (c.client_type == "webrtc" && c.websocket_url.empty())
            throw std::runtime_error("WebSocket URL required for WebRTC camera " + std::to_string(c.id) + " but not provided");
        out[i] = c;
    }
    return out;
}

} // namespace config
} // namespace app
