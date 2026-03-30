// Interface for application configuration.
// Declares AppConfig (singleton) and CameraConfig, which hold all settings
// read from environment variables at startup: camera URLs, ZMQ ports, fall
// detection thresholds, frame buffer sizes, and integration credentials.

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace app {
namespace config {

struct CameraConfig {
    int id = 0;
    std::string url;
    std::string client_type;
    int store_id = 0;
    std::string websocket_url;
};

/**
 * Global configuration loaded from environment (and optional .env in cwd).
 * Singleton — first access loads values.
 */
class AppConfig {
public:
    static AppConfig& getInstance();

    void load();

    // Camera / inference
    int total_cameras = 0;
    int num_workers = 1;
    std::string client_type = "rtsp";
    std::string model_path = "yolov8m-pose.pt";
    int torso_angle = 45;
    int leg_angle = 75;
    float keypoints_conf_threshold = 0.7f;
    int fps = 15;
    int buffer_size = 50;

    // Frame buffer (fall orchestrator)
    int video_buffer_size = 75;
    int frames_before_detection = 30;
    int frames_after_detection = 45;
    int fall_sample_interval = 15; // process 1 of every N incoming frames

    // Redis
    std::string redis_host = "localhost";
    int redis_port = 6379;
    std::string redis_password;
    int redis_expiry = 200;
    int64_t longint_max = 9223372036854775807LL;

    // AWS S3
    std::string aws_bucket;
    std::string aws_region;
    std::string aws_object_name;

    // Kafka / MSK
    std::vector<std::string> kafka_bootstrap_servers;
    std::string kafka_client_id;
    std::string kafka_topic;
    std::string kafka_aws_region;

    // RabbitMQ
    std::string rabbitmq_user;
    std::string rabbitmq_host;
    int rabbitmq_port = 5671;
    std::string rabbitmq_pass;
    bool rabbitmq_use_ssl = true;

    bool use_generic_queue = false;
    std::string generic_queue_name;

    // ZMQ
    int video_sender_port = 5560;
    int fall_inference_port = 5559;

    // API / Ant Media (TokenManager)
    std::string api_base_url;
    std::string antmedia_base_url;
    std::string api_email;
    std::string api_password;
    std::string api_env = "default";

    int max_retries = 3;

    std::unordered_map<int, CameraConfig> load_camera_configs() const;

private:
    AppConfig() = default;
};

} // namespace config
} // namespace app
