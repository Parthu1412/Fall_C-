// MsgGen Orchestrator — Video creation and notification process.
// Receives ZmqVideoTaskPackets from fall_inference via ZMQ PULL, assembles JPEG
// frames into an MP4 clip (with detection overlay), uploads the clip and a
// snapshot image to S3, then publishes a fall-event notification to Kafka and
// RabbitMQ with S3 URLs and metadata.

#include "zmq_io.hpp"
#include "../../config.hpp"
#include "../../kafka/kafka_producer.hpp"
#include "../../mqtt/rabbitmq.hpp"
#include "../../utils/aws.hpp"
#include "../../utils/logger.hpp"
#include "../../utils/message.hpp"
#include "../services/video_helper.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <zmq.hpp>

static std::sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

static std::string utc_iso_timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return std::string(buf);
}

// Process a single video task: encode video, upload to S3, send Kafka and RabbitMQ messages
static void process_task(app::utils::AwsApiManager& /*aws_life*/,
                         app::utils::S3Client& s3,
                         app::kafka::KafkaProducer& kafka,
                         app::mqtt::RabbitMQClient& rmq,
                         const app::core::orchestrators::ZmqVideoTaskPacket& ev)
{
    auto& cfg = app::config::AppConfig::getInstance();
    const std::string& trace_id = ev.trace_id;
    int camera_id = ev.camera_id;
    int store_id = ev.store_id;

    if (ev.video_frames.empty()) {
        app::utils::Logger::warning("[MsgGen] No video_frames trace_id=" + trace_id);
        return;
    }
    if (ev.detection_frame.empty()) {
        app::utils::Logger::warning("[MsgGen] No detection_frame trace_id=" + trace_id);
        return;
    }

    // Create temp video file path
    std::string tmp_dir = "/tmp/fall_videos";
    std::filesystem::path dir(tmp_dir);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm_buf{};

    // Use thread-safe localtime variants
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    std::ostringstream tss;
    tss << std::put_time(&tm_buf, "%Y_%m_%d_%H_%M_%S");
    std::string video_path = tmp_dir + "/" + trace_id + "_" + std::to_string(camera_id) + "_timestamp_" + tss.str() + ".mp4";

    // Upload detection frame to S3 and encode/upload video, then send Kafka and RabbitMQ messages
    std::optional<std::string> image_url;
    std::optional<std::string> video_url;

    {
        std::vector<uchar> jpg;
        if (!cv::imencode(".jpg", ev.detection_frame, jpg)) {
            app::utils::Logger::error("[MsgGen] JPEG encode failed");
            std::exit(1);
        }
        std::string obj = "fall-detection/" + std::to_string(store_id) + "/" + trace_id + ".jpg";
        image_url = s3.upload_bytes_and_get_url(jpg.data(), jpg.size(), obj, "image/jpeg");
        if (!image_url) {
            app::utils::Logger::error("[MsgGen] S3 image upload failed");
            std::exit(1);
        }
    }
    // Encode video from frames and upload to S3
    if (!app::core::services::VideoHelper::write_video(ev.video_frames, video_path, 15)) {
        app::utils::Logger::warning("[MsgGen] Video encode failed, dropping task trace_id=" + trace_id);
        return;
    }

    // Verify video file exists before attempting upload
    std::ifstream vf(video_path, std::ios::binary);
    if (vf.good()) {
        std::string objv = "fall-detection/" + std::to_string(store_id) + "/" + trace_id + ".mp4";
        video_url = s3.upload_video_file_and_get_url(video_path, objv, "video/mp4");
        if (!video_url) {
            app::utils::Logger::error("[MsgGen] S3 video upload failed");
            std::exit(1);
        }
    }

    if (!video_url || !image_url) {
        app::utils::Logger::error("[MsgGen] Missing S3 URLs");
        std::exit(1);
    }

    app::utils::FallMessage msg;
    msg.store_id = store_id;
    msg.moksa_camera_id = camera_id;
    msg.s3_uri = *image_url;
    msg.video_uri = *video_url;
    msg.trace_id = trace_id;
    msg.timestamp = utc_iso_timestamp();

    kafka.produce(cfg.kafka_topic, msg);
    app::utils::Logger::info("Fall detection sent to Kafka");

    if (rmq.is_connected()) {
        std::string queue = cfg.use_generic_queue ? cfg.generic_queue_name : ("fall_" + std::to_string(store_id));
        rmq.publish(queue, msg);
        app::utils::Logger::info("Published fall detection to RabbitMQ");
    }

    try {
        std::filesystem::remove(video_path, ec);
    } catch (...) {}
}


int main() { 
    using namespace app::core::orchestrators;
    // Initialize logging, AWS API, Kafka producer, RabbitMQ client, and ZMQ socket
    app::utils::Logger::set_level_from_env();
    // Handle SIGINT/SIGTERM for graceful shutdown
    auto& cfg = app::config::AppConfig::getInstance();
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    // Initialize AWS API manager (lifetime tied to main)
    app::utils::AwsApiManager aws_life;
    app::kafka::KafkaProducer kafka;
    kafka.start_with_retry();

    // Initialize RabbitMQ client with retry logic
    std::unique_ptr<app::mqtt::RabbitMQClient> rmq;
    for (int a = 0; a < cfg.max_retries; ++a) {
        rmq = std::make_unique<app::mqtt::RabbitMQClient>();
        rmq->connect_with_retry(); //Explicit connect_rabbitmq_with_retry()
        if (rmq->is_connected() || cfg.rabbitmq_host.empty()) break;
        app::utils::Logger::error("[MsgGen] RabbitMQ connect failed attempt " + std::to_string(a + 1));
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!cfg.rabbitmq_host.empty() && rmq && !rmq->is_connected()) {
        app::utils::Logger::error("[MsgGen] RabbitMQ connection failed after max retries — exiting");
        return 1;
    }

    // Initialize S3 client
    app::utils::S3Client s3;

    zmq::context_t ctx(1);
    zmq::socket_t pull(ctx, zmq::socket_type::pull);
    pull.bind("tcp://*:" + std::to_string(cfg.video_sender_port));
    pull.set(zmq::sockopt::rcvhwm, 100);
    app::utils::Logger::info(
        "Bound to port " + std::to_string(cfg.video_sender_port) + " for receiving from fall orchestrator");
    app::utils::Logger::info("MsgGen initialized");

    std::vector<std::future<void>> active_tasks;

    while (!g_stop) {
        // prune completed futures
        active_tasks.erase(
            std::remove_if(active_tasks.begin(), active_tasks.end(),
                [](std::future<void>& f) {
                    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                }),
            active_tasks.end());

        ZmqVideoTaskPacket ev;
        if (!zmq_recv_video_task(pull, ev, zmq::recv_flags::dontwait)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        active_tasks.push_back(std::async(std::launch::async,
            [ev = std::move(ev), &aws_life, &s3, &kafka]() {
                thread_local std::unique_ptr<app::mqtt::RabbitMQClient> tl_rmq;
                if (!tl_rmq) tl_rmq = std::make_unique<app::mqtt::RabbitMQClient>();
                try {
                    process_task(aws_life, s3, kafka, *tl_rmq, ev);
                } catch (const std::exception& e) {
                    app::utils::Logger::error(std::string("[MsgGen] process_task: ") + e.what());
                }
            }));
    }

    // drain in-flight tasks before shutdown
    for (auto& f : active_tasks) {
        try { f.get(); } catch (...) {}
    }

    kafka.stop();
    app::utils::Logger::info("[MsgGen] shutdown");
    return 0;
}
