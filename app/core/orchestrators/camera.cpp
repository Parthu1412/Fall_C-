#include "zmq_io.hpp"
#include "../../config.hpp"
#include "../../utils/logger.hpp"
#include "../../utils/redis_client.hpp"
#include "../../utils/rtsp_camera.hpp"
#include <atomic>
#include <chrono>
#include <mutex>
#include <csignal>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

using app::config::CameraConfig;

static std::atomic<bool> g_stop{false};
static std::mutex g_zmq_send_mtx;
static void on_sig(int) { g_stop = true; }

// Matches Python's thread.is_alive() — set to false when thread function exits
static void run_camera_worker(const CameraConfig& cam, zmq::socket_t& push_sock,
                              std::atomic<bool>& alive) {
    using namespace app::core::orchestrators;
    auto& cfg = app::config::AppConfig::getInstance();

    app::utils::Logger::info("[Camera] Thread start camera_id=" + std::to_string(cam.id) +
        " client_type=" + cfg.client_type);

    if (cfg.client_type == "redis") {
        app::utils::RedisConsumer redis(std::to_string(cam.id), static_cast<float>(cfg.fps));
        while (!g_stop) {
            auto r = redis.read();
            if (!r.ok) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            if (r.frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            ZmqFramePacket p;
            p.camera_id = cam.id;
            p.store_id = cam.store_id;
            p.source_path = cam.url;
            p.frame = r.frame;
            p.redis_keypoints = r.redis_keypoints;
            {
                std::lock_guard<std::mutex> lk(g_zmq_send_mtx);
                try {
                    if (!zmq_send_frame_packet(push_sock, p)) {
                        // EAGAIN: HWM full — matches Python zmq.Again → warn + drop
                        app::utils::Logger::warning("[Camera] ZMQ queue full, dropping frame camera_id=" +
                            std::to_string(cam.id));
                    }
                } catch (const zmq::error_t& e) {
                    // Fatal ZMQ error — matches Python zmq.ZMQError → sys.exit(1)
                    app::utils::Logger::error("[Camera] Fatal ZMQ send error: " + std::string(e.what()));
                    std::exit(1);
                }
            }
        }
        alive = false;
        return;
    }

    if (cfg.client_type == "rtsp" || cfg.client_type == "video") {
        // Match Python camera_initialize.py: RTSPCamera(url, fps=config.FPS, buffer_size=config.BUFFER_SIZE)
        app::utils::RTSPCamera cap(cam.url, cfg.fps, cfg.buffer_size);
        while (!g_stop) {
            cv::Mat frame;
            if (!cap.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            ZmqFramePacket p;
            p.camera_id = cam.id;
            p.store_id = cam.store_id;
            p.source_path = cam.url;
            p.frame = frame;
            {
                std::lock_guard<std::mutex> lk(g_zmq_send_mtx);
                try {
                    if (!zmq_send_frame_packet(push_sock, p)) {
                        // EAGAIN: HWM full — matches Python zmq.Again → warn + drop
                        app::utils::Logger::warning("[Camera] ZMQ queue full, dropping frame camera_id=" +
                            std::to_string(cam.id));
                    }
                } catch (const zmq::error_t& e) {
                    // Fatal ZMQ error — matches Python zmq.ZMQError → sys.exit(1)
                    app::utils::Logger::error("[Camera] Fatal ZMQ send error: " + std::string(e.what()));
                    std::exit(1);
                }
            }
        }
        cap.release();
        alive = false;
        return;
    }

    app::utils::Logger::error(
        "[Camera] CLIENT_TYPE=" + cfg.client_type +
        " not supported in C++ camera_reader (use redis, rtsp, or video).");
    alive = false;
}

int main() {
    app::utils::Logger::set_level_from_env();
    auto& cfg = app::config::AppConfig::getInstance();
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    auto cameras = cfg.load_camera_configs();
    if (cameras.empty()) {
        app::utils::Logger::error("[CameraOrche] No cameras configured (check TOTAL_CAMERAS / CAMERA_URL_*).");
        return 1;
    }

    try {
        zmq::context_t ctx(1);
        zmq::socket_t push(ctx, zmq::socket_type::push);
        push.bind("tcp://*:" + std::to_string(cfg.fall_inference_port));
        push.set(zmq::sockopt::sndhwm, 300);
        app::utils::Logger::info("[CameraOrche] Bound ZMQ PUSH *:" + std::to_string(cfg.fall_inference_port));

        // Per-thread alive flags — mirrors Python's thread.is_alive()
        std::vector<std::atomic<bool>> alive_flags(cameras.size());
        for (auto& f : alive_flags) f.store(true);

        std::vector<std::thread> threads;
        threads.reserve(cameras.size());
        std::size_t idx = 0;
        for (const auto& kv : cameras) {
            threads.emplace_back(run_camera_worker, kv.second,
                                 std::ref(push), std::ref(alive_flags[idx++]));
        }

        // Matches Python: monitor every 5s, warn if any thread died
        while (!g_stop) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            for (std::size_t i = 0; i < alive_flags.size(); ++i) {
                if (!alive_flags[i].load()) {
                    app::utils::Logger::warning(
                        "[CameraOrche] A camera thread died (index " + std::to_string(i) + ")");
                }
            }
        }

        g_stop = true;
        for (auto& th : threads) {
            if (th.joinable()) th.join();
        }
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[CameraOrche] Fatal: ") + e.what());
        return 1;
    }
    return 0;
}
