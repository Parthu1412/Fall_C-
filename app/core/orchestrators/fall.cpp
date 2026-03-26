#include "zmq_io.hpp"
#include "../../config.hpp"
#include "../../utils/logger.hpp"
#include "../services/fall.hpp"
#include "../services/frame_buffer_manager.hpp"
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <zmq.hpp>

static std::sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

int main() {
    using namespace app::core::orchestrators;
    app::utils::Logger::set_level_from_env();
    auto& cfg = app::config::AppConfig::getInstance();

    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);

    try {
        zmq::context_t ctx(1);
        zmq::socket_t pull(ctx, zmq::socket_type::pull);
        pull.connect("tcp://127.0.0.1:" + std::to_string(cfg.fall_inference_port));
        zmq::socket_t push(ctx, zmq::socket_type::push);
        push.connect("tcp://127.0.0.1:" + std::to_string(cfg.video_sender_port));
        push.set(zmq::sockopt::sndhwm, 100);

        app::utils::Logger::info("[FallOrche] Pull connect :" + std::to_string(cfg.fall_inference_port) +
            " push connect :" + std::to_string(cfg.video_sender_port));

        app::core::services::FrameBufferManager buffer;
        app::core::services::FallDetectionService fall_svc(true);

        while (!g_stop) {
            ZmqFramePacket pkt;
            if (!zmq_recv_frame_packet(pull, pkt, zmq::recv_flags::dontwait)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            int camera_id = pkt.camera_id;
            buffer.initialize_camera(camera_id);
            buffer.add_frame(camera_id, pkt.frame, pkt.frame_jpg);

            auto ready = buffer.check_and_process_pending_detection(camera_id);
            if (ready.has_value()) {
                ZmqVideoTaskPacket task;
                task.trace_id = make_trace_id();
                task.camera_id = ready->camera_id;
                task.store_id = ready->store_id;
                task.video_frames = std::move(ready->video_frames);
                task.detection_frame = ready->detection_frame.clone();
                try {
                    if (!zmq_send_video_task(push, task)) {
                        // EAGAIN: HWM full — matches Python zmq.Again → warn + drop
                        app::utils::Logger::warning("[FallOrche] Msg-gen queue full, dropping video task trace_id=" + task.trace_id);
                    }
                } catch (const zmq::error_t& e) {
                    // Fatal ZMQ error — matches Python zmq.ZMQError → sys.exit(1)
                    app::utils::Logger::error("[FallOrche] Fatal ZMQ send error: " + std::string(e.what()));
                    std::exit(1);
                }
                continue;
            }

            if (buffer.has_pending_detection(camera_id)) continue;
            if (!buffer.should_process_frame(camera_id)) continue;

            int processed_frame_num = buffer.increment_processed_count(camera_id);

            auto t0 = std::chrono::steady_clock::now();
            try {
            auto [fall_detected, annotated] = fall_svc.detect_fall(pkt.frame, pkt.redis_keypoints);
            const auto elapsed = std::chrono::steady_clock::now() - t0;
            const double inference_time_s =
                std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

            // Match Python: logger.info("Fall inference took", extra={inference_time,camera_id,processed_frame_num})
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3);
            oss << "Fall inference took"
                << " | inference_time=" << inference_time_s
                << " | camera_id=" << camera_id
                << " | processed_frame_num=" << processed_frame_num;
            app::utils::Logger::info(oss.str());

            if (fall_detected && !annotated.empty()) {
                buffer.handle_fall_detection(camera_id, annotated, pkt.store_id, processed_frame_num);
                auto ready2 = buffer.check_and_process_pending_detection(camera_id);
                if (ready2.has_value()) {
                    ZmqVideoTaskPacket task;
                    task.trace_id = make_trace_id();
                    task.camera_id = ready2->camera_id;
                    task.store_id = ready2->store_id;
                    task.video_frames = std::move(ready2->video_frames);
                    task.detection_frame = ready2->detection_frame.clone();
                    try {
                        if (!zmq_send_video_task(push, task)) {
                            // EAGAIN: HWM full — matches Python zmq.Again → warn + drop
                            app::utils::Logger::warning("[FallOrche] Msg-gen queue full, dropping video task trace_id=" + task.trace_id);
                        }
                    } catch (const zmq::error_t& ze) {
                        // Fatal ZMQ error — matches Python zmq.ZMQError → sys.exit(1)
                        app::utils::Logger::error("[FallOrche] Fatal ZMQ send error: " + std::string(ze.what()));
                        std::exit(1);
                    }
                }
            }
            } catch (const std::exception& e) {
                app::utils::Logger::error("[FallOrche] Error processing frame camera_id=" +
                    std::to_string(camera_id) + " : " + e.what());
            }
        }
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[FallOrche] Fatal: ") + e.what());
        return 1;
    }
    return 0;
}
