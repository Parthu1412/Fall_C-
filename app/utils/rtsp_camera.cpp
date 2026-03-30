// RTSP/video camera reader — background-threaded frame capture for a single camera.
// Supports both live RTSP streams and local video files. For video files, throttles
// read speed to match target_fps to prevent fast-forward playback. Exposes a
// thread-safe ring buffer sampled at a configurable FPS for downstream consumers.

#include "rtsp_camera.hpp"
#include "logger.hpp"
#include <stdexcept>

namespace app::utils {

RTSPCamera::RTSPCamera(std::string url, int fps, int buffer_size)
    : url_(std::move(url)), target_fps_(fps), buffer_size_(buffer_size)
{
    openCapture();  // throws ConnectionError on failure (matches Python)

    auto now = std::chrono::steady_clock::now();
    last_success_time_ = now;
    buffer_start_time_ = now;
    last_log_time_ = now;

    thread_ = std::thread(&RTSPCamera::readFramesLoop, this);

    Logger::info("[RTSPCamera] Initialized | url=" + url_ +
                 " | target_fps=" + std::to_string(target_fps_) +
                 " | buffer_size=" + std::to_string(buffer_size_));
}

RTSPCamera::~RTSPCamera() {
    release();
}

void RTSPCamera::openCapture() {
    {
        std::lock_guard<std::mutex> lk(cap_mtx_);
        if (cap_.isOpened()) cap_.release();
        cap_.open(url_);
    }
    if (!isOpened()) {
        Logger::error("[RTSPCamera] Failed to open RTSP stream | url=" + url_);
        throw std::runtime_error("RTSP Camera Initialization Failed");
    }
}

std::vector<cv::Mat> RTSPCamera::sampleFramesToTargetFps(const std::vector<cv::Mat>& frames) {
    if (frames.empty()) return {};
    int total = static_cast<int>(frames.size());
    if (total <= target_fps_) return frames;

    double interval = static_cast<double>(total) / target_fps_;
    std::vector<cv::Mat> sampled;
    sampled.reserve(target_fps_);
    for (int i = 0; i < target_fps_; ++i) {
        int idx = static_cast<int>(i * interval);
        if (idx < total) sampled.push_back(frames[idx]);
    }
    return sampled;
}

// Background thread: read frames from RTSP stream
void RTSPCamera::readFramesLoop() {
    // Detect video file vs live stream — live streams return 0 or negative frame count.
    // Video files must be throttled to target_fps to avoid fast-forward playback;
    bool is_video_file = false;
    {
        std::lock_guard<std::mutex> lk(cap_mtx_);
        double fc = cap_.get(cv::CAP_PROP_FRAME_COUNT);
        is_video_file = (fc > 0);
    }
    const auto frame_interval = std::chrono::microseconds(
        static_cast<long long>(1e6 / std::max(1, target_fps_)));

    while (running_) {
        const auto frame_start = std::chrono::steady_clock::now();
        try {
            cv::Mat frame;
            bool ret;
            {
                std::lock_guard<std::mutex> lk(cap_mtx_);
                ret = cap_.read(frame);
            }

            if (!ret || frame.empty()) {
                consecutive_failures_++;
                auto now = std::chrono::steady_clock::now();
                double time_since_last = std::chrono::duration<double>(now - last_success_time_).count();

                // 10 seconds without a good frame triggers reconnect
                if (time_since_last >= 10.0) {
                    Logger::warning("[RTSPCamera] RTSP stream stalled, attempting reconnect"
                        " | url=" + url_ +
                        " | consecutive_failures=" + std::to_string(consecutive_failures_) +
                        " | time_since_last_success=" + std::to_string(static_cast<int>(time_since_last)) + "s");
                    try {
                        openCapture();
                        consecutive_failures_ = 0;
                        last_success_time_ = std::chrono::steady_clock::now();
                    } catch (const std::exception& e) {
                        Logger::error("[RTSPCamera] Failed to reconnect | url=" + url_ +
                                      " | error=" + e.what());
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    }
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                continue;
            }

            auto now = std::chrono::steady_clock::now();
            last_success_time_ = now;
            consecutive_failures_ = 0;
            frame_buffer_1s_.push_back(frame.clone());
            frames_received_count_++;

            double elapsed = std::chrono::duration<double>(now - buffer_start_time_).count();
            if (elapsed >= 1.0) {
                auto sampled = sampleFramesToTargetFps(frame_buffer_1s_);
                {
                    std::lock_guard<std::mutex> lk(buffer_lock_);
                    for (auto& f : sampled) {
                        frame_buffer_.push_back(std::move(f));
                        while (static_cast<int>(frame_buffer_.size()) > buffer_size_)
                            frame_buffer_.pop_front();
                    }
                }
                frames_sampled_count_ += static_cast<int>(sampled.size());

                // Debug FPS stats log every second
                double log_elapsed = std::chrono::duration<double>(now - last_log_time_).count();
                if (log_elapsed >= 1.0) {
                    double received_fps = log_elapsed > 0 ? frames_received_count_ / log_elapsed : 0.0;
                    double sampled_fps  = log_elapsed > 0 ? frames_sampled_count_  / log_elapsed : 0.0;
                    Logger::debug("[RTSPCamera] FPS stats | url=" + url_ +
                        " | received_fps=" + std::to_string(received_fps).substr(0, 5) +
                        " | sampled_fps=" + std::to_string(sampled_fps).substr(0, 5) +
                        " | target_fps=" + std::to_string(target_fps_));
                    frames_received_count_ = 0;
                    frames_sampled_count_ = 0;
                    last_log_time_ = now;
                }

                frame_buffer_1s_.clear();
                buffer_start_time_ = now;
            }

            // For video files: throttle reading to target_fps so the 1-second sampling
            // window sees ~target_fps frames (not hundreds) — prevents fast-forward output.
            if (is_video_file) {
                auto elapsed = std::chrono::steady_clock::now() - frame_start;
                if (elapsed < frame_interval) {
                    std::this_thread::sleep_for(frame_interval - elapsed);
                }
            }

        } catch (const std::exception& e) {
            if (running_) {
                Logger::error("[RTSPCamera] Error reading frame in background thread"
                              " | url=" + url_ + " | error=" + e.what());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

// Match Python: read() — pops oldest frame from buffer
bool RTSPCamera::read(cv::Mat& out) {
    std::lock_guard<std::mutex> lk(buffer_lock_);
    if (frame_buffer_.empty()) return false;
    out = std::move(frame_buffer_.front());
    frame_buffer_.pop_front();
    return true;
}

bool RTSPCamera::isOpened() const {
    std::lock_guard<std::mutex> lk(cap_mtx_);
    return cap_.isOpened();
}

// Match Python: release() — stops thread, releases cap, clears buffers
void RTSPCamera::release() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
        if (thread_.joinable()) {
            Logger::warning("[RTSPCamera] Background thread did not terminate, may cause resource leaks"
                            " | url=" + url_);
        }
    }
    {
        std::lock_guard<std::mutex> lk(cap_mtx_);
        cap_.release();
    }
    {
        std::lock_guard<std::mutex> lk(buffer_lock_);
        frame_buffer_.clear();
    }
    frame_buffer_1s_.clear();
    Logger::info("[RTSPCamera] Released | url=" + url_);
}

}  // namespace app::utils
