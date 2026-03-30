// Interface for the background-threaded RTSP/video camera reader.
// Declares RTSPCamera, which captures frames from an RTSP stream or video file
// on a dedicated thread and exposes getLatestFrames() for FPS-sampled
// non-blocking access by the camera orchestrator.

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

namespace app::utils {

/**
 * RTSP Camera with background frame-reading thread and FPS-based sampling.
 * Matches Python RTSPCamera: background thread reads all frames, samples to
 * target_fps per second, buffers them. Caller's read() pops from the buffer.
 * Auto-reconnects if no frame received for 10 seconds.
 */
class RTSPCamera
{
public:
    explicit RTSPCamera(std::string url, int fps = 1, int buffer_size = 60);
    ~RTSPCamera();

    bool read(cv::Mat& out);
    bool isOpened() const;
    void release();

private:
    void openCapture();
    std::vector<cv::Mat> sampleFramesToTargetFps(const std::vector<cv::Mat>& frames);
    void readFramesLoop();

    std::string url_;
    int target_fps_;
    int buffer_size_;

    cv::VideoCapture cap_;
    mutable std::mutex cap_mtx_;

    std::deque<cv::Mat> frame_buffer_;
    mutable std::mutex buffer_lock_;

    std::atomic<bool> running_{true};
    std::thread thread_;

    std::chrono::steady_clock::time_point last_success_time_;
    std::chrono::steady_clock::time_point buffer_start_time_;
    std::chrono::steady_clock::time_point last_log_time_;

    int consecutive_failures_{0};
    int frames_received_count_{0};
    int frames_sampled_count_{0};

    std::vector<cv::Mat> frame_buffer_1s_;
};

}  // namespace app::utils
