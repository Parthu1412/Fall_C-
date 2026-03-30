// Interface for per-camera frame buffering and fall-detection state management.
// Declares FrameBufferManager, which stores rolling JPEG frame windows per camera
// and exposes should_process_frame(), update(), and get_result() for use by
// the fall_inference orchestrator.

#pragma once

#include <deque>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <optional>
#include <opencv2/opencv.hpp>
#include "../../config.hpp"

namespace app {
namespace core {
namespace services {

// Represents a completed fall detection result, including the annotated detection frame and associated video frames.
struct DetectionResult {
    int camera_id = 0;
    int store_id = 0;
    cv::Mat detection_frame;
    std::vector<std::vector<uchar>> video_frames;  // JPEG bytes
};

// Internal struct to track pending detection info for a camera_id, including the annotated frame and buffers of frames before/after detection.
struct DetectionInfo {
    int store_id = 0;
    int buffer_size_at_detection = 0;
    int frame_counter_at_detection = 0;
    int processed_frame_num = 0;
    int required_after_frames = 0;
    cv::Mat frame;
    std::vector<std::vector<uchar>> frames_before_detection;  // JPEG bytes
    std::vector<std::vector<uchar>> frames_after_pending;     // JPEG bytes
};

class FrameBufferManager {
public:
    FrameBufferManager();

    // Prevent copying to maintain strict mutex ownership
    FrameBufferManager(const FrameBufferManager&) = delete;
    FrameBufferManager& operator=(const FrameBufferManager&) = delete;

    void initialize_camera(int camera_id);
    void add_frame(int camera_id, const cv::Mat& frame, const std::vector<uchar>& jpg);

    // Returns detection with video_frames if enough 'after' frames collected, else std::nullopt
    std::optional<DetectionResult> check_and_process_pending_detection(int camera_id);

    bool has_pending_detection(int camera_id) const;

    // Process 1 of every 15 frames (matches Python)
    bool should_process_frame(int camera_id) const;

    // Returns the number of frames processed so far for the given camera_id.
    int increment_processed_count(int camera_id);

    // Store detection info when fall is detected (only call when is_fall == true)
    void handle_fall_detection(int camera_id, const cv::Mat& annotated_frame,
                              int store_id, int processed_frame_num);

    void reset_camera(int camera_id);
    void reset_camera_preserving_frames(int camera_id, const std::vector<std::vector<uchar>>& frames_to_preserve);

private:
    // Core state mapped by camera_id
    std::unordered_map<int, std::deque<std::vector<uchar>>> frame_buffers_;  // stores JPEG bytes
    std::unordered_map<int, int> frame_counters_;
    std::unordered_map<int, int> processed_frame_counts_;
    std::unordered_map<int, DetectionInfo> detection_info_;

    // Configuration limits
    int video_buffer_size_;
    int frames_before_detection_;
    int frames_after_detection_;

    // Mutex for thread-safe operations across all maps
    mutable std::mutex mtx_;
};

} // namespace services
} // namespace core
} // namespace app