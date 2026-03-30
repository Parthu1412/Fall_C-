// Frame buffering utilities for managing fall-detection context per camera.
// Maintains a ring buffer of JPEG frames per camera, tracks fall state
// (pre/post-detection windows), and assembles the before+after frame lists
// needed by msg_gen to produce a fall-event video clip.

#include "frame_buffer_manager.hpp"

#include <algorithm>

namespace app {
namespace core {
namespace services {

FrameBufferManager::FrameBufferManager()
{
    auto& cfg = app::config::AppConfig::getInstance();
    video_buffer_size_ = cfg.video_buffer_size;
    frames_before_detection_ = cfg.frames_before_detection;
    frames_after_detection_ = cfg.frames_after_detection;
}

void FrameBufferManager::initialize_camera(int camera_id)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (frame_buffers_.count(camera_id) == 0)
    {
        frame_buffers_[camera_id] = {};
        frame_counters_[camera_id] = 0;
        processed_frame_counts_[camera_id] = 0;
    }
}

void FrameBufferManager::add_frame(int camera_id, const cv::Mat& frame,
                                   const std::vector<uchar>& jpg)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto& buf = frame_buffers_[camera_id];
    if (frame.empty() || jpg.empty())
        return;
    buf.push_back(jpg);
    while (static_cast<int>(buf.size()) > video_buffer_size_)
    {
        buf.pop_front();
    }
    frame_counters_[camera_id]++;

    // Append to pending "after" sequence if collecting post-fall frames
    if (detection_info_.count(camera_id))
    {
        auto& info = detection_info_[camera_id];
        if (info.required_after_frames > 0)
        {
            int need = info.required_after_frames;
            int have = static_cast<int>(info.frames_after_pending.size());
            if (have < need)
            {
                info.frames_after_pending.push_back(jpg);
            }
        }
    }
}

std::optional<DetectionResult> FrameBufferManager::check_and_process_pending_detection(
    int camera_id)
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = detection_info_.find(camera_id);
    if (it == detection_info_.end())
        return std::nullopt;

    DetectionInfo& info = it->second;
    if (info.required_after_frames <= 0)
    {
        DetectionResult r;
        r.camera_id = camera_id;
        r.store_id = info.store_id;
        r.detection_frame = info.frame.clone();
        std::vector<uchar> det_jpg;
        cv::imencode(".jpg", info.frame, det_jpg);
        r.video_frames = info.frames_before_detection;
        r.video_frames.push_back(std::move(det_jpg));
        for (const auto& af : info.frames_after_pending)
            r.video_frames.push_back(af);
        // Match Python: video_frames = video_frames[:self.video_buffer_size]
        if (static_cast<int>(r.video_frames.size()) > video_buffer_size_)
            r.video_frames.resize(video_buffer_size_);
        detection_info_.erase(it);
        // Match Python _create_and_return_detection -> reset_camera(camera_id)
        // after returning a completed detection payload.
        frame_buffers_[camera_id].clear();
        frame_counters_[camera_id] = 0;
        processed_frame_counts_[camera_id] = 0;
        return r;
    }

    if (static_cast<int>(info.frames_after_pending.size()) < info.required_after_frames)
        return std::nullopt;

    DetectionResult r;
    r.camera_id = camera_id;
    r.store_id = info.store_id;
    r.detection_frame = info.frame.clone();

    std::vector<uchar> det_jpg2;
    cv::imencode(".jpg", info.frame, det_jpg2);
    r.video_frames = info.frames_before_detection;
    r.video_frames.push_back(std::move(det_jpg2));
    for (const auto& af : info.frames_after_pending)
    {
        r.video_frames.push_back(af);
    }
    // Match Python: video_frames = video_frames[:self.video_buffer_size]
    if (static_cast<int>(r.video_frames.size()) > video_buffer_size_)
        r.video_frames.resize(video_buffer_size_);

    detection_info_.erase(it);
    // Match Python _create_and_return_detection -> reset_camera(camera_id)
    // after returning a completed detection payload.
    frame_buffers_[camera_id].clear();
    frame_counters_[camera_id] = 0;
    processed_frame_counts_[camera_id] = 0;
    return r;
}

bool FrameBufferManager::has_pending_detection(int camera_id) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    // Match Python semantics exactly:
    //   return camera_id in self.detection_info
    return detection_info_.find(camera_id) != detection_info_.end();
}

bool FrameBufferManager::should_process_frame(int camera_id) const
{
    std::lock_guard<std::mutex> lock(mtx_);
    auto& cfg = app::config::AppConfig::getInstance();
    int interval = std::max(1, cfg.fall_sample_interval);
    int fc = frame_counters_.count(camera_id) ? frame_counters_.at(camera_id) : 0;
    // Do not run inference until the rolling buffer has accumulated at least
    // frames_before_detection_ frames, so the pre-detection window is always full.
    auto buf_it = frame_buffers_.find(camera_id);
    if (buf_it == frame_buffers_.end() ||
        static_cast<int>(buf_it->second.size()) < frames_before_detection_)
        return false;
    return (fc % interval) == 1;
}

int FrameBufferManager::increment_processed_count(int camera_id)
{
    std::lock_guard<std::mutex> lock(mtx_);
    return ++processed_frame_counts_[camera_id];
}

void FrameBufferManager::handle_fall_detection(int camera_id, const cv::Mat& annotated_frame,
                                               int store_id, int processed_frame_num)
{
    std::lock_guard<std::mutex> lock(mtx_);
    DetectionInfo info;
    info.store_id = store_id;
    info.processed_frame_num = processed_frame_num;
    info.frame = annotated_frame.clone();
    info.required_after_frames = frames_after_detection_;

    const auto& buf = frame_buffers_[camera_id];
    info.buffer_size_at_detection = static_cast<int>(buf.size());
    int take = std::min(frames_before_detection_, static_cast<int>(buf.size()));
    if (take > 0)
    {
        auto start = buf.end() - take;
        for (auto it = start; it != buf.end(); ++it)
        {
            info.frames_before_detection.push_back(*it);  // already JPEG bytes
        }
    }
    // Match Python: required_after = video_buffer_size - actual_before_count - 1
    int actual_before_count = static_cast<int>(info.frames_before_detection.size());
    info.required_after_frames = std::max(0, video_buffer_size_ - actual_before_count - 1);
    info.frames_after_pending.clear();
    detection_info_[camera_id] = std::move(info);
}

void FrameBufferManager::reset_camera(int camera_id)
{
    std::lock_guard<std::mutex> lock(mtx_);
    frame_buffers_.erase(camera_id);
    frame_counters_.erase(camera_id);
    processed_frame_counts_.erase(camera_id);
    detection_info_.erase(camera_id);
}

void FrameBufferManager::reset_camera_preserving_frames(
    int camera_id, const std::vector<std::vector<uchar>>& frames_to_preserve)
{
    std::lock_guard<std::mutex> lock(mtx_);
    frame_buffers_[camera_id] =
        std::deque<std::vector<uchar>>(frames_to_preserve.begin(), frames_to_preserve.end());
    frame_counters_[camera_id] = 0;
    processed_frame_counts_[camera_id] = 0;
    detection_info_.erase(camera_id);
}

}  // namespace services
}  // namespace core
}  // namespace app
