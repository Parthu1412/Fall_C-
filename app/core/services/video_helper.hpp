#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace app {
namespace core {
namespace services {

class VideoHelper {
public:
    /**
     * Writes a video file from a list of frames using FFmpeg via POSIX pipes.
     * Falls back to OpenCV cv::VideoWriter if the pipe fails.
     * * @param frames Vector of OpenCV frames to encode
     * @param output_path Destination path for the .mp4 file
     * @param fps Target frames per second
     * @return true if successful, false otherwise
     */
    static bool write_video(const std::vector<cv::Mat>& frames, 
                            const std::string& output_path, 
                            int fps = 15);

    /** Overload: accepts pre-encoded JPEG bytes — no re-encode needed. */
    static bool write_video(const std::vector<std::vector<uchar>>& jpeg_frames,
                            const std::string& output_path,
                            int fps = 15);
};

} // namespace services
} // namespace core
} // namespace app