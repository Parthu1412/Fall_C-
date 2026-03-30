// Video assembly helper — encodes a sequence of JPEG frames into an MP4 clip.
// Writes frames through an FFmpeg pipe (H.264, configurable FPS) and falls back
// to OpenCV cv::VideoWriter if FFmpeg is unavailable or the pipe open fails.

#include "video_helper.hpp"
#include "../../utils/logger.hpp"
#include <cstdio>
#include <filesystem>

// Cross-platform pipe macros
#ifdef _WIN32
    #define POPEN _popen
    #define PCLOSE _pclose
#else
    #define POPEN popen
    #define PCLOSE pclose
#endif

namespace app {
namespace core {
namespace services {

// Writes a video file from a sequence of OpenCV Mats, using FFmpeg for encoding with an OpenCV fallback.
bool VideoHelper::write_video(const std::vector<cv::Mat>& frames, 
                              const std::string& output_path, 
                              int fps) 
{
    if (frames.empty()) {
        app::utils::Logger::error("[VideoHelper] No frames to write for " + output_path);
        return false;
    }

    
    try {
        auto parent = std::filesystem::path(output_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    } catch (const std::filesystem::filesystem_error& e) {
        app::utils::Logger::warning(std::string("[VideoHelper] Could not create output directory: ") + e.what());
    }

    // Redirect stderr to a temp file so we can log FFmpeg errors on failure
    std::string ffmpeg_log = output_path + ".ffmpeg_stderr.tmp";
    std::string stderr_redirect = " 2>" + ffmpeg_log;

    std::string command = "ffmpeg -y -f image2pipe -vcodec mjpeg -r " + std::to_string(fps) +
                          " -i - -an -c:v libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p "
                          "-movflags +faststart -tune zerolatency " + output_path + stderr_redirect;

    // Open a pipe to the FFmpeg process for writing ("w")
    FILE* pipe = POPEN(command.c_str(), "w");
    
    if (!pipe) {
        app::utils::Logger::warning("[VideoHelper] FFmpeg pipe failed to open. Falling back to OpenCV VideoWriter.");
        
        // --- OpenCV Fallback Implementation ---
        int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // H.264 codec
        cv::Size frame_size = frames[0].size();
        cv::VideoWriter writer(output_path, cv::CAP_FFMPEG, fourcc, (double)fps, frame_size);
        
        if (!writer.isOpened()) {
            app::utils::Logger::error("[VideoHelper] OpenCV VideoWriter fallback also failed.");
            return false;
        }
        
        for (const auto& frame : frames) {
            if (!frame.empty()) {
                writer.write(frame);
            }
        }
        writer.release();
        app::utils::Logger::info("[VideoHelper] Video written successfully via OpenCV fallback: " + output_path);
        return true;
    }

    // --- FFmpeg Pipe Implementation ---
    for (const auto& frame : frames) {
        if (frame.empty()) continue;

        std::vector<uchar> buffer;
        // Encode the Mat to jpeg as expected by the `-f image2pipe -vcodec mjpeg` flag
        if (cv::imencode(".jpg", frame, buffer)) {
            // Write the raw bytes directly to the FFmpeg stdin pipe
            fwrite(buffer.data(), 1, buffer.size(), pipe);
        }
    }

    // Close the pipe and wait for FFmpeg to finish finalizing the mp4 container
    int returnCode = PCLOSE(pipe);
    
    // In POSIX, pclose returns the exit status of the shell. 
    // A non-zero return code means FFmpeg failed or crashed.
    if (returnCode != 0) {
        // Read and log stderr output on failure
        std::string ffmpeg_err;
        if (FILE* ferr = std::fopen(ffmpeg_log.c_str(), "r")) {
            char buf[256];
            while (std::fgets(buf, sizeof(buf), ferr)) ffmpeg_err += buf;
            std::fclose(ferr);
        }
        std::filesystem::remove(ffmpeg_log);
        app::utils::Logger::error("[VideoHelper] FFmpeg error for " + output_path + ": " + ffmpeg_err);
        return false;
    }
    std::filesystem::remove(ffmpeg_log);

    app::utils::Logger::info("[VideoHelper] Video written successfully via FFmpeg: " + output_path);
    return true;
}

// Overload: accepts pre-encoded JPEG bytes — no re-encode needed, just pipe to FFmpeg or fallback to OpenCV if pipe fails.
bool VideoHelper::write_video(const std::vector<std::vector<uchar>>& jpeg_frames,
                              const std::string& output_path,
                              int fps)
{
    if (jpeg_frames.empty()) {
        app::utils::Logger::error("[VideoHelper] No frames to write for " + output_path);
        return false;
    }

    // Create output directory if it doesn't exist
    try {
        auto parent = std::filesystem::path(output_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    } catch (const std::filesystem::filesystem_error& e) {
        app::utils::Logger::warning(std::string("[VideoHelper] Could not create output directory: ") + e.what());
    }

    // Redirect stderr to a temp file so we can log FFmpeg errors on failure
    std::string ffmpeg_log2 = output_path + ".ffmpeg_stderr.tmp";
    std::string stderr_redirect = " 2>" + ffmpeg_log2;

    std::string command = "ffmpeg -y -f image2pipe -vcodec mjpeg -r " + std::to_string(fps) +
                          " -i - -an -c:v libx264 -preset ultrafast -crf 35 -pix_fmt yuv420p "
                          "-movflags +faststart -tune zerolatency " + output_path + stderr_redirect;

    FILE* pipe = POPEN(command.c_str(), "w");
    if (!pipe) {
        app::utils::Logger::warning("[VideoHelper] FFmpeg pipe failed. Falling back to OpenCV VideoWriter.");
        // Decode first frame to get dimensions
        cv::Mat first;
        for (const auto& jpg : jpeg_frames) {
            cv::Mat buf(1, static_cast<int>(jpg.size()), CV_8U, const_cast<uchar*>(jpg.data()));
            first = cv::imdecode(buf, cv::IMREAD_COLOR);
            if (!first.empty()) break;
        }
        if (first.empty()) {
            app::utils::Logger::error("[VideoHelper] Could not decode any frame for OpenCV fallback.");
            return false;
        }
        // --- OpenCV Fallback Implementation ---
        int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
        cv::VideoWriter writer(output_path, cv::CAP_FFMPEG, fourcc, (double)fps, first.size());
        if (!writer.isOpened()) {
            app::utils::Logger::error("[VideoHelper] OpenCV VideoWriter fallback also failed.");
            return false;
        }
        // Write pre-encoded JPEG frames to the VideoWriter (decoding them first)
        for (const auto& jpg : jpeg_frames) {
            cv::Mat buf(1, static_cast<int>(jpg.size()), CV_8U, const_cast<uchar*>(jpg.data()));
            cv::Mat frame = cv::imdecode(buf, cv::IMREAD_COLOR);
            if (!frame.empty()) writer.write(frame);
        }
        writer.release();
        app::utils::Logger::info("[VideoHelper] Video written via OpenCV fallback: " + output_path);
        return true;
    }

    for (const auto& jpg : jpeg_frames) {
        fwrite(jpg.data(), 1, jpg.size(), pipe);
    }

    int returnCode = PCLOSE(pipe);
    // In POSIX, pclose returns the exit status of the shell. A non-zero return code means FFmpeg failed or crashed.
    if (returnCode != 0) {
        std::string ffmpeg_err;
        if (FILE* ferr = std::fopen(ffmpeg_log2.c_str(), "r")) {
            char buf[256];
            while (std::fgets(buf, sizeof(buf), ferr)) ffmpeg_err += buf;
            std::fclose(ferr);
        }
        std::filesystem::remove(ffmpeg_log2);
        app::utils::Logger::error("[VideoHelper] FFmpeg error for " + output_path + ": " + ffmpeg_err);
        return false;
    }
    std::filesystem::remove(ffmpeg_log2);

    app::utils::Logger::info("[VideoHelper] Video written via FFmpeg: " + output_path);
    return true;
}

} // namespace services
} // namespace core
} // namespace app