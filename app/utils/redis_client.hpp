// Redis client helper — reads pre-computed pose keypoints published by the
// centralized-yolo Redis producer into the frame pipeline, enabling
// CLIENT_TYPE=redis mode where C++ skips local pose inference entirely.

#pragma once

#include <sw/redis++/redis++.h>

#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "app/config.hpp"
#include "detection_json.hpp"
#include "logger.hpp"

using namespace sw::redis;
using json = nlohmann::json;

namespace app {
namespace utils {

/**
 * RedisConsumer - Consume frames from Redis (matches centralized-yolo producer).
 *
 * Keys (C++ reads JSON format):
 *   - {camera_id}_latest_str  -> plain string frame number (C++)
 *   - {camera_id}_{frame_number}_json -> JSON with frame_base64, detections, timestamp
 */
class RedisConsumer
{
public:
    RedisConsumer(const std::string& camera_id, float fps = 0.0f)
        : camera_id_(camera_id),
          frame_number_(0),
          last_read_time_(0.0),
          start_frame_number_(0),
          last_successful_frame_(-1),
          last_frame_number_(0)
    {
        auto& config = app::config::AppConfig::getInstance();
        fps_ = (fps > 0) ? fps : static_cast<float>(config.fps);
        if (fps_ <= 0)
            fps_ = 15.0f;

        ConnectionOptions opts;
        opts.host = config.redis_host;
        opts.port = config.redis_port;
        if (!config.redis_password.empty())
        {
            opts.password = config.redis_password;
        }
        redis_ = std::make_unique<Redis>(opts);
        app::utils::Logger::info("[RedisConsumer] Connected for camera: " + camera_id_);
    }

    struct ReadResult {
        bool ok;
        cv::Mat frame;
        std::vector<std::vector<std::vector<float>>> redis_keypoints;
    };

    ReadResult read()
    {
        ReadResult result{false, cv::Mat(), {}};
        try
        {
            auto latest_opt = get_latest_frame_number();
            if (!latest_opt.has_value())
            {
                result.ok = true;
                return result;
            }
            int latest_frame_number = latest_opt.value();

            if (fps_ > 0)
            {
                auto target_opt = calculate_target_frame(latest_frame_number);
                if (!target_opt.has_value())
                {
                    result.ok = true;
                    return result;
                }
                frame_number_ = target_opt.value();
            }

            std::string json_key = camera_id_ + "_" + std::to_string(frame_number_) + "_json";
            auto frame_data_opt = redis_->get(json_key);
            if (!frame_data_opt)
            {
                if (should_wait_or_reset(latest_frame_number))
                {
                    result.ok = true;
                    return result;
                }
                return result;
            }

            ReadResult parsed = parse_frame_payload(*frame_data_opt);
            if (parsed.frame.empty())
            {
                result.ok = true;
                return result;
            }
            result.frame = std::move(parsed.frame);
            result.redis_keypoints = std::move(parsed.redis_keypoints);

            last_successful_frame_ = frame_number_;
            last_frame_number_ = frame_number_;

            long long max_val = 9223372036854775807LL;
            if (frame_number_ >= max_val)
            {
                frame_number_ = 0;
                last_successful_frame_ = -1;
            }

            result.ok = true;
            return result;
        } catch (const Error& e)
        {
            app::utils::Logger::error("[RedisConsumer] Error reading from Redis: " +
                                      std::string(e.what()));
            // Attempt one reconnect on connection error
            try
            {
                reconnect();
            } catch (...)
            {
            }
            result.ok = false;
            return result;
        }
    }

    void reconnect()
    {
        // Release and recreate connection
        try
        {
            auto& config = app::config::AppConfig::getInstance();
            ConnectionOptions opts;
            opts.host = config.redis_host;
            opts.port = config.redis_port;
            if (!config.redis_password.empty())
                opts.password = config.redis_password;
            redis_ = std::make_unique<Redis>(opts);
            redis_->ping();
            app::utils::Logger::info("[RedisConsumer] Reconnected for camera: " + camera_id_);
        } catch (const Error& e)
        {
            app::utils::Logger::error("[RedisConsumer] Reconnect failed: " + std::string(e.what()));
            throw;
        }
    }

private:
    std::string camera_id_;
    float fps_;
    int frame_number_;
    double last_read_time_;
    int start_frame_number_;
    int last_successful_frame_;
    int last_frame_number_;
    std::unique_ptr<Redis> redis_;

    std::optional<int> get_latest_frame_number()
    {
        std::string key = camera_id_ + "_latest_str";
        auto val_opt = redis_->get(key);
        if (!val_opt)
        {
            app::utils::Logger::warning(
                "[RedisConsumer] No latest frame data in Redis for camera: " + camera_id_);
            return std::nullopt;
        }
        try
        {
            int fn = std::stoi(*val_opt);
            if (frame_number_ == 0)
            {
                frame_number_ = fn;
            }
            return fn;
        } catch (...)
        {
            return std::nullopt;
        }
    }

    bool is_ahead(int frame_number) const
    {
        return frame_number_ > frame_number;
    }

    bool is_behind(int frame_number) const
    {
        if (frame_number_ < frame_number)
        {
            app::utils::Logger::warning("[RedisConsumer] Consumer behind producer for " +
                                        camera_id_);
            return true;
        }
        return false;
    }

    std::optional<int> calculate_target_frame(int latest_frame_number)
    {
        if (fps_ <= 0)
            return frame_number_;

        double current_time =
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                .count();

        if (last_read_time_ == 0.0 || start_frame_number_ == 0)
        {
            last_read_time_ = current_time;
            start_frame_number_ = frame_number_;
            return frame_number_;
        }

        double elapsed_time = current_time - last_read_time_;
        // Accounts for camera running at ~15fps raw; advances by that many raw frames per target
        // frame.
        int raw_per_target = (fps_ > 0) ? static_cast<int>(15.0f / fps_) : 1;
        if (raw_per_target < 1)
            raw_per_target = 1;
        int frames_to_advance = static_cast<int>(elapsed_time * fps_) * raw_per_target;

        if (frames_to_advance > 0)
        {
            int target_frame = frame_number_ + frames_to_advance;
            if (target_frame > latest_frame_number)
            {
                target_frame = latest_frame_number;
            }
            if (last_successful_frame_ >= 0 && target_frame == last_successful_frame_)
            {
                return std::nullopt;
            }
            last_read_time_ = current_time;
            return target_frame;
        }
        return std::nullopt;
    }

    bool should_wait_or_reset(int latest_frame_number)
    {
        if (is_ahead(latest_frame_number))
            return true;
        if (is_behind(latest_frame_number))
        {
            frame_number_ = latest_frame_number;
            return true;
        }
        return (frame_number_ == last_frame_number_ || frame_number_ == latest_frame_number);
    }

    ReadResult parse_frame_payload(const std::string& data)
    {
        ReadResult r{true, cv::Mat(), {}};
        try
        {
            json payload = json::parse(data);
            if (!payload.contains("frame_base64"))
            {
                r.ok = true;
                return r;
            }
            app::utils::parse_detections_keypoints(payload, r.redis_keypoints);
            std::string b64 = payload["frame_base64"].get<std::string>();
            std::vector<uchar> img_bytes;
            size_t out_len = base64_decode(b64, img_bytes);
            if (out_len == 0)
                return r;
            r.frame = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
            return r;
        } catch (const json::exception& e)
        {
            app::utils::Logger::warning("[RedisConsumer] JSON parse error: " +
                                        std::string(e.what()));
            return r;
        }
    }

    static size_t base64_decode(const std::string& in, std::vector<uchar>& out)
    {
        static const char tbl[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::vector<int> T(256, -1);
        for (int i = 0; i < 64; ++i)
            T[static_cast<uchar>(tbl[i])] = i;

        size_t len = in.size();
        out.resize(len * 3 / 4 + 4);
        size_t out_len = 0;
        int val = 0, bits = -8;
        for (size_t i = 0; i < len; ++i)
        {
            int c = T[static_cast<uchar>(in[i])];
            if (c < 0)
                continue;
            val = (val << 6) + c;
            bits += 6;
            if (bits >= 0)
            {
                if (out_len < out.size())
                    out[out_len++] = static_cast<uchar>((val >> bits) & 0xFF);
                bits -= 8;
            }
        }
        out.resize(out_len);
        return out_len;
    }
};

}  // namespace utils
}  // namespace app
