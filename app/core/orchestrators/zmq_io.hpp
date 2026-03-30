// ZMQ I/O interface — declares the shared packet structs and send/recv helpers
// used by all three pipeline processes (camera_reader, fall_inference, msg_gen).
// ZmqFramePacket carries a camera frame from camera_reader to fall_inference.
// ZmqVideoTaskPacket carries a confirmed-fall clip task from fall_inference to msg_gen.

#pragma once

#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace app::core::orchestrators {

struct ZmqFramePacket {
    int camera_id = 0;
    int store_id = 0;
    cv::Mat frame;
    std::vector<uchar> frame_jpg;   // raw JPEG bytes — stored alongside Mat to avoid re-encode
    /** Camera URL / video file path (for logs); optional in JSON. */
    std::string source_path;
    /** Populated when CLIENT_TYPE=redis JSON carries pose detections. */
    std::vector<std::vector<std::vector<float>>> redis_keypoints;
};

/** Two-part message: JSON meta + JPEG frame. */
bool zmq_send_frame_packet(zmq::socket_t& sock, const ZmqFramePacket& p);
/** @return true if a full packet was read; false if would block / incomplete. */
bool zmq_recv_frame_packet(zmq::socket_t& sock, ZmqFramePacket& p,
                           zmq::recv_flags flags = zmq::recv_flags::dontwait);

struct ZmqVideoTaskPacket {
    std::string trace_id;
    int camera_id = 0;
    int store_id = 0;
    std::vector<std::vector<uchar>> video_frames;  // JPEG bytes — no re-encode on send
    cv::Mat detection_frame;
};

bool zmq_send_video_task(zmq::socket_t& sock, const ZmqVideoTaskPacket& t);
bool zmq_recv_video_task(zmq::socket_t& sock, ZmqVideoTaskPacket& t,
                            zmq::recv_flags flags = zmq::recv_flags::dontwait);

std::string make_trace_id();

void json_to_redis_keypoints(const nlohmann::json& j,
                             std::vector<std::vector<std::vector<float>>>& out);

}  // namespace app::core::orchestrators
