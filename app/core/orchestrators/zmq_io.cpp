#include "zmq_io.hpp"
#include "../../utils/detection_json.hpp"
#include "../../utils/logger.hpp"
#include <chrono>
#include <random>
#include <sstream>

namespace app::core::orchestrators {

namespace {

/** OpenCV imdecode expects InputArray; wrap raw JPEG bytes without copying. */
cv::Mat imdecode_jpeg_message(const zmq::message_t& msg) {
    if (msg.size() == 0) return {};
    const int n = static_cast<int>(msg.size());
    // cppzmq: data() is const void*; cv::Mat(rows,cols,type,ptr) needs void*. imdecode reads only.
    void* bytes = const_cast<void*>(static_cast<const void*>(msg.data()));
    cv::Mat buf(1, n, CV_8U, bytes);
    return cv::imdecode(buf, cv::IMREAD_COLOR);
}

void put_keypoints_json(nlohmann::json& j, const std::vector<std::vector<std::vector<float>>>& kpts) {
    if (kpts.empty()) return;
    nlohmann::json dets = nlohmann::json::array();
    for (const auto& person : kpts) {
        nlohmann::json det;
        det["keypoints"] = person;
        dets.push_back(det);
    }
    j["detections"] = dets;
}

}  // namespace

void json_to_redis_keypoints(const nlohmann::json& j,
                             std::vector<std::vector<std::vector<float>>>& out)
{
    app::utils::parse_detections_keypoints(j, out);
}

bool zmq_send_frame_packet(zmq::socket_t& sock, const ZmqFramePacket& p) {
    std::vector<uchar> jpg;
    if (!cv::imencode(".jpg", p.frame, jpg)) {
        app::utils::Logger::error("[zmq] imencode failed");
        return false;
    }
    nlohmann::json meta;
    meta["camera_id"] = p.camera_id;
    meta["store_id"] = p.store_id;
    if (!p.source_path.empty()) {
        meta["source_path"] = p.source_path;
    }
    put_keypoints_json(meta, p.redis_keypoints);
    std::string meta_s = meta.dump();
    zmq::message_t m0(meta_s.size());
    memcpy(m0.data(), meta_s.data(), meta_s.size());
    zmq::message_t m1(jpg.size());
    memcpy(m1.data(), jpg.data(), jpg.size());
    try {
        // dontwait on first part: EAGAIN = HWM full (matches Python zmq.NOBLOCK → zmq.Again)
        sock.send(m0, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
        sock.send(m1, zmq::send_flags::none);
    } catch (const zmq::error_t& e) {
        if (e.num() == EAGAIN) return false;  // HWM full — caller warns and drops
        // Fatal ZMQ error — re-throw so caller can exit(1) (matches Python zmq.ZMQError → sys.exit)
        throw;
    }
    return true;
}

bool zmq_recv_frame_packet(zmq::socket_t& sock, ZmqFramePacket& p, zmq::recv_flags flags) {
    zmq::message_t m0, m1;
    try {
        auto r0 = sock.recv(m0, flags);
        if (!r0) return false;
        if (!sock.get(zmq::sockopt::rcvmore)) return false;
        auto r1 = sock.recv(m1, zmq::recv_flags::none);
        if (!r1) return false;
    } catch (const zmq::error_t&) {
        return false;
    }
    try {
        std::string meta_s(static_cast<char*>(m0.data()), m0.size());
        auto j = nlohmann::json::parse(meta_s);
        p.camera_id = j.value("camera_id", 0);
        p.store_id = j.value("store_id", 0);
        p.source_path = j.value("source_path", std::string());
        json_to_redis_keypoints(j, p.redis_keypoints);
    } catch (const std::exception& e) {
        app::utils::Logger::error(std::string("[zmq] bad frame meta JSON: ") + e.what());
        return false;
    }
    p.frame_jpg.assign(static_cast<const uchar*>(m1.data()),
                       static_cast<const uchar*>(m1.data()) + m1.size());
    p.frame = imdecode_jpeg_message(m1);
    if (p.frame.empty()) return false;
    return true;
}

std::string make_trace_id() {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    thread_local std::mt19937_64 gen(std::random_device{}());
    std::stringstream ss;
    ss << std::hex << now << "-" << gen();
    return ss.str();
}

bool zmq_send_video_task(zmq::socket_t& sock, const ZmqVideoTaskPacket& t) {
    // video_frames are already JPEG bytes — no encode needed
    std::vector<uchar> det_jpg;
    bool has_det = !t.detection_frame.empty();
    if (has_det && !cv::imencode(".jpg", t.detection_frame, det_jpg)) {
        app::utils::Logger::error("[zmq] imencode detection frame failed");
        return false;
    }
    nlohmann::json meta;
    meta["trace_id"] = t.trace_id;
    meta["camera_id"] = t.camera_id;
    meta["store_id"] = t.store_id;
    meta["has_detection"] = has_det;
    meta["n_frames"] = static_cast<int>(t.video_frames.size());
    if (has_det) meta["det_len"] = static_cast<int>(det_jpg.size());
    std::string meta_s = meta.dump();
    try {
        zmq::message_t m(meta_s.size());
        memcpy(m.data(), meta_s.data(), meta_s.size());
        // dontwait on first part: EAGAIN = HWM full (matches Python zmq.NOBLOCK → zmq.Again)
        sock.send(m, zmq::send_flags::sndmore | zmq::send_flags::dontwait);
        if (has_det) {
            zmq::message_t md(det_jpg.size());
            memcpy(md.data(), det_jpg.data(), det_jpg.size());
            sock.send(md, zmq::send_flags::sndmore);
        }
        for (size_t i = 0; i < t.video_frames.size(); ++i) {
            const auto& jpg = t.video_frames[i];
            zmq::message_t mf(jpg.size());
            memcpy(mf.data(), jpg.data(), jpg.size());
            bool last = (i + 1 == t.video_frames.size());
            sock.send(mf, last ? zmq::send_flags::none : zmq::send_flags::sndmore);
        }
    } catch (const zmq::error_t& e) {
        if (e.num() == EAGAIN) return false;  // HWM full — caller warns and drops
        // Fatal ZMQ error — re-throw so caller can exit(1) (matches Python zmq.ZMQError → sys.exit)
        throw;
    }
    return true;
}

bool zmq_recv_video_task(zmq::socket_t& sock, ZmqVideoTaskPacket& t, zmq::recv_flags flags) {
    zmq::message_t m0;
    try {
        auto r0 = sock.recv(m0, flags);
        if (!r0) return false;
        if (!sock.get(zmq::sockopt::rcvmore)) return false;
    } catch (const zmq::error_t&) {
        return false;
    }

    std::string meta_s(static_cast<char*>(m0.data()), m0.size());
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(meta_s);
    } catch (...) {
        return false;
    }
    t.trace_id = j.value("trace_id", "");
    t.camera_id = j.value("camera_id", 0);
    t.store_id = j.value("store_id", 0);
    bool has_det = j.value("has_detection", false);
    int n_frames = j.value("n_frames", 0);

    zmq::message_t chunk;
    if (has_det) {
        try {
            if (!sock.recv(chunk, zmq::recv_flags::none)) return false;
        } catch (const zmq::error_t&) {
            return false;
        }
        t.detection_frame = imdecode_jpeg_message(chunk);
        if (n_frames > 0 && !sock.get(zmq::sockopt::rcvmore)) return false;
    } else {
        t.detection_frame.release();
    }

    t.video_frames.clear();
    t.video_frames.reserve(n_frames);
    for (int i = 0; i < n_frames; ++i) {
        try {
            if (!sock.recv(chunk, zmq::recv_flags::none)) return false;
        } catch (const zmq::error_t&) {
            return false;
        }
        if (chunk.size() == 0) return false;
        t.video_frames.push_back(std::vector<uchar>(
            static_cast<const uchar*>(chunk.data()),
            static_cast<const uchar*>(chunk.data()) + chunk.size()));
    }
    return true;
}

}  // namespace app::core::orchestrators
