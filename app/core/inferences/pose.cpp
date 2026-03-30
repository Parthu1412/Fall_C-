// Pose inference using a TorchScript YOLO-pose model.
// Loads the model onto GPU (or CPU fallback), auto-detects FP16/BF16/FP32 weight dtype,
// runs 100 warmup forward passes, then exposes detect() which:
//   - letterbox-resizes each frame (aspect-ratio preserving, grey padding),
//   - runs the TorchScript forward pass,
//   - applies GPU confidence filtering + CPU greedy NMS, and
//   - decodes 56-dim rows into 17 keypoints × (x, y, visibility) per detected person.

#include "pose.hpp"

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <torch/cuda.h>
#include <torch/nn/functional.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <opencv2/dnn.hpp>

#include "../../config.hpp"
#include "../../utils/logger.hpp"

namespace app::core::inferences {

// Constructor — matches pose.hpp: dtype auto-detected

PoseInference::PoseInference(const std::string& model_path)
    : device_(at::hasCUDA() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU))
{
    try
    {
        module_ = torch::jit::load(model_path, device_);
        module_.eval();

        // Auto-detect model weight dtype (handles FP16 TorchScript exports)
        for (const auto& param : module_.parameters())
        {
            const auto st = static_cast<c10::ScalarType>(param.scalar_type());
            if (c10::isFloatingType(st))
            {
                model_elem_dtype_ = st;
                break;
            }
        }
        if (model_elem_dtype_ == c10::ScalarType::Half)
            app::utils::Logger::info("[PoseInference] FP16 model detected; inputs will be cast.");
        else if (model_elem_dtype_ == c10::ScalarType::BFloat16)
            app::utils::Logger::info("[PoseInference] BF16 model detected; inputs will be cast.");

        app::utils::Logger::info("[PoseInference] Loaded on " +
                                 std::string(device_.is_cuda() ? "GPU" : "CPU") + ": " +
                                 model_path);

        // Warmup: run 100 dummy frames
        app::utils::Logger::info("[PoseInference] Warming up model with 100 dummy frames...");
        {
            torch::NoGradGuard no_grad;
            cv::Mat dummy(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
            for (int i = 0; i < 100; ++i)
            {
                auto inp = preprocess_(dummy);
                module_.forward({inp});
            }
            if (device_.is_cuda())
                torch::cuda::synchronize();
        }
        app::utils::Logger::info("[PoseInference] Warmup complete.");
    } catch (const std::exception& e)
    {
        app::utils::Logger::error(std::string("[PoseInference] Failed to load model: ") + e.what());
        std::exit(EXIT_FAILURE);
    }
}

// preprocess_ — GPU letterbox: BGR→RGB, normalize, aspect-ratio-preserving
//               resize + grey padding, then cast to model dtype
torch::Tensor PoseInference::preprocess_(const cv::Mat& frame) const
{
    // .clone() ensures the tensor owns its data before the async .to(device_) copy
    auto raw = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3}, torch::kByte)
                   .clone()
                   .to(device_);

    // HWC uint8 → NCHW float32 in [0,1], BGR→RGB via channel flip
    auto t = raw.permute({0, 3, 1, 2}).to(torch::kFloat).div_(255.0f).flip(1);

    // Aspect-ratio preserving resize
    float r = std::min(static_cast<float>(input_h_) / frame.rows,
                       static_cast<float>(input_w_) / frame.cols);
    int new_h = static_cast<int>(std::round(frame.rows * r));
    int new_w = static_cast<int>(std::round(frame.cols * r));

    // Use PyTorch's native interpolation (GPU-accelerated if on CUDA) for resizing
    auto resized =
        torch::nn::functional::interpolate(t, torch::nn::functional::InterpolateFuncOptions()
                                                  .size(std::vector<int64_t>{new_h, new_w})
                                                  .mode(torch::enumtype::kBilinear{})
                                                  .align_corners(false));

    // Symmetric grey padding to reach input_h_ × input_w_
    int dw = (input_w_ - new_w) / 2;
    int dh = (input_h_ - new_h) / 2;
    int pad_r = input_w_ - new_w - dw;
    int pad_b = input_h_ - new_h - dh;

    auto padded = torch::nn::functional::pad(
        resized,
        torch::nn::functional::PadFuncOptions({dw, pad_r, dh, pad_b}).value(114.0f / 255.0f));

    // Cast to model dtype (FP16/BF16/FP32)
    return padded.to(model_elem_dtype_);
}

// decode_and_nms_ — GPU confidence filter → CPU decode → float NMS → unscale

void PoseInference::decode_and_nms_(const torch::Tensor& raw, int orig_w, int orig_h,
                                    std::vector<std::vector<std::vector<float>>>& out)
{
    out.clear();

    // Recompute letterbox params (must match preprocess_ exactly)
    float r =
        std::min(static_cast<float>(input_h_) / orig_h, static_cast<float>(input_w_) / orig_w);
    int new_w = static_cast<int>(std::round(orig_w * r));
    int new_h = static_cast<int>(std::round(orig_h * r));
    float pad_w = static_cast<float>((input_w_ - new_w) / 2);
    float pad_h = static_cast<float>((input_h_ - new_h) / 2);

    // Squeeze batch + transpose to [8400, 56]
    torch::Tensor t = raw.squeeze(0).transpose(0, 1);

    // GPU boolean mask on obj_conf (col 4) — only survivors cross PCIe
    torch::Tensor filtered = t.index({t.select(1, 4).gt(conf_thresh_)}).cpu().contiguous();

    if (filtered.size(0) == 0)
        return;

    const int rows = static_cast<int>(filtered.size(0));
    const int cols = static_cast<int>(filtered.size(1));
    const float* data = filtered.data_ptr<float>();

    // Use Rect2d (float) so NMS IoU is accurate in letterboxed 640×640 space
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<std::vector<float>> raw_kpts;

    boxes.reserve(rows);
    scores.reserve(rows);
    raw_kpts.reserve(rows);

    for (int i = 0; i < rows; ++i)
    {
        const float* row = data + i * cols;
        float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
        boxes.emplace_back(cx - bw * 0.5, cy - bh * 0.5, bw, bh);
        scores.push_back(row[4]);

        // 17 keypoints × [x, y, conf] packed at offset 5
        std::vector<float> kp(17 * 3);
        for (int k = 0; k < 17 * 3; ++k)
            kp[k] = row[5 + k];
        raw_kpts.push_back(std::move(kp));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh_, iou_thresh_, indices);

    out.reserve(indices.size());
    for (int idx : indices)
    {
        const auto& kp = raw_kpts[idx];
        std::vector<std::vector<float>> person;
        person.reserve(17);
        for (int i = 0; i < 17; ++i)
        {
            float kx = kp[i * 3 + 0];
            float ky = kp[i * 3 + 1];
            float kc = kp[i * 3 + 2];
            // Unscale from letterboxed space → original frame coords
            person.push_back({(kx - pad_w) / r, (ky - pad_h) / r, kc});
        }
        out.push_back(std::move(person));
    }
}

// detect — Redis passthrough OR GPU inference pipeline

std::vector<std::vector<std::vector<float>>> PoseInference::detect(
    const cv::Mat& frame, const std::vector<std::vector<std::vector<float>>>& from_redis)
{
    //   if CLIENT_TYPE == "redis": return redis detections (empty list if none)

    const auto& cfg = app::config::AppConfig::getInstance();
    if (cfg.client_type == "redis")
    {
        return from_redis;  // empty if Redis had no data this frame
    }
    if (frame.empty())
        return {};

    const bool log_timing = (std::getenv("INFERENCE_TIMING_BREAKDOWN") != nullptr);

    try
    {
        torch::NoGradGuard no_grad;

        // --- Preprocess ---
        auto t0 = std::chrono::high_resolution_clock::now();
        auto input = preprocess_(frame);
        if (device_.is_cuda())
            torch::cuda::synchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        // --- Inference ---
        auto raw_out = module_.forward({input});
        if (device_.is_cuda())
            torch::cuda::synchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        torch::Tensor pred;
        if (raw_out.isTuple())
            pred = raw_out.toTuple()->elements()[0].toTensor();
        else
            pred = raw_out.toTensor();

        // Always decode in FP32 regardless of model dtype
        if (pred.scalar_type() != c10::ScalarType::Float)
            pred = pred.to(c10::ScalarType::Float);

        // --- Decode + NMS ---
        std::vector<std::vector<std::vector<float>>> results;
        decode_and_nms_(pred, frame.cols, frame.rows, results);
        auto t3 = std::chrono::high_resolution_clock::now();

        if (log_timing)
        {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            app::utils::Logger::info("[PoseInference] preprocess=" + std::to_string(ms(t0, t1)) +
                                     "ms  infer=" + std::to_string(ms(t1, t2)) +
                                     "ms  postprocess=" + std::to_string(ms(t2, t3)) + "ms");
        }

        return results;
    } catch (const std::exception& e)
    {
        app::utils::Logger::error(std::string("[PoseInference] Inference error: ") + e.what());
        return {};
    }
}

}  // namespace app::core::inferences