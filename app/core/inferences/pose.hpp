// Interface for TorchScript YOLO-pose inference.
// Declares PoseInference: loads a .torchscript model, preprocesses frames via GPU letterbox,
// runs inference, and returns 17 keypoints per detected person as (x, y, visibility) vectors.

#pragma once

#include <c10/core/ScalarType.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <torch/script.h>

namespace app::core::inferences {

// YOLO pose (56-dim rows: 4 box, obj, skip1, 51 kp packed) — decode matches centralized-yolo TensorRT path.
class PoseInference {
public:
    explicit PoseInference(const std::string& model_path);

    /** Each person: 17 keypoints × (x, y, visibility). */
    std::vector<std::vector<std::vector<float>>> detect(
        const cv::Mat& frame,
        const std::vector<std::vector<std::vector<float>>>& from_redis);

private:
    torch::jit::script::Module module_;
    torch::Device device_;
    /** First floating parameter dtype (FP16 TorchScript → Half); FP32 preprocessing is cast to this. */
    c10::ScalarType model_elem_dtype_{c10::ScalarType::Float};
    float conf_thresh_ = 0.25f;
    float iou_thresh_ = 0.45f;
    int input_w_ = 640;
    int input_h_ = 640;

    torch::Tensor preprocess_(const cv::Mat& frame) const;
    void decode_and_nms_(const torch::Tensor& raw, int orig_w, int orig_h,
                         std::vector<std::vector<std::vector<float>>>& out);
};

}  // namespace app::core::inferences
