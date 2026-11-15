#ifndef OCR_ENGINE_H
#define OCR_ENGINE_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

#include "params.h"

struct OcrResultLine {
    std::string text;
    float       confidence;
    float       box_x1, box_y1, box_x2, box_y2;
};

struct OcrResult {
    std::vector<OcrResultLine> lines;
    double det_ms = 0, merge_ms = 0, rec_ms = 0, total_ms = 0;
    int    raw_boxes = 0, merged_boxes = 0;
    int    num_chars = 0;
    std::string debug_image_path;
    std::string error;
};

class OcrEngine {
public:
    OcrEngine();
    ~OcrEngine();

    std::string init(const std::string& model_dir,
                     const std::string& model_size,
                     int intra_op_threads = 4,
                     bool enable_cpu_mem_arena = true);
    OcrResult run(const std::string& image_path, const OcrParams& params) const;
    OcrResult run(const cv::Mat& image_bgr, const OcrParams& params,
                  const std::string& debug_name = "image.jpg") const;
    bool is_initialized() const { return initialized_; }

private:
    // ONNX Runtime
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ppocr6"};
    Ort::MemoryInfo mem_info_{Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)};
    std::unique_ptr<Ort::Session> det_session_, rec_session_;
    std::vector<std::string> char_dict_;
    bool initialized_ = false;

    // Internal types
    struct Box {
        std::vector<cv::Point2f> points;
        float score;
    };

    // Pipeline steps (stateless & thread-safe)
    static cv::Mat normalize_bgr(const cv::Mat& input_img);
    static cv::Size det_resize(cv::Mat& img, const std::string& limit_type,
                               int limit_side_len, int max_side_limit,
                               int align, cv::Size& scaled_size);
    static std::vector<float> det_preprocess(const cv::Mat& img_bgr);
    static std::vector<Box> det_postprocess(const float* prob, int ph, int pw,
                                            const cv::Size& orig_size,
                                            const cv::Size& pre_resize_size,
                                            const cv::Size& scaled_size,
                                            const cv::Size& padded_size,
                                            float thresh, float box_thresh, float unclip_ratio);
    static std::vector<Box> merge_adjacent_boxes(const std::vector<Box>& boxes,
                                                 float overlap_ratio, float height_ratio, float gap_px);
    static cv::Mat get_rotate_crop(const cv::Mat& img, const std::vector<cv::Point2f>& pts,
                                   int pad_lr, int pad_tb);
    static std::vector<float> rec_preprocess(const cv::Mat& crop_bgr, int target_h, int target_w, int max_img_w);
    static std::string ctc_decode(const float* logits, int T, int C,
                                  const std::vector<std::string>& char_dict, float* out_conf);
};

#endif // OCR_ENGINE_H
