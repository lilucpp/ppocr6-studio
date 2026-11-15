#ifndef PARAMS_H
#define PARAMS_H

#include <sstream>
#include <string>

/// All tunable parameters for the OCR pipeline.
struct OcrParams {
    // Model
    std::string model_size = "tiny";

    // Detection (aligned with official PaddleOCR / PaddleX DetResizeForTest)
    float       det_thresh       = 0.30f;   // official PP-OCRv6 default
    float       det_box_thresh   = 0.60f;   // official PP-OCRv6 default
    float       unclip_ratio     = 1.50f;   // official PP-OCRv6 default
    std::string det_limit_type   = "min";   // "min" | "max" | "resize_long"
    int         limit_side_len   = 64;      // official PP-OCRv6 OCR default
    int         max_side_limit   = 4000;    // official PaddleX DetResizeForTest default
    int         align            = 32;

    // Box Merge
    float merge_overlap_ratio = 0.50f;
    float merge_height_ratio  = 2.50f;
    float merge_gap_px        = 30.0f;

    // Crop padding
    int   pad_lr = 3;
    int   pad_tb = 3;

    // Recognition
    int   rec_h      = 48;
    int   rec_w      = 320;
    int   rec_max_w  = 3200;

    // Output
    float rec_conf_threshold = 0.0f;  // official PP-OCRv6 OCR default (score_thresh=0.0)
    bool  save_debug_image   = false;

    std::string to_string() const {
        std::ostringstream os;
        os << "model=" << model_size
           << " det_thr=" << det_thresh
           << " box_thr=" << det_box_thresh
           << " unclip=" << unclip_ratio
           << " limit=" << det_limit_type << ":" << limit_side_len << "/" << max_side_limit
           << " ov=" << merge_overlap_ratio
           << " gap=" << merge_gap_px
           << " pad=" << pad_lr
           << " conf=" << rec_conf_threshold;
        return os.str();
    }
};

#endif // PARAMS_H
