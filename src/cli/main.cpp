#include <chrono>
#include <iostream>
#include <string>

#include "ocr_engine.h"
#include "params.h"

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <image_path> <tiny|small|medium> [options]\n"
              << "Options:\n"
              << "  --model-dir <dir>         Model root directory (default: models)\n"
              << "  --threads <N>             ONNX Runtime intra-op threads (default: 4)\n"
              << "  --det-limit-type <T>      Resize mode: min|max|resize_long (default: min)\n"
              << "  --det-limit-side-len <N>  Detection resize target side len (default: 64)\n"
              << "  --max-side-limit <N>      Detection long-side cap (default: 4000)\n"
              << "  --debug                   Save det_boxes_<name> visualization\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_size = argv[2];
    std::string model_dir = "models";
    int threads = 4;
    int det_limit_side_len = -1;
    int max_side_limit = -1;
    std::string det_limit_type;
    bool save_debug = false;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            save_debug = true;
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--det-limit-type" && i + 1 < argc) {
            det_limit_type = argv[++i];
        } else if (arg == "--det-limit-side-len" && i + 1 < argc) {
            det_limit_side_len = std::stoi(argv[++i]);
        } else if (arg == "--max-side-limit" && i + 1 < argc) {
            max_side_limit = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_size != "tiny" && model_size != "small" && model_size != "medium") {
        std::cerr << "model_size must be tiny, small, or medium\n";
        return 1;
    }

    OcrParams params;
    params.model_size = model_size;
    params.save_debug_image = save_debug;
    if (!det_limit_type.empty()) params.det_limit_type = det_limit_type;
    if (det_limit_side_len > 0) params.limit_side_len = det_limit_side_len;
    if (max_side_limit > 0) params.max_side_limit = max_side_limit;
    if (model_size == "small") {
        params.det_box_thresh = 0.60f;
    } else if (model_size == "medium") {
        params.det_box_thresh = 0.62f;
    }

    auto t_init0 = std::chrono::steady_clock::now();
    OcrEngine engine;
    std::string err = engine.init(model_dir, model_size, threads);
    if (!err.empty()) {
        std::cerr << err << "\n";
        return 1;
    }
    double init_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_init0).count();

    OcrResult res = engine.run(image_path, params);
    if (!res.error.empty()) {
        std::cerr << res.error << "\n";
        return 1;
    }

    std::cout << "Loaded " << res.num_chars << " characters (including blank)\n";
    std::cout << "Init: " << init_ms << " ms\n";
    std::cout << "Detection: " << res.raw_boxes << " boxes, " << res.det_ms << " ms\n";
    std::cout << "Merge: " << res.merged_boxes << " boxes\n";
    if (!res.debug_image_path.empty()) {
        std::cout << "Debug image saved: " << res.debug_image_path << "\n";
    }
    for (size_t i = 0; i < res.lines.size(); ++i) {
        std::cout << "  [" << i << "] " << res.lines[i].text << "\n";
    }
    std::cout << "\nRecognition: " << res.rec_ms << " ms\n";
    std::cout << "Total: " << res.total_ms << " ms\n";
    return 0;
}
