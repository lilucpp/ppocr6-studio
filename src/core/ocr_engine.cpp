#include "ocr_engine.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace {

#ifdef _WIN32
std::wstring to_ort_path(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}
#else
const std::string& to_ort_path(const std::string& s) {
    return s;
}
#endif

} // namespace

OcrEngine::OcrEngine() = default;
OcrEngine::~OcrEngine() = default;

std::string OcrEngine::init(const std::string& model_dir,
                            const std::string& model_size,
                            int intra_op_threads,
                            bool enable_cpu_mem_arena) {
    try {
        initialized_ = false;
        char_dict_.clear();

        Ort::SessionOptions session_opts;
        session_opts.SetInterOpNumThreads(1);
        session_opts.SetIntraOpNumThreads(std::max(1, intra_op_threads));
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        cv::setNumThreads(std::max(1, intra_op_threads));
        if (enable_cpu_mem_arena) {
            session_opts.EnableCpuMemArena();
        } else {
            session_opts.DisableCpuMemArena();
        }

        std::string prefix    = model_dir + "/PP-OCRv6_" + model_size;
        std::string det_model = prefix + "_det_onnx/inference.onnx";
        std::string rec_model = prefix + "_rec_onnx/inference.onnx";
        std::string char_file = prefix + "_rec_onnx/chars.txt";

        for (const auto& p : {det_model, rec_model, char_file}) {
            if (!std::filesystem::exists(p)) {
                return "Missing model file: " + p +
                       "\nRun: python3 scripts/download_models.py --model " + model_size;
            }
        }

        det_session_ = std::make_unique<Ort::Session>(
            env_, to_ort_path(det_model).c_str(), session_opts);
        rec_session_ = std::make_unique<Ort::Session>(
            env_, to_ort_path(rec_model).c_str(), session_opts);

        std::ifstream ifs(char_file);
        if (!ifs.is_open()) return "Cannot open " + char_file;
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            char_dict_.push_back(line);
        }
        initialized_ = true;
        return {};
    } catch (const std::exception& e) {
        return std::string("Init failed: ") + e.what();
    }
}

// ────────────────── Image Normalization & Detection ──────────────────

cv::Mat OcrEngine::normalize_bgr(const cv::Mat& input_img) {
    cv::Mat src8;
    if (input_img.depth() == CV_8U) {
        src8 = input_img;
    } else if (input_img.depth() == CV_16U) {
        input_img.convertTo(src8, CV_8U, 1.0 / 256.0);
    } else {
        double min_v = 0.0, max_v = 0.0;
        cv::minMaxLoc(input_img.reshape(1), &min_v, &max_v);
        double scale = (max_v <= 1.0 + 1e-3) ? 255.0 : 1.0;
        input_img.convertTo(src8, CV_8U, scale);
    }

    if (src8.channels() == 3) {
        return src8;
    }
    if (src8.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(src8, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (src8.channels() == 4) {
        // Composite BGRA onto white background so transparent-bg dark text remains legible
        cv::Mat bgr(src8.rows, src8.cols, CV_8UC3);
        for (int y = 0; y < src8.rows; ++y) {
            const cv::Vec4b* sp = src8.ptr<cv::Vec4b>(y);
            cv::Vec3b* dp = bgr.ptr<cv::Vec3b>(y);
            for (int x = 0; x < src8.cols; ++x) {
                int a = sp[x][3];
                int inv_a = 255 - a;
                dp[x][0] = static_cast<uchar>((sp[x][0] * a + 255 * inv_a + 127) / 255);
                dp[x][1] = static_cast<uchar>((sp[x][1] * a + 255 * inv_a + 127) / 255);
                dp[x][2] = static_cast<uchar>((sp[x][2] * a + 255 * inv_a + 127) / 255);
            }
        }
        return bgr;
    }
    cv::Mat bgr;
    cv::cvtColor(src8, bgr, cv::COLOR_BGRA2BGR);
    return bgr;
}

// Mirrors PaddleX DetResizeForTest (resize_image_type0): scale according to
// limit_type ("min" | "max" | "resize_long") and limit_side_len, clamp long
// side to max_side_limit, then round each side to a multiple of `align` (>= align).
cv::Size OcrEngine::det_resize(cv::Mat& img, const std::string& limit_type,
                               int limit_side_len, int max_side_limit,
                               int align, cv::Size& scaled_size) {
    const int h = img.rows, w = img.cols;
    cv::Size pre_resize(w, h);

    double ratio = 1.0;
    if (limit_type == "max") {
        int max_side = (std::max)(h, w);
        if (max_side > limit_side_len) {
            ratio = double(limit_side_len) / max_side;
        }
    } else if (limit_type == "resize_long") {
        ratio = double(limit_side_len) / (std::max)(h, w);
    } else {
        int min_side = (std::min)(h, w);
        if (min_side < limit_side_len) {
            ratio = double(limit_side_len) / min_side;
        }
    }
    int rh = int(h * ratio), rw = int(w * ratio);
    if (max_side_limit > 0 && (std::max)(rh, rw) > max_side_limit) {
        double r = double(max_side_limit) / (std::max)(rh, rw);
        rh = int(rh * r);
        rw = int(rw * r);
    }
    auto round_align = [align](int v) {
        return (std::max)(int(std::nearbyint(double(v) / align) * align), align);
    };
    rh = round_align(rh);
    rw = round_align(rw);

    if (rh != h || rw != w) {
        cv::resize(img, img, cv::Size(rw, rh));
    }
    scaled_size = cv::Size(rw, rh);
    return pre_resize;
}

std::vector<float> OcrEngine::det_preprocess(const cv::Mat& img_bgr) {
    cv::Mat rgb;
    cv::cvtColor(img_bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);
    for (int c = 0; c < 3; ++c) ch[c] = (ch[c] - mean[c]) / stdv[c];
    int h = ch[0].rows, w = ch[0].cols;
    std::vector<float> data(3 * h * w);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                data[c * h * w + y * w + x] = ch[c].at<float>(y, x);
    return data;
}

std::vector<OcrEngine::Box> OcrEngine::det_postprocess(
    const float* prob, int ph, int pw, const cv::Size& orig_size,
    const cv::Size& pre_resize_size, const cv::Size& scaled_size,
    const cv::Size& padded_size, float thresh, float box_thresh,
    float unclip_ratio) {
    cv::Mat prob_mat(ph, pw, CV_32F, (void*)prob);
    cv::Mat mask;
    cv::threshold(prob_mat, mask, thresh, 1.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    float sh = (float(padded_size.height) / float(ph)) *
               (float(pre_resize_size.height) / float(scaled_size.height));
    float sw = (float(padded_size.width) / float(pw)) *
               (float(pre_resize_size.width) / float(scaled_size.width));
    std::vector<Box> boxes;

    for (const auto& c : contours) {
        if (c.size() < 4) continue;
        double eps = 0.002 * cv::arcLength(c, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, eps, true);
        if (approx.size() < 4) continue;

        cv::Mat cm = cv::Mat::zeros(ph, pw, CV_8U);
        cv::fillPoly(cm, std::vector<std::vector<cv::Point>>{c}, cv::Scalar(255));
        float score = (float)cv::mean(prob_mat, cm)[0];
        if (score < box_thresh) continue;

        double area = cv::contourArea(approx), perim = cv::arcLength(approx, true);
        if (perim <= 0.0) continue;
        double dist = area * unclip_ratio / perim;
        cv::Mat pm = cv::Mat::zeros(ph, pw, CV_8U);
        cv::drawContours(pm, std::vector<std::vector<cv::Point>>{approx}, 0, cv::Scalar(255), cv::FILLED);
        int ks = std::max(1, (int)std::ceil(dist));
        cv::dilate(pm, pm, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ks, ks)));
        std::vector<std::vector<cv::Point>> dc;
        cv::findContours(pm, dc, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (dc.empty()) continue;
        auto& uc = *std::max_element(dc.begin(), dc.end(),
            [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });

        cv::RotatedRect rr = cv::minAreaRect(uc);
        std::vector<cv::Point2f> pts(4);
        rr.points(pts.data());
        std::sort(pts.begin(), pts.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y - 1e-3f || (std::abs(a.y - b.y) < 1e-3f && a.x < b.x);
        });
        if (pts[0].x > pts[1].x) std::swap(pts[0], pts[1]);
        if (pts[2].x > pts[3].x) std::swap(pts[2], pts[3]);
        for (auto& p : pts) {
            p.x = std::clamp(p.x * sw, 0.0f, float(orig_size.width));
            p.y = std::clamp(p.y * sh, 0.0f, float(orig_size.height));
        }
        boxes.push_back({pts, score});
    }
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return (std::min)(a.points[0].y, a.points[1].y) < (std::min)(b.points[0].y, b.points[1].y);
    });
    return boxes;
}

// ────────────────── Merge ──────────────────

std::vector<OcrEngine::Box> OcrEngine::merge_adjacent_boxes(
    const std::vector<Box>& boxes, float ov_ratio, float h_ratio, float gap_px) {
    if (boxes.size() <= 1) return boxes;
    struct R { float l, t, r, b; };
    std::vector<R> rs(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        rs[i] = {FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (auto& p : boxes[i].points) {
            rs[i].l = (std::min)(rs[i].l, p.x); rs[i].t = (std::min)(rs[i].t, p.y);
            rs[i].r = (std::max)(rs[i].r, p.x); rs[i].b = (std::max)(rs[i].b, p.y);
        }
    }
    std::vector<Box> merged;
    std::vector<bool> used(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (used[i]) continue;
        std::vector<size_t> grp = {i}; used[i] = true;
        bool chg;
        do {
            chg = false;
            float gl = rs[grp[0]].l, gt = rs[grp[0]].t, gr = rs[grp[0]].r, gb = rs[grp[0]].b;
            for (auto k : grp) {
                gl = (std::min)(gl, rs[k].l); gt = (std::min)(gt, rs[k].t);
                gr = (std::max)(gr, rs[k].r); gb = (std::max)(gb, rs[k].b);
            }
            float gh = gb - gt;
            for (size_t j = 0; j < boxes.size(); ++j) {
                if (used[j]) continue;
                float ov = (std::min)(gb, rs[j].b) - (std::max)(gt, rs[j].t);
                if (ov <= 0) continue;
                float jh = rs[j].b - rs[j].t;
                if (ov / (std::min)(gh, jh) < ov_ratio) continue;
                if ((std::max)(gh, jh) / (std::min)(gh, jh) > h_ratio) continue;
                float gap = 0;
                if (rs[j].l > gr) gap = rs[j].l - gr; else if (gl > rs[j].r) gap = gl - rs[j].r;
                if (gap > gap_px) continue;
                grp.push_back(j); used[j] = true; chg = true; break;
            }
        } while (chg);
        std::sort(grp.begin(), grp.end(), [&](size_t a, size_t b) { return rs[a].l < rs[b].l; });
        std::vector<cv::Point2f> all;
        float ms = 0;
        for (auto k : grp) {
            for (auto& p : boxes[k].points) all.push_back(p);
            ms = (std::max)(ms, boxes[k].score);
        }
        cv::RotatedRect mr = cv::minAreaRect(all);
        std::vector<cv::Point2f> mp(4); mr.points(mp.data());
        std::sort(mp.begin(), mp.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.y < b.y - 1e-3f || (std::abs(a.y - b.y) < 1e-3f && a.x < b.x);
        });
        if (mp[0].x > mp[1].x) std::swap(mp[0], mp[1]);
        if (mp[2].x > mp[3].x) std::swap(mp[2], mp[3]);
        merged.push_back({mp, ms});
    }
    std::sort(merged.begin(), merged.end(), [](const Box& a, const Box& b) {
        return (std::min)(a.points[0].y, a.points[1].y) < (std::min)(b.points[0].y, b.points[1].y);
    });
    return merged;
}

// ────────────────── Recognition ──────────────────

cv::Mat OcrEngine::get_rotate_crop(const cv::Mat& img, const std::vector<cv::Point2f>& pts,
                                   int pl, int pt) {
    float w1 = (float)cv::norm(pts[1] - pts[0]), w2 = (float)cv::norm(pts[3] - pts[2]);
    float dw = (std::max)(w1, w2);
    float h1 = (float)cv::norm(pts[2] - pts[0]), h2 = (float)cv::norm(pts[3] - pts[1]);
    float dh = (std::max)(h1, h2);
    int iw = (std::max)(int(dw), 1), ih = (std::max)(int(dh), 1);
    std::vector<cv::Point2f> src = {pts[0], pts[1], pts[2], pts[3]};
    std::vector<cv::Point2f> dst = {
        {float(pl), float(pt)}, {float(iw - 1 + pl), float(pt)},
        {float(pl), float(ih - 1 + pt)}, {float(iw - 1 + pl), float(ih - 1 + pt)}};
    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat crop;
    cv::warpPerspective(img, crop, M, cv::Size(iw + 2 * pl, ih + 2 * pt),
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return crop;
}

std::vector<float> OcrEngine::rec_preprocess(const cv::Mat& crop, int th, int tw, int mw) {
    int h = crop.rows, w = crop.cols;
    float wr = float(w) / float(h);
    float mwr = (std::max)(float(tw) / float(th), wr);
    int img_w = int(th * mwr);
    if (img_w > mw) img_w = mw;
    int rw = int(std::ceil(th * wr)) > img_w ? img_w : int(std::ceil(th * wr));
    cv::Mat img;
    cv::resize(crop, img, cv::Size(rw, th));
    std::vector<float> data(3 * th * img_w, 0.f);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < th; ++y)
            for (int x = 0; x < rw; ++x)
                data[c * th * img_w + y * img_w + x] = (img.at<cv::Vec3b>(y, x)[c] / 255.0f - 0.5f) / 0.5f;
    return data;
}

std::string OcrEngine::ctc_decode(const float* logits, int T, int C,
                                  const std::vector<std::string>& dict, float* conf) {
    std::string result;
    int prev = -1;
    float cs = 0; int cc = 0;
    for (int t = 0; t < T; ++t) {
        int mc = 0; float mv = logits[t * C];
        for (int c = 1; c < C; ++c) { if (logits[t * C + c] > mv) { mv = logits[t * C + c]; mc = c; } }
        if (mc == 0) { prev = -1; continue; }
        if (mc == prev) continue;
        if (mc >= (int)dict.size()) continue;
        result += dict[mc]; cs += mv; cc++; prev = mc;
    }
    if (conf) *conf = cc > 0 ? cs / cc : 0.f;
    return result;
}

// ────────────────── Main Pipeline ──────────────────

OcrResult OcrEngine::run(const std::string& image_path, const OcrParams& params) const {
    OcrResult res;
    res.num_chars = (int)char_dict_.size();
    if (!initialized_) { res.error = "Engine not initialized."; return res; }
    cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) { res.error = "Cannot read image: " + image_path; return res; }
    std::string filename = std::filesystem::path(image_path).filename().string();
    return run(img, params, filename);
}

OcrResult OcrEngine::run(const cv::Mat& input_img, const OcrParams& params,
                         const std::string& debug_name) const {
    OcrResult res;
    res.num_chars = (int)char_dict_.size();
    if (!initialized_) { res.error = "Engine not initialized."; return res; }
    if (input_img.empty()) { res.error = "Input image is empty."; return res; }
    try {
        auto t0 = std::chrono::steady_clock::now();
        cv::Mat img = normalize_bgr(input_img);
        const cv::Size orig = img.size();

        // Official behaviour: very small images (h + w < 64) are zero-padded
        // (top-left) to at least 32x32 before detection.
        if (img.rows + img.cols < 64) {
            cv::Mat padded = cv::Mat::zeros((std::max)(32, img.rows),
                                            (std::max)(32, img.cols), CV_8UC3);
            img.copyTo(padded(cv::Rect(0, 0, img.cols, img.rows)));
            img = padded;
        }

        const char* in_name = "x", *out_name = "fetch_name_0";

        // Detection
        auto td = std::chrono::steady_clock::now();
        cv::Mat di = img.clone();
        cv::Size scaled_size;
        cv::Size pre_resize_size = det_resize(di, params.det_limit_type,
                                              params.limit_side_len, params.max_side_limit,
                                              params.align, scaled_size);
        cv::Size padded_size(di.cols, di.rows);
        auto inp = det_preprocess(di);
        std::vector<int64_t> dsh = {1, 3, di.rows, di.cols};
        Ort::Value dt = Ort::Value::CreateTensor<float>(mem_info_, inp.data(), inp.size(), dsh.data(), dsh.size());
        auto dout = det_session_->Run(Ort::RunOptions{}, &in_name, &dt, 1, &out_name, 1);
        auto dsp = dout[0].GetTensorTypeAndShapeInfo().GetShape();
        auto boxes = det_postprocess(dout[0].GetTensorMutableData<float>(),
                                     (int)dsp[2], (int)dsp[3], orig,
                                     pre_resize_size, scaled_size, padded_size,
                                     params.det_thresh, params.det_box_thresh, params.unclip_ratio);
        res.raw_boxes = (int)boxes.size();
        res.det_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();

        // Merge
        auto tm = std::chrono::steady_clock::now();
        boxes = merge_adjacent_boxes(boxes, params.merge_overlap_ratio, params.merge_height_ratio, params.merge_gap_px);
        res.merged_boxes = (int)boxes.size();
        res.merge_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tm).count();

        // Optional Debug Visualization
        if (params.save_debug_image) {
            cv::Mat debug_img = img.clone();
            for (size_t i = 0; i < boxes.size(); ++i) {
                std::vector<cv::Point> pts;
                for (const auto& p : boxes[i].points)
                    pts.emplace_back(int(p.x), int(p.y));
                cv::polylines(debug_img, pts, true, cv::Scalar(0, 255, 0), 2);
                cv::putText(debug_img, std::to_string(i),
                            cv::Point(int(boxes[i].points[0].x), int(boxes[i].points[0].y) - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            std::string filename = std::filesystem::path(debug_name).filename().string();
            std::string out_path = "det_boxes_" + (filename.empty() ? "image.jpg" : filename);
            if (cv::imwrite(out_path, debug_img)) {
                res.debug_image_path = out_path;
            }
        }

        // Recognition
        auto tr = std::chrono::steady_clock::now();
        for (auto& box : boxes) {
            cv::Mat crop = get_rotate_crop(img, box.points, params.pad_lr, params.pad_tb);
            if (crop.empty() || crop.rows < 2 || crop.cols < 2) continue;
            auto ri = rec_preprocess(crop, params.rec_h, params.rec_w, params.rec_max_w);
            int aw = (int)ri.size() / (3 * params.rec_h);
            std::vector<int64_t> rsh = {1, 3, params.rec_h, aw};
            Ort::Value rt = Ort::Value::CreateTensor<float>(mem_info_, ri.data(), ri.size(), rsh.data(), rsh.size());
            auto rout = rec_session_->Run(Ort::RunOptions{}, &in_name, &rt, 1, &out_name, 1);
            auto rsp = rout[0].GetTensorTypeAndShapeInfo().GetShape();
            float conf = 0;
            auto text = ctc_decode(rout[0].GetTensorMutableData<float>(),
                                   (int)rsp[1], (int)rsp[2], char_dict_, &conf);
            if (text.empty() ||
                (params.rec_conf_threshold > 0.0f && conf < params.rec_conf_threshold)) {
                continue;
            }
            float minx = FLT_MAX, miny = FLT_MAX, maxx = -FLT_MAX, maxy = -FLT_MAX;
            for (auto& p : box.points) {
                minx = (std::min)(minx, p.x); miny = (std::min)(miny, p.y);
                maxx = (std::max)(maxx, p.x); maxy = (std::max)(maxy, p.y);
            }
            res.lines.push_back({text, conf, minx, miny, maxx, maxy});
        }
        res.rec_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count();
        res.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    } catch (const std::exception& e) {
        res.error = std::string("Error: ") + e.what();
    }
    return res;
}
