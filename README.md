# ppocr6-studio — PP-OCRv6 纯 C++ 推理与调参套件

> 基于 PaddleOCR **PP-OCRv6** 的纯 C++17 离线 OCR 工程，提供 **命令行推理工具**（`ppocr6`）和可选的 **Qt5 可视化调参工具**（`ppocr6-tuner`）。  
> 不包含预置模型与庞大二进制依赖：模型通过脚本按需动态拉取，第三方库支持 `apt` / `vcpkg` 安装或由 CMake 自动拉取。

---

## 项目背景

PaddleOCR 官方 Python 版本（`paddleocr`）虽然使用方便，但在 C++ 桌面端或嵌入式部署中存在两个痛点：

1. **启动与运行开销** — Python 运行时与框架边界切换开销大
2. **集成困难** — 若要在 C++ 应用（Qt / WPF / MFC 等）中嵌入 OCR，需要额外的进程通信

## 核心优势

1. **零 Python 运行时、进程内直接集成**
   - 彻底摆脱 Python 解释器与 HTTP/gRPC/管道等跨进程通信（IPC）开销，无序列化与进程管理负担。
   - 核心引擎封装为独立静态库 **`ppocr6::core`**（`libppocr6_core.a`），不依赖 Qt，可直接嵌入 Qt / MFC / Win32 / Linux 后台服务等任意 C++17 项目；同时支持文件路径与内存 `cv::Mat` 直接推理，`run()` 为 `const` 线程安全接口。
2. **完整的 DBPostProcess 检测后处理**
   - 不同于仅做二值化 + `boundingRect` 的简化 Demo，本项目完整实现了 `approxPolyDP` 轮廓简化、区域概率均值打分（`det_box_thresh`），并用椭圆核形态学膨胀近似 `unclip` 扩边 + `minAreaRect` 旋转框提取，检测框更贴合真实文字边界。
3. **同行文本框合并（`merge_adjacent_boxes`）**
   - 基于垂直重叠率、高度比防误合（避免印章大框与正文小字合并）和水平间距自动拼接同行的断裂文本块，输出符合阅读顺序的整行结果。
4. **透视裁剪垫边与 CTC 置信度输出**
   - 透视变换裁剪时通过 `BORDER_REPLICATE` 四周垫边，防止边缘笔画被裁切；对识别模型输出的 softmax 概率在非 blank、非重复时间步上计算均值置信度并随结果返回，可按需通过 `rec_conf_threshold`（默认 `0.0`）筛选输出。
5. **可视化参数调优工具（`ppocr6-tuner`）**
   - 提供 Qt5 图形界面实时调节全部检测、合并、裁剪与识别阈值并查看各阶段耗时，无需改代码和重编译即可适配发票、票据、印章遮挡、倾斜、手写等场景。

---

## 目录结构

```text
ppocr6-studio/
├── CMakeLists.txt                  # 统一顶层 CMake（默认构建 ppocr6，-DBUILD_GUI=ON 构建 ppocr6-tuner）
├── vcpkg.json                      # vcpkg manifest（opencv4, onnxruntime, 可选 feature: gui）
├── .gitignore
├── README.md
│
├── src/
│   ├── core/                       # 核心 OCR 引擎（纯 C++17，无 Qt 依赖）
│   │   ├── ocr_engine.h
│   │   ├── ocr_engine.cpp
│   │   └── params.h                # 可调参数结构体
│   ├── cli/
│   │   └── main.cpp                # 命令行推理程序 (ppocr6)
│   └── gui/                        # Qt5 参数调优图形界面 (ppocr6-tuner, 可选)
│       ├── main.cpp
│       ├── mainwindow.h
│       └── mainwindow.cpp
│
├── scripts/
│   ├── download_models.py          # 动态拉取 PP-OCRv6 ONNX 模型并自动生成 chars.txt
│   └── test_ocr.py                 # Python 官方 PaddleOCR 参考对比脚本
│
└── models/                         # (.gitignore 忽略) 运行 download_models.py 后生成
```

---

## 快速开始

### 1. 动态拉取 OCR 模型

仓库不再内置模型文件，使用标准库 Python 脚本按需从 Hugging Face（`PaddlePaddle/PP-OCRv6_*_onnx`）拉取，并自动从 `inference.yml` 提取生成 `chars.txt`：

```bash
# 拉取 tiny 模型（默认，约 6MB）
python3 scripts/download_models.py --model tiny

# 拉取指定规格：tiny / small / medium / all
python3 scripts/download_models.py --model all

# 国内网络可使用 hf-mirror 镜像
python3 scripts/download_models.py --model tiny --mirror
```

---

### 2. 安装依赖与编译构建

依赖需求：
- **C++17 编译器**（GCC ≥ 8 / Clang ≥ 7 / MSVC 2019+）
- **CMake** ≥ 3.16
- **OpenCV**（≥ 4.x）
- **ONNX Runtime**（默认由 CMake 自动拉取 1.32.0；也可用 `-DONNXRUNTIME_VERSION=x.y.z` 指定）
- **Qt5**（仅当启用 `-DBUILD_GUI=ON` 时需要）

#### 方式 A：Ubuntu / Debian（`apt` + CMake 自动拉取 ONNX Runtime）

```bash
# 1. 安装 OpenCV 与编译工具（若需编译 GUI 再额外安装 qtbase5-dev）
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
# 可选 GUI 依赖：
# sudo apt install -y qtbase5-dev

# 2. 配置与编译（若本地未安装 ONNX Runtime，CMake 会自动下载对应平台的预编译包）
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

> 若同时构建 Qt5 调参 GUI，添加 `-DBUILD_GUI=ON`：
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=ON
> cmake --build build -j
> ```

#### 方式 B：Windows / Linux 使用 `vcpkg`

项目根目录已提供 [vcpkg.json](vcpkg.json)：

```bash
# 仅构建命令行工具 ppocr6
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

# 同时构建 Qt5 调参 GUI (启用 vcpkg manifest feature: gui)
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake -DVCPKG_MANIFEST_FEATURES=gui -DBUILD_GUI=ON
cmake --build build --config Release
```

#### 方式 C：手动指定本地已安装的库

```bash
cmake -B build \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DOpenCV_DIR=/path/to/opencv/build
cmake --build build --config Release
```

---

## 使用指南

### 1. 命令行推理（`ppocr6`）

```bash
# 单张图片推理（Linux）
./build/ppocr6 <图片路径> tiny|small|medium [--model-dir models] [--threads 4] [--debug]

# Windows
.\build\Release\ppocr6.exe <图片路径> tiny
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `tiny` | 最快，适合标准印刷文档（6905 字符集） |
| `small` | 兼顾速度与精度，可识别部分手写字（18709 字符集） |
| `medium` | ResNet50_vd 骨干网络，精度最高（18709 字符集） |
| `--model-dir` | 指定模型根目录，默认为当前目录下的 `models` |
| `--threads` | ONNX Runtime 算子内线程数（默认 `4`） |
| `--det-limit-type` | 检测缩放策略：`min` / `max` / `resize_long`（默认 `min`，与官方一致） |
| `--det-limit-side-len` | 检测缩放目标边长（默认 `64`，与官方一致） |
| `--max-side-limit` | 检测长边硬上限（默认 `4000`，与官方一致） |
| `--debug` | 保存检测框可视化图 `det_boxes_<图片名>` |

### 2. Qt5 参数调优 GUI（`ppocr6-tuner`）

启用 `-DBUILD_GUI=ON` 编译后运行 `./build/ppocr6-tuner`（Windows 为 `.\build\Release\ppocr6-tuner.exe`）。  
在左侧面板实时调整检测尺寸策略与阈值、膨胀系数、同行合并参数、裁剪垫边与输出置信度阈值，点击 **Run** 即可查看识别结果与各阶段耗时。

### 3. 作为 C++ 库集成到既有项目

项目核心引擎 `ppocr6::core` 无 Qt 依赖，可通过 `add_subdirectory` 直接嵌入现有 CMake 工程：

```cmake
add_subdirectory(ppocr6-studio)
target_link_libraries(your_app PRIVATE ppocr6::core)
```

C++ 调用示例（支持文件路径或内存 `cv::Mat`，`run()` 为线程安全 `const` 方法）：

```cpp
#include "ocr_engine.h"

OcrEngine engine;
// 参数 3: intra_op_threads (单核模式传 1，多核高性能传 4+)
// 参数 4: enable_cpu_mem_arena (低内存模式可传 false)
std::string err = engine.init("models", "tiny", /*intra_op_threads=*/4, /*enable_cpu_mem_arena=*/true);
if (err.empty()) {
    OcrParams params;
    cv::Mat bgr = cv::imread("0.jpg");
    OcrResult res = engine.run(bgr, params); // 可被多线程并发调用
}
```

### 4. Python 官方参考实现（可选）

用于将 C++ 推理结果与 PaddleOCR 官方 Python 管线进行对比验证：

```bash
pip install paddleocr onnxruntime opencv-contrib-python
python3 scripts/test_ocr.py <图片路径> --model tiny|small|medium
```

---

## 完整参数速查

### 预处理与检测阶段

| 参数 | 默认 | 范围 | 作用 | 调大 | 调小 |
|------|------|------|------|------|------|
| `det_limit_type` | `"min"`（官方） | `min` / `max` / `resize_long` | 检测前缩放策略（对应 PaddleX `DetResizeForTest`） | — | — |
| `limit_side_len` | 64（与官方一致） | 32~2048 | `min` 模式下短边小于该值才放大；`max` 模式下长边超过该值才缩小 | 小图放大更多 | 基本按原图尺寸送入 |
| `max_side_limit` | 4000（与官方一致） | 256~8192 | 长边硬上限，防极端长条图/超大分辨率 OOM | 保留高分辨率细节 | 更省内存更快 |
| `det_thresh` | 0.30（官方） | 0.05~0.50 | 概率图二值化门槛 | 框变少 | 框变多 |
| `det_box_thresh` | 0.60（官方） | 0.20~0.90 | 检测框置信度门槛 | 噪声减少 | 碎片增多 |
| `unclip_ratio` | 1.5（官方） | 1.0~4.0 | 框膨胀系数 | 覆盖更广 | 紧贴文字 |

### 合并阶段

| 参数 | 默认 | 范围 | 作用 |
|------|------|------|------|
| `merge_overlap_ratio` | 0.50 | 0.1~0.9 | 同行垂直重叠比例门槛 |
| `merge_height_ratio` | 2.5 | 1.5~5.0 | 防止印章框与文字框误合 |
| `merge_gap_px` | 30px | 5~100 | 同行水平最大间隔 |

### 识别与输出阈值

| 参数 | 默认 | 范围 | 作用 |
|------|------|------|------|
| `pad_lr` / `pad_tb` | 3px | 0~20 | 裁剪垫边，防止切字 |
| `rec_conf_threshold` | 0.00（与官方一致） | 0.00~0.99 | 输出置信度下限（`0.0` 不丢弃任何非空识别行；调高可剔除低置信度结果） |

### 典型场景参数参考

| 场景 | det_thresh | box_thresh | unclip | overlap | gap | conf |
|------|:----------:|:----------:|:------:|:-------:|:---:|:----:|
| 标准文档（默认） | 0.30 | 0.60 | 1.5 | 0.50 | 30 | 0.00 |
| 密集小字 | 0.15 | 0.40 | 2.5 | 0.40 | 20 | 0.80 |
| 印章/水印 | 0.25 | **0.60** | 1.5 | 0.60 | 40 | 0.90 |
| 倾斜文字 | 0.20 | 0.45 | **3.0** | 0.30 | 50 | 0.80 |
| 手写文档 | 0.15 | 0.50 | 2.5 | 0.40 | **80** | 0.75 |

---

## 模型选择指南

| 模型 | 骨干网络 | 字符集 | 适用场景 |
|------|----------|--------|----------|
| **tiny** | MobileNetV3 | 6905 | 性能敏感、标准印刷文档 |
| **small** | MobileNetV3 | 18709 | 通用场景、含部分生僻字/手写 |
| **medium** | ResNet50_vd | 18709 | 精度优先 |

---

## 技术要点

### 检测后处理（DBPostProcess）

原始 DBPostProcess 包含 `approxPolyDP → unclip → get_mini_boxes` 流程，本项目在 C++ 中通过形态学膨胀近似 unclip：

```text
原始轮廓 → approxPolyDP（ε=0.002×周长）→ fillPoly → 膨胀（椭圆核）→ findContours → minAreaRect
```

### CTC 置信度计算

识别模型输出每个时间步的 softmax 概率。对非 blank、非连续重复的时间步取最大概率的均值作为整行置信度。默认 `rec_conf_threshold = 0.0`（与官方一致，不丢弃结果），调参时可按需调高以剔除低置信度碎片。

### 裁剪垫边（Padding）

通过调整透视变换目标坐标并配合 `BORDER_REPLICATE` 实现垫边，避免文本框边缘字符被截断。
