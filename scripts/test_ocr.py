"""
Test PP-OCRv6 OCR using official paddleocr library (onnxruntime engine) on a single image.

Usage:
    python3 scripts/test_ocr.py <image_path> [--model medium|small|tiny] [--threads 4]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent


def run_single(inner, img_path: Path):
    t0 = time.perf_counter()
    image_arrays = inner.img_reader([str(img_path)])
    text_det_params = inner.get_text_det_params()

    t_det = time.perf_counter()
    det_results = list(inner.text_det_model(image_arrays, **text_det_params))
    det_ms = (time.perf_counter() - t_det) * 1000

    dt_polys_list = [item["dt_polys"] for item in det_results]
    dt_polys_list = [inner._sort_boxes(item) for item in dt_polys_list]

    texts = []
    scores = []
    rec_ms = 0.0

    if dt_polys_list[0]:
        all_subs = list(inner._crop_by_polys(image_arrays[0], dt_polys_list[0]))
        filtered_subs = [
            sub_img for sub_img in all_subs
            if sub_img.size > 0 and sub_img.shape[0] > 0 and sub_img.shape[1] > 0
        ]
        if filtered_subs:
            t_rec = time.perf_counter()
            for rec_res in inner.text_rec_model(filtered_subs):
                texts.append(rec_res["rec_text"])
                scores.append(rec_res["rec_score"])
            rec_ms = (time.perf_counter() - t_rec) * 1000

    total_ms = (time.perf_counter() - t0) * 1000
    return det_ms, rec_ms, total_ms, len(dt_polys_list[0]), texts, scores


def main():
    parser = argparse.ArgumentParser(description="PP-OCRv6 OCR Test (official paddleocr)")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--model", default="medium", choices=["medium", "small", "tiny"])
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for ONNX Runtime")
    args = parser.parse_args()

    target = Path(args.input)
    if not target.is_file():
        print(f"Error: image file not found: {target}")
        sys.exit(1)

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import cv2
    cv2.setNumThreads(args.threads)
    from paddleocr import PaddleOCR

    det_name = f"PP-OCRv6_{args.model}_det"
    rec_name = f"PP-OCRv6_{args.model}_rec"

    print(f"Loading PP-OCRv6 {args.model} (onnxruntime, threads={args.threads})...", flush=True)
    t_load = time.perf_counter()
    ocr = PaddleOCR(
        text_detection_model_name=det_name,
        text_recognition_model_name=rec_name,
        engine="onnxruntime",
        engine_config={
            "intra_op_num_threads": args.threads,
            "inter_op_num_threads": 1,
        },
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    load_ms = (time.perf_counter() - t_load) * 1000
    inner = ocr.paddlex_pipeline._pipeline

    det_ms, rec_ms, total_ms, n_boxes, texts, scores = run_single(inner, target)
    print(f"\n{'='*60}")
    print(f"Model:       PP-OCRv6 {args.model} (onnxruntime, threads={args.threads})")
    print(f"Image:       {target}")
    print(f"Load:        {load_ms:.1f}ms")
    print(f"Detection:   {n_boxes} boxes, {det_ms:.1f}ms")
    print(f"Recognition: {rec_ms:.1f}ms")
    print(f"Total:       {total_ms:.1f}ms")
    print(f"{'='*60}\n")
    for i, text in enumerate(texts):
        print(f"[{i:2d}] {text}")
    out_path = OUTPUT_DIR / f"result_{target.stem}_{args.model}.json"
    output = {
        "model": f"PP-OCRv6_{args.model}",
        "engine": "onnxruntime",
        "image": target.name,
        "load_ms": round(load_ms, 1),
        "det_ms": round(det_ms, 1),
        "rec_ms": round(rec_ms, 1),
        "total_ms": round(total_ms, 1),
        "boxes": n_boxes,
        "lines": len(texts),
        "text": texts,
        "scores": scores,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
