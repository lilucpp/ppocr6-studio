#!/usr/bin/env python3
"""
Download PP-OCRv6 ONNX models and generate chars.txt from inference.yml.

Usage:
    python3 scripts/download_models.py --model tiny
    python3 scripts/download_models.py --model all --mirror
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL_SIZES = ("tiny", "small", "medium")
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
MIRROR_HF_ENDPOINT = "https://hf-mirror.com"


def parse_chars_from_yml(yml_text: str) -> list[str]:
    """Extract PostProcess.character_dict entries from PaddleX inference.yml."""
    lines = yml_text.splitlines()
    in_dict = False
    chars = ["blank"]
    for line in lines:
        if line.strip() == "character_dict:":
            in_dict = True
            continue
        if in_dict:
            if not line.startswith("  - "):
                break
            val = line[4:]
            if val.startswith("'") and val.endswith("'") and len(val) >= 2:
                val = val[1:-1].replace("''", "'")
            elif val.startswith('"') and val.endswith('"') and len(val) >= 2:
                val = val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            chars.append(val)
    return chars


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [skip] {dest} (already exists)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get]  {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "pp-ocr-cpp-downloader/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256
            with open(tmp_dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        print(
                            f"\r         {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)",
                            end="",
                            flush=True,
                        )
            if total > 0:
                print()
        tmp_dest.replace(dest)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def download_model_size(endpoint: str, output_dir: Path, size: str, force: bool) -> None:
    print(f"\n=== PP-OCRv6 {size} ===")
    det_repo = f"PaddlePaddle/PP-OCRv6_{size}_det_onnx"
    rec_repo = f"PaddlePaddle/PP-OCRv6_{size}_rec_onnx"

    det_dir = output_dir / f"PP-OCRv6_{size}_det_onnx"
    rec_dir = output_dir / f"PP-OCRv6_{size}_rec_onnx"

    download_file(
        f"{endpoint}/{det_repo}/resolve/main/inference.onnx",
        det_dir / "inference.onnx",
        force=force,
    )
    download_file(
        f"{endpoint}/{rec_repo}/resolve/main/inference.onnx",
        rec_dir / "inference.onnx",
        force=force,
    )

    chars_path = rec_dir / "chars.txt"
    if chars_path.exists() and chars_path.stat().st_size > 0 and not force:
        print(f"  [skip] {chars_path} (already exists)")
        return

    yml_path = rec_dir / "inference.yml"
    download_file(
        f"{endpoint}/{rec_repo}/resolve/main/inference.yml",
        yml_path,
        force=force,
    )
    yml_text = yml_path.read_text(encoding="utf-8")
    chars = parse_chars_from_yml(yml_text)
    if len(chars) <= 1:
        raise RuntimeError(f"Failed to parse character_dict from {yml_path}")
    chars_path.write_text("\n".join(chars) + "\n", encoding="utf-8")
    yml_path.unlink(missing_ok=True)
    print(f"  [gen]  {chars_path} ({len(chars)} entries)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PP-OCRv6 ONNX models.")
    parser.add_argument(
        "--model",
        default="tiny",
        choices=[*MODEL_SIZES, "all"],
        help="Model tier to download (default: tiny)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Target directory for models (default: models)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help=f"Use {MIRROR_HF_ENDPOINT} instead of {DEFAULT_HF_ENDPOINT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if not endpoint:
        endpoint = MIRROR_HF_ENDPOINT if args.mirror else DEFAULT_HF_ENDPOINT

    output_dir = Path(args.output_dir)
    sizes = MODEL_SIZES if args.model == "all" else (args.model,)

    print(f"Endpoint: {endpoint}")
    print(f"Output:   {output_dir.resolve()}")

    try:
        for size in sizes:
            download_model_size(endpoint, output_dir, size, args.force)
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1

    print("\nAll requested models are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
