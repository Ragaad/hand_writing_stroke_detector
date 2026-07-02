"""
convert_arabic_ocr.py — Convert mssqpi/Arabic-OCR-Dataset to Sedrah JSON format.

Images are tiny (~74×35 px, printed Arabic text). We upscale 4x before
skeletonizing so the thinning algorithm has enough pixels to work with.
Outputs are text-only training samples (no image manifest entry) because
the source images are printed text, not handwriting.

Usage:
    python code/data_augmentation/convert_arabic_ocr.py \
        --output-dir sedrah_pipeline/calliar_combined_dataset \
        --train-samples 5000 \
        --val-samples 600 \
        --test-samples 600 \
        --workers 8
"""

import argparse
import io
import json
import os
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

UPSCALE = 4
MIN_STROKE_PX = 3
MAX_STROKES = 64
INDEX_OFFSET = 90000


# ── Skeleton tracing (same pipeline as convert_khatt.py) ─────────────────────

def preprocess_image(pil_img) -> np.ndarray:
    img = np.array(pil_img.convert("L"))
    # Upscale so thinning has enough pixels
    h, w = img.shape
    img = cv2.resize(img, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
    # Invert if dark-on-white (most OCR images are black text on white)
    if img.mean() > 127:
        img = 255 - img
    # Otsu binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Skeletonize
    try:
        import cv2.ximgproc as xip
        skeleton = xip.thinning(binary, thinningType=xip.THINNING_ZHANGSUEN)
    except (AttributeError, ImportError):
        from skimage.morphology import skeletonize as ski_skel
        skeleton = (ski_skel(binary > 0).astype(np.uint8)) * 255
    return skeleton


def _pixel_neighbors(y: int, x: int, skeleton: np.ndarray):
    h, w = skeleton.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                yield ny, nx


def _trace_component(ys, xs, skeleton) -> list:
    pixels = set(zip(ys.tolist(), xs.tolist()))
    degree = {p: sum(1 for _ in _pixel_neighbors(p[0], p[1], skeleton)) for p in pixels}
    endpoints = sorted([p for p, d in degree.items() if d == 1], key=lambda p: -p[1])
    if not endpoints:
        endpoints = [max(pixels, key=lambda p: -p[1])]
    visited_edges: set = set()
    strokes = []

    def trace_from(start):
        path = [start]
        prev, cur = None, start
        while True:
            nbrs = [n for n in _pixel_neighbors(cur[0], cur[1], skeleton)
                    if n != prev and n in pixels and (cur, n) not in visited_edges]
            if not nbrs:
                break
            nxt = nbrs[0]
            visited_edges.add((cur, nxt))
            visited_edges.add((nxt, cur))
            path.append(nxt)
            if degree.get(nxt, 0) >= 3:
                break
            prev, cur = cur, nxt
        return [[float(p[1]) / UPSCALE, float(p[0]) / UPSCALE] for p in path]

    visited_starts: set = set()
    for ep in endpoints:
        if ep in visited_starts:
            continue
        visited_starts.add(ep)
        s = trace_from(ep)
        if len(s) >= MIN_STROKE_PX:
            strokes.append(s)
    return strokes


def image_to_strokes(pil_img) -> list:
    skeleton = preprocess_image(pil_img)
    if skeleton.max() == 0:
        return []
    num_labels, labels = cv2.connectedComponents(skeleton)
    all_strokes = []
    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        if len(xs) < MIN_STROKE_PX:
            continue
        paths = _trace_component(ys, xs, mask)
        rightmost_x = float(xs.max()) / UPSCALE
        for p in paths:
            all_strokes.append((rightmost_x, p))
    all_strokes.sort(key=lambda t: -t[0])
    return [s for _, s in all_strokes[:MAX_STROKES]]


def strokes_to_json(strokes: list, text: str) -> list:
    """Convert raw stroke paths to Sedrah JSON format [{label: [[x,y],...]}]."""
    # Use text as label for all strokes (no per-char breakdown for printed text)
    return [{text: pts} for pts in strokes]


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_split(rows, out_dir: Path, index_offset: int, workers: int) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    failed = 0

    def process(i_row):
        i, row = i_row
        text = (row.get("text") or "").strip()
        if not text:
            return None
        try:
            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                pil_img = Image.open(io.BytesIO(img_data["bytes"]))
            elif isinstance(img_data, bytes):
                pil_img = Image.open(io.BytesIO(img_data))
            else:
                pil_img = img_data  # already PIL
            strokes = image_to_strokes(pil_img)
            if not strokes:
                return None
            json_data = strokes_to_json(strokes, text)
            # Sanitize text for filename (replace path-unsafe chars)
            safe_text = text.replace("/", "").replace("\\", "").replace("\x00", "")
            sample_index = index_offset + i
            fname = f"{safe_text}_{sample_index}.json"
            out_path = out_dir / fname
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, separators=(",", ":"))
            return fname
        except Exception as e:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process, (i, row)): i for i, row in enumerate(rows)}
        for fut in as_completed(futs):
            result = fut.result()
            if result:
                saved += 1
            else:
                failed += 1
            if (saved + failed) % 500 == 0:
                print(f"  {saved} saved, {failed} failed / {saved + failed} processed", flush=True)

    print(f"  Done: {saved} saved, {failed} skipped")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Convert mssqpi/Arabic-OCR-Dataset to Sedrah JSON")
    parser.add_argument("--output-dir", default="sedrah_pipeline/calliar_combined_dataset",
                        help="Root dir of the combined dataset (json/train, json/validation, json/test subdirs)")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=600)
    parser.add_argument("--test-samples", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (or set HF_TOKEN env)")
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HF_TOCKEN")

    from datasets import load_dataset

    out_root = Path(args.output_dir)
    total_needed = args.train_samples + args.val_samples + args.test_samples

    print(f"Loading mssqpi/Arabic-OCR-Dataset (streaming, need {total_needed} rows)...")
    ds = load_dataset(
        "mssqpi/Arabic-OCR-Dataset",
        split="train",
        streaming=True,
        token=token,
    )

    rows = []
    print(f"Collecting {total_needed} samples...")
    for row in ds:
        rows.append(row)
        if len(rows) >= total_needed:
            break
    print(f"Collected {len(rows)} samples")

    # Split
    train_rows = rows[:args.train_samples]
    val_rows = rows[args.train_samples: args.train_samples + args.val_samples]
    test_rows = rows[args.train_samples + args.val_samples:]

    splits = [
        ("train", train_rows, INDEX_OFFSET),
        ("validation", val_rows, INDEX_OFFSET + args.train_samples),
        ("test", test_rows, INDEX_OFFSET + args.train_samples + args.val_samples),
    ]

    for split_name, split_rows, offset in splits:
        if not split_rows:
            continue
        out_dir = out_root / "json" / split_name
        print(f"\n[{split_name}] Converting {len(split_rows)} samples → {out_dir}")
        saved = convert_split(split_rows, out_dir, offset, args.workers)
        print(f"[{split_name}] {saved}/{len(split_rows)} samples written")

    print("\nDone. No image manifest entries added (text-only training samples).")
    print("Run training with --data-dir pointing to the updated combined dataset.")


if __name__ == "__main__":
    main()
