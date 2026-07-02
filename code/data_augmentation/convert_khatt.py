"""
convert_khatt.py — Download the Khatt Arabic handwriting dataset from HuggingFace
and convert it to Sedrah-compatible Parquet without executing the dataset script.

Khatt is an *offline* dataset (scanned images, no pen-trajectory data).
This converter extracts stroke paths by skeletonizing each word/line image
and tracing the resulting skeleton into ordered [x, y] point sequences that
the existing JsonOnlyCollator can consume as training targets.

Usage:
    python code/data_augmentation/convert_khatt.py \
        --repo-id ARBML/khatt \
        --output-dir sedrah_pipeline/khatt_dataset \
        --workers 8

    # With explicit HF token (falls back to HF_TOKEN env var):
    python code/data_augmentation/convert_khatt.py \
        --repo-id ARBML/khatt \
        --output-dir sedrah_pipeline/khatt_dataset \
        --hf-token hf_...

Output layout:
    sedrah_pipeline/khatt_dataset/
        raw/                        ← downloaded HF repo (images + annotations)
        json/
            train/   *.json         ← one Sedrah JSON per sample (for train_sandbox.py)
            validation/ *.json
            test/    *.json
        train.parquet               ← fast-load alternative (text, target_json, image_path)
        validation.parquet
        test.parquet

Integration with JsonOnlyCollator — see bottom of this file.
"""

import argparse
import json
import os
import re
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore", category=UserWarning)

# ── Skeleton tracing ──────────────────────────────────────────────────────────

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """Return a clean binary skeleton (255 = stroke pixel) from a BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Denoise slightly before thresholding
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu binarization — works for both dark-on-light and light-on-dark
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Skeletonize using Zhang-Suen thinning (OpenCV ximgproc or iterative erosion)
    try:
        import cv2.ximgproc as xip
        skeleton = xip.thinning(binary, thinningType=xip.THINNING_ZHANGSUEN)
    except (AttributeError, ImportError):
        skeleton = _iterative_skeletonize(binary)

    return skeleton


def _iterative_skeletonize(binary: np.ndarray) -> np.ndarray:
    """Fallback skeletonizer using repeated morphological erosion + hit-or-miss."""
    from skimage.morphology import skeletonize as ski_skel
    bool_img = binary > 0
    thinned = ski_skel(bool_img)
    return (thinned.astype(np.uint8)) * 255


def _pixel_neighbors(y: int, x: int, skeleton: np.ndarray):
    """Return 8-connected skeleton neighbors of (y, x)."""
    h, w = skeleton.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                yield ny, nx


def _trace_component(ys: np.ndarray, xs: np.ndarray, skeleton: np.ndarray) -> list[list[list[float]]]:
    """
    Trace a single connected skeleton component into ordered stroke paths.

    Strategy:
    - Find branch points (≥3 neighbors) and endpoints (1 neighbor).
    - BFS from each unvisited endpoint; at branch points start a new stroke.
    - Arabic text is right-to-left so we sort start-points by x descending.
    """
    pixels = set(zip(ys.tolist(), xs.tolist()))
    degree = {p: sum(1 for _ in _pixel_neighbors(p[0], p[1], skeleton)) for p in pixels}

    endpoints = sorted([p for p, d in degree.items() if d == 1], key=lambda p: -p[1])
    if not endpoints:
        # Closed loop — pick the rightmost pixel as entry
        endpoints = [max(pixels, key=lambda p: -p[1])]

    visited_edges: set[tuple] = set()
    strokes: list[list[list[float]]] = []

    def trace_from(start) -> list[list[float]]:
        path = [start]
        prev = None
        cur = start
        while True:
            neighbors = [
                n for n in _pixel_neighbors(cur[0], cur[1], skeleton)
                if n != prev and n in pixels
            ]
            # Exclude already-traversed edges to avoid infinite loops
            neighbors = [n for n in neighbors if (cur, n) not in visited_edges]
            if not neighbors:
                break
            nxt = neighbors[0]
            visited_edges.add((cur, nxt))
            visited_edges.add((nxt, cur))
            path.append(nxt)
            if degree.get(nxt, 0) >= 3:
                break  # hit branch point — stop this stroke
            prev, cur = cur, nxt
        return [[float(p[1]), float(p[0])] for p in path]  # (x, y)

    visited_starts: set = set()
    for ep in endpoints:
        if ep in visited_starts:
            continue
        visited_starts.add(ep)
        stroke = trace_from(ep)
        if len(stroke) >= 2:
            strokes.append(stroke)

    return strokes


def image_to_strokes(img_bgr: np.ndarray, max_strokes: int = 64) -> list[list[list[float]]]:
    """
    Convert a handwriting image to a list of stroke paths [[x,y], ...].

    Each connected component of the skeleton becomes a separate stroke (pen-lift).
    Components are ordered right-to-left to approximate Arabic writing direction.
    """
    skeleton = preprocess_image(img_bgr)
    if skeleton.max() == 0:
        return []

    # Label connected components
    num_labels, labels = cv2.connectedComponents(skeleton)
    all_strokes: list[tuple[float, list]] = []  # (rightmost_x, stroke_paths)

    for label in range(1, num_labels):
        mask = (labels == label).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        if len(xs) < 2:
            continue
        paths = _trace_component(ys, xs, mask)
        rightmost_x = float(xs.max())
        for p in paths:
            if len(p) >= 2:
                all_strokes.append((rightmost_x, p))

    # Sort right-to-left (Arabic writing direction)
    all_strokes.sort(key=lambda t: -t[0])
    strokes = [s for _, s in all_strokes[:max_strokes]]
    return strokes


# ── Annotation parsers ────────────────────────────────────────────────────────

def _parse_tsv_manifest(path: Path) -> list[tuple[str, str]]:
    """Parse a tab-separated file with columns: image_path\\ttext."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def _parse_csv_manifest(path: Path) -> list[tuple[str, str]]:
    """Parse a CSV with at least two columns: image, text (header optional)."""
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            img, text = parts[0].strip().strip('"'), parts[1].strip().strip('"')
            if i == 0 and not Path(img).suffix:
                continue  # skip header row
            rows.append((img, text))
    return rows


def _parse_flat_text(ann_path: Path, img_dir: Path) -> list[tuple[str, str]]:
    """
    Khatt often ships one .txt per image with the transcription.
    Match <stem>.txt → <stem>.png / .jpg / .bmp.
    """
    rows = []
    for txt_file in sorted(ann_path.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            img_candidate = img_dir / (txt_file.stem + ext)
            if img_candidate.exists():
                rows.append((str(img_candidate), text))
                break
    return rows


def discover_annotations(raw_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Auto-detect annotation format in the downloaded Khatt repo.

    Returns {split_name: [(image_path, text), ...]} for train/validation/test.
    """
    raw_dir = Path(raw_dir)
    splits: dict[str, list] = {}

    # 1. Explicit split manifests (TSV or CSV)
    split_aliases = {
        "train": ["train", "training"],
        "validation": ["val", "valid", "validation", "dev"],
        "test": ["test", "testing"],
    }
    for split, aliases in split_aliases.items():
        for alias in aliases:
            for ext in (".tsv", ".csv", ".txt"):
                candidate = raw_dir / f"{alias}{ext}"
                if candidate.exists():
                    rows = _parse_tsv_manifest(candidate) if ext == ".tsv" else _parse_csv_manifest(candidate)
                    if rows:
                        splits[split] = rows
                        break
            if split in splits:
                break

    if splits:
        return splits

    # 2. Per-split subdirectories with flat text annotations
    for split, aliases in split_aliases.items():
        for alias in aliases:
            split_dir = raw_dir / alias
            if split_dir.is_dir():
                # Check for images + sidecar .txt files
                img_dir = split_dir / "images" if (split_dir / "images").is_dir() else split_dir
                ann_dir = split_dir / "annotations" if (split_dir / "annotations").is_dir() else split_dir
                rows = _parse_flat_text(ann_dir, img_dir)
                if rows:
                    splits[split] = rows

    if splits:
        return splits

    # 3. Flat structure: all images + sidecar .txt files at repo root
    print("  No split manifests found — collecting all samples into 'train' and splitting 90/5/5.")
    all_rows = _parse_flat_text(raw_dir, raw_dir)
    if not all_rows:
        # Walk recursively for nested structures
        for txt_file in sorted(raw_dir.rglob("*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                img = txt_file.with_suffix(ext)
                if img.exists():
                    all_rows.append((str(img), text))
                    break

    if not all_rows:
        raise FileNotFoundError(
            f"Could not find any (image, text) pairs under {raw_dir}. "
            "Check --raw-dir or adjust the parser for the actual Khatt layout."
        )

    n = len(all_rows)
    t = int(n * 0.90)
    v = int(n * 0.95)
    splits["train"] = all_rows[:t]
    splits["validation"] = all_rows[t:v]
    splits["test"] = all_rows[v:]
    return splits


# ── Core conversion ───────────────────────────────────────────────────────────

def _safe_stem(text: str, max_len: int = 40) -> str:
    safe = re.sub(r"[^\w؀-ۿ]", "_", text)
    return safe[:max_len] or "sample"


def process_sample(
    image_path: str,
    text: str,
    sample_index: int,
    min_strokes: int = 1,
) -> dict | None:
    """
    Load one image, skeletonize it, and return a Sedrah-schema record or None on failure.

    Returns:
        {
            "text": str,
            "target_json": str,   # JSON string — this is what JsonOnlyCollator trains on
            "image_path": str,    # absolute path to source image
        }
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    strokes = image_to_strokes(img)
    if len(strokes) < min_strokes:
        return None

    record = {
        "text": text,
        "sample_index": sample_index,
        "strokes": [{"label": "", "points": pts} for pts in strokes],
    }
    return {
        "text": text,
        "target_json": json.dumps(record, ensure_ascii=False),
        "image_path": str(Path(image_path).resolve()),
    }


def convert_split(
    rows: list[tuple[str, str]],
    output_dir: Path,
    split: str,
    json_dir: Path | None,
    workers: int = 4,
) -> Path:
    """
    Convert a list of (image_path, text) pairs to a Parquet file.
    Optionally also writes individual JSON files (for train_sandbox.py compatibility).
    """
    import concurrent.futures

    records: list[dict] = []
    json_split_dir = (json_dir / split) if json_dir else None
    if json_split_dir:
        json_split_dir.mkdir(parents=True, exist_ok=True)

    def _process(args):
        idx, (img_path, text) = args
        result = process_sample(img_path, text, idx)
        return idx, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, (i, row)): i for i, row in enumerate(rows)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            idx, result = fut.result()
            if result is None:
                continue
            records.append(result)

            # Write individual JSON file for train_sandbox.py compatibility
            if json_split_dir:
                stem = f"{_safe_stem(result['text'])}_{idx}"
                json_path = json_split_dir / f"{stem}.json"
                parsed = json.loads(result["target_json"])
                json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

            if done % 500 == 0 or done == len(rows):
                print(f"  [{split}] {done}/{len(rows)} processed, {len(records)} valid")

    df = pd.DataFrame(records, columns=["text", "target_json", "image_path"])
    parquet_path = output_dir / f"{split}.parquet"
    df.to_parquet(parquet_path, index=False, compression="snappy")
    print(f"  Saved {len(df)} rows → {parquet_path}")
    return parquet_path


# ── Download ──────────────────────────────────────────────────────────────────

def download_khatt(repo_id: str, local_dir: Path, token: str | None) -> Path:
    """
    Download raw Khatt files from HuggingFace Hub without executing dataset_script.py.
    The ignore_patterns list skips all Python files and the loading script specifically.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} → {local_dir}  (skipping .py files) ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        ignore_patterns=["*.py", "*.pyc", "__pycache__/**"],
        token=token,
    )
    print("  Download complete.")
    return local_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Khatt Arabic handwriting dataset to Sedrah Parquet format."
    )
    p.add_argument("--repo-id", default="ARBML/khatt", help="HuggingFace dataset repo ID.")
    p.add_argument("--output-dir", required=True, help="Root output directory.")
    p.add_argument("--raw-dir", default="", help="Skip download and use an already-downloaded repo here.")
    p.add_argument("--hf-token", default="", help="HuggingFace token. Falls back to HF_TOKEN env var.")
    p.add_argument("--workers", type=int, default=8, help="Parallel image-processing workers.")
    p.add_argument("--min-strokes", type=int, default=1, help="Skip samples with fewer skeleton strokes.")
    p.add_argument("--no-json", action="store_true", help="Skip writing individual JSON files.")
    return p


def main():
    args = build_parser().parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HF_TOCKEN", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
        print(f"Using existing raw dir: {raw_dir}")
    else:
        raw_dir = download_khatt(args.repo_id, output_dir / "raw", token or None)

    # Discover annotations
    print("Discovering annotation files ...")
    splits = discover_annotations(raw_dir)
    print(f"  Found splits: { {k: len(v) for k, v in splits.items()} }")

    # Convert each split
    json_dir = None if args.no_json else output_dir / "json"
    for split, rows in splits.items():
        print(f"\nConverting [{split}] ({len(rows)} samples) ...")
        convert_split(rows, output_dir, split, json_dir, workers=args.workers)

    print(f"\nDone. Output in {output_dir}")
    print(_integration_snippet(output_dir))


def _integration_snippet(output_dir: Path) -> str:
    return f"""
─────────────────────────────────────────────────────────
Integration with train_sandbox.py / JsonOnlyCollator
─────────────────────────────────────────────────────────
Option A — Use the JSON files directly (zero code changes):

    accelerate launch code/training/train_sandbox.py \\
        --data-dir {output_dir}/json \\
        --image-manifest <build one with render_augmented_images.py> \\
        ...

Option B — Load Parquet and feed to JsonOnlyCollator:

    import pandas as pd
    from torch.utils.data import DataLoader
    from code.training.train_sandbox import JsonOnlyCollator
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    collator  = JsonOnlyCollator(processor, max_length=8192)

    class KhattParquetDataset:
        def __init__(self, parquet_path):
            self.df = pd.read_parquet(parquet_path)
        def __len__(self):
            return len(self.df)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            return {{
                "text":        row["text"],
                "target_text": row["target_json"],
                "image_path":  row["image_path"],
            }}

    train_ds = KhattParquetDataset("{output_dir}/train.parquet")
    loader   = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collator)
─────────────────────────────────────────────────────────
"""


if __name__ == "__main__":
    main()
