import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT_CANDIDATES = [
    "/System/Library/Fonts/GeezaPro.ttc",
    "/System/Library/Fonts/SFArabic.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]

ARABIC_COMPATIBILITY_MAP = str.maketrans(
    {
        "ی": "ي",
        "ے": "ي",
        "ک": "ك",
        "ھ": "ه",
        "ە": "ه",
        "ۀ": "ة",
    }
)


def require_pypdf():
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pypdf. Install it with `pip install -r requirements.txt`."
        ) from exc
    return PdfReader


def resolve_font(font_path=None):
    if font_path:
        path = Path(font_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Font file not found: {path}")
        return path

    for candidate in DEFAULT_FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(
        "No Arabic-capable font found. Pass one explicitly with `--font /path/to/font.ttf`."
    )


def load_font(font_path, font_size):
    try:
        return ImageFont.truetype(str(font_path), font_size, layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        return ImageFont.truetype(str(font_path), font_size)


def extract_pdf_pages(pdf_path):
    PdfReader = require_pypdf()
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_index, text))
    return pages


def fix_rtl_extraction_order(text):
    fixed_lines = []
    for line in text.splitlines():
        if any(is_arabic_codepoint(char) for char in line):
            fixed_lines.append(line[::-1])
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)


def is_arabic_codepoint(char):
    codepoint = ord(char)
    return (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    )


def is_arabic_letter_or_mark(char):
    if not is_arabic_codepoint(char):
        return False
    category = unicodedata.category(char)
    return category.startswith("L") or category.startswith("M")


def clean_arabic_letters(text):
    text = unicodedata.normalize("NFKC", text).translate(ARABIC_COMPATIBILITY_MAP)
    letters = []
    for char in text:
        if is_arabic_letter_or_mark(char):
            letters.append(char)
        elif char.isspace() and letters and letters[-1] != " ":
            letters.append(" ")

    cleaned = "".join(letters)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def chunk_letters(text, max_letters):
    current = []
    for char in text:
        if char == " ":
            continue
        current.append(char)
        if len(current) >= max_letters:
            yield "".join(current)
            current = []
    if current:
        yield "".join(current)


def slugify_arabic(text, max_chars=36):
    slug = re.sub(r"\s+", "_", text.strip())
    slug = re.sub(r"[^\w\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF_-]+", "", slug)
    return slug[:max_chars] or "arabic"


def text_bbox(draw, position, text, font):
    bbox = draw.textbbox(position, text, font=font)
    return bbox[0], bbox[1], bbox[2], bbox[3]


def draw_centered_letter(draw, letter, bbox, font, fill):
    x1, y1, x2, y2 = bbox
    letter_bbox = text_bbox(draw, (0, 0), letter, font)
    letter_width = letter_bbox[2] - letter_bbox[0]
    letter_height = letter_bbox[3] - letter_bbox[1]
    cx = x1 + (x2 - x1 - letter_width) / 2 - letter_bbox[0]
    cy = y1 + (y2 - y1 - letter_height) / 2 - letter_bbox[1]
    draw.text((cx, cy), letter, font=font, fill=fill)


def boundary_points(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = center & ~neighbors
    bys, bxs = np.nonzero(boundary)
    if len(bxs) == 0:
        return np.column_stack([xs, ys]).astype(np.float32)

    return np.column_stack([bxs, bys]).astype(np.float32)


def connected_components(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_x), int(start_y))]
        visited[start_y, start_x] = True
        pixels = []

        while stack:
            x, y = stack.pop()
            pixels.append((x, y))

            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if visited[next_y, next_x] or not mask[next_y, next_x]:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_x, next_y))

        components.append(pixels)

    return components


def component_mask_from_pixels(pixels, shape):
    mask = np.zeros(shape, dtype=bool)
    for x, y in pixels:
        mask[y, x] = True
    return mask


def smooth_path(points, window):
    if window <= 1 or len(points) < 3:
        return points

    if window % 2 == 0:
        window += 1

    array = np.array(points, dtype=np.float32)
    radius = window // 2
    smoothed = []
    for index in range(len(array)):
        indexes = [(index + offset) % len(array) for offset in range(-radius, radius + 1)]
        smoothed.append(array[indexes].mean(axis=0))

    return smoothed


def order_and_sample_points(points, max_points, smooth_window=1):
    if len(points) == 0:
        return []

    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    distances = np.linalg.norm(points - centroid, axis=1)
    order = np.lexsort((distances, angles))
    ordered = points[order]

    if len(ordered) > max_points:
        indexes = np.linspace(0, len(ordered) - 1, max_points).astype(int)
        ordered = ordered[indexes]

    ordered = smooth_path(ordered, smooth_window)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in ordered]


def letter_to_point_groups(
    letter,
    font,
    cell_bbox,
    canvas_size,
    points_per_letter,
    mask_threshold,
    min_component_pixels,
    smooth_window,
    merge_components,
):
    mask_image = Image.new("L", (canvas_size, canvas_size), 0)
    mask_draw = ImageDraw.Draw(mask_image)
    draw_centered_letter(mask_draw, letter, cell_bbox, font, fill=255)

    mask = np.array(mask_image) >= mask_threshold
    components = [
        pixels
        for pixels in connected_components(mask)
        if len(pixels) >= min_component_pixels
    ]

    if not components:
        x1, y1, x2, y2 = cell_bbox
        return [[[round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)]]]

    if merge_components:
        component_masks = [component_mask_from_pixels(pixels, mask.shape) for pixels in components]
        merged_mask = np.logical_or.reduce(component_masks)
        points = boundary_points(merged_mask)
        sampled = order_and_sample_points(points, points_per_letter, smooth_window=smooth_window)
        return [sampled] if sampled else []

    point_groups = []
    components = sorted(components, key=len, reverse=True)
    for pixels in components:
        component_mask = component_mask_from_pixels(pixels, mask.shape)
        points = boundary_points(component_mask)
        sampled = order_and_sample_points(points, points_per_letter, smooth_window=smooth_window)
        if sampled:
            point_groups.append(sampled)

    return point_groups


def render_sample(
    letters,
    font_path,
    canvas_size,
    points_per_letter,
    columns,
    mask_threshold,
    min_component_pixels,
    smooth_window,
    merge_components,
):
    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)

    columns = min(columns, max(1, len(letters)))
    rows = math.ceil(len(letters) / columns)
    margin = max(8, canvas_size // 24)
    gap = max(3, canvas_size // 100)
    cell_width = (canvas_size - 2 * margin - gap * (columns - 1)) / columns
    cell_height = (canvas_size - 2 * margin - gap * (rows - 1)) / rows
    font_size = max(10, int(min(cell_width, cell_height) * 0.78))
    font = load_font(font_path, font_size)

    strokes = []
    for index, letter in enumerate(letters):
        row = index // columns
        col_from_right = index % columns
        col = columns - 1 - col_from_right

        x1 = margin + col * (cell_width + gap)
        y1 = margin + row * (cell_height + gap)
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        cell_bbox = (x1, y1, x2, y2)

        draw_centered_letter(draw, letter, cell_bbox, font, fill="black")
        point_groups = letter_to_point_groups(
            letter,
            font,
            cell_bbox,
            canvas_size,
            points_per_letter,
            mask_threshold,
            min_component_pixels,
            smooth_window,
            merge_components,
        )
        for points in point_groups:
            strokes.append({letter: points})

    return image, strokes


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def synthesize_from_pdfs(args):
    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json" / args.split
    image_dir = output_dir / "images" / args.split
    manifest_path = output_dir / "manifest.jsonl"

    json_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    font_path = resolve_font(args.font)
    manifest_rows = []
    sample_index = 0

    for pdf_arg in args.pdf:
        pdf_path = Path(pdf_arg).expanduser()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        for page_number, page_text in extract_pdf_pages(pdf_path):
            if args.fix_rtl_order:
                page_text = fix_rtl_extraction_order(page_text)
            arabic_text = clean_arabic_letters(page_text)
            if not arabic_text:
                continue

            for letters in chunk_letters(arabic_text, args.max_letters_per_sample):
                image, strokes = render_sample(
                    letters,
                    font_path=font_path,
                    canvas_size=args.canvas_size,
                    points_per_letter=args.points_per_letter,
                    columns=args.columns,
                    mask_threshold=args.mask_threshold,
                    min_component_pixels=args.min_component_pixels,
                    smooth_window=args.smooth_window,
                    merge_components=args.merge_components,
                )

                sample_index += 1
                slug = slugify_arabic(letters)
                stem = f"sample_{sample_index:06d}_{slug}"
                json_path = json_dir / f"{stem}.json"
                image_path = image_dir / f"{stem}.png"

                write_json(json_path, strokes)
                image.save(image_path)

                manifest_rows.append(
                    {
                        "sample_id": stem,
                        "text": letters,
                        "json": str(json_path),
                        "image": str(image_path),
                        "source_pdf": str(pdf_path),
                        "source_page": page_number,
                        "canvas_size": [args.canvas_size, args.canvas_size],
                        "label_type": "synthetic_glyph_boundary_pseudo_strokes",
                        "font": str(font_path),
                        "mask_threshold": args.mask_threshold,
                        "min_component_pixels": args.min_component_pixels,
                        "smooth_window": args.smooth_window,
                        "merge_components": args.merge_components,
                    }
                )

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(manifest_rows)} synthetic samples")
    print(f"JSON output: {json_dir}")
    print(f"Image output: {image_dir}")
    print(f"Manifest: {manifest_path}")
    if not manifest_rows:
        print("No Arabic text was extracted. If your PDF is scanned, run OCR first or provide selectable-text PDFs.")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract Arabic text from PDFs and synthesize letter-level pseudo-stroke JSON data."
    )
    parser.add_argument("--pdf", nargs="+", required=True, help="One or more selectable-text Arabic PDF files.")
    parser.add_argument("--output-dir", default="synthesis_arabic_data")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--font", help="Arabic-capable .ttf/.otf/.ttc font path.")
    parser.add_argument("--canvas-size", type=int, default=300)
    parser.add_argument("--max-letters-per-sample", type=int, default=16)
    parser.add_argument("--points-per-letter", type=int, default=96)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=64,
        help="Minimum rendered glyph mask value to keep. Higher removes faint anti-aliased pixels.",
    )
    parser.add_argument(
        "--min-component-pixels",
        type=int,
        default=8,
        help="Drop connected components smaller than this. Dots above this size are preserved.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Moving-average window for sampled boundary paths. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--merge-components",
        action="store_true",
        help="Merge each letter's connected components into one stroke instead of keeping dots/marks separate.",
    )
    parser.add_argument(
        "--fix-rtl-order",
        action="store_true",
        help="Reverse each extracted Arabic line. Use this if generated filenames/text look backwards.",
    )
    return parser


if __name__ == "__main__":
    synthesize_from_pdfs(build_arg_parser().parse_args())
