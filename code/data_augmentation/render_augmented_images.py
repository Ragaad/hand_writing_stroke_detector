import argparse
import json
import multiprocessing as mp
from pathlib import Path

from PIL import Image, ImageDraw

import render_json_overlay as rjo


def render_one(args):
    json_path, out_path, padding, line_width, point_radius = args
    try:
        strokes = rjo.normalize_strokes(json.loads(Path(json_path).read_text(encoding="utf-8")))
    except Exception as exc:
        return (json_path, None, f"parse_error: {exc}")

    min_x, min_y, max_x, max_y = rjo.point_bounds(strokes)
    width = max(int(max_x - min_x + 2 * padding), 32)
    height = max(int(max_y - min_y + 2 * padding), 32)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    render_args = argparse.Namespace(alpha=255, line_width=line_width, point_radius=point_radius, labels=False, max_points_drawn=64)

    shifted = [
        {
            "label": s["label"],
            "points": [(x - min_x + padding, y - min_y + padding) for x, y in s["points"]],
        }
        for s in strokes
    ]
    # Force plain black lines (not the multicolor PALETTE from draw_strokes) to match the original imgs/ style.
    for stroke in shifted:
        points = stroke["points"]
        if len(points) >= 2:
            draw.line(points, fill=(0, 0, 0), width=line_width, joint="curve")
        for x, y in points[:: max(1, len(points) // render_args.max_points_drawn)]:
            r = point_radius
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return (json_path, str(out_path), None)


def main():
    parser = argparse.ArgumentParser(description="Render stroke JSON files to PNG images (1:1 coordinate scale, content-fit canvas).")
    parser.add_argument("--input-dir", required=True, help="Root json dir containing train/validation/test subfolders.")
    parser.add_argument("--output-dir", required=True, help="Root dir to write rendered images into (mirrors split structure).")
    parser.add_argument("--manifest-out", required=True, help="Path to write the resulting image_manifest.jsonl.")
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    tasks = []
    for split in args.splits:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
        for json_path in sorted(split_dir.glob("*.json")):
            out_path = output_dir / split / f"{json_path.stem}.png"
            tasks.append((str(json_path), str(out_path), args.padding, args.line_width, args.point_radius))

    print(f"Rendering {len(tasks)} images with {args.workers} workers...")
    manifest_rows = []
    errors = 0
    with mp.Pool(args.workers) as pool:
        for json_path, out_path, error in pool.imap_unordered(render_one, tasks, chunksize=64):
            if error:
                errors += 1
                print(f"WARNING: {json_path}: {error}")
                continue
            manifest_rows.append({"json": json_path, "image": out_path, "source": "rendered"})

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Rendered {len(manifest_rows)} images, {errors} errors -> {manifest_path}")


if __name__ == "__main__":
    main()
