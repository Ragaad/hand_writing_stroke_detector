import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
]


def load_font(size=14):
    candidates = [
        "/System/Library/Fonts/GeezaPro.ttc",
        "/System/Library/Fonts/SFArabic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def normalize_strokes(raw):
    if isinstance(raw, list):
        strokes = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Stroke item {index} must be a one-key object.")
            label, points = next(iter(item.items()))
            strokes.append({"label": label, "points": points})
        return strokes

    if isinstance(raw, dict) and isinstance(raw.get("strokes"), list):
        strokes = []
        for index, item in enumerate(raw["strokes"]):
            if "label" in item and "points" in item:
                strokes.append({"label": item["label"], "points": item["points"]})
            elif isinstance(item, dict) and len(item) == 1:
                label, points = next(iter(item.items()))
                strokes.append({"label": label, "points": points})
            else:
                raise ValueError(f"Unsupported stroke item {index}: {item!r}")
        return strokes

    raise ValueError("Unsupported JSON format. Expected list of {label: points} or object with strokes.")


def point_bounds(strokes):
    xs = []
    ys = []
    for stroke in strokes:
        for point in stroke["points"]:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))

    if not xs or not ys:
        return 0.0, 0.0, 1.0, 1.0

    return min(xs), min(ys), max(xs), max(ys)


def transform_points(points, image_size, coord_size=None, fit_coords=False, padding=12):
    width, height = image_size

    valid_points = [
        (float(point[0]), float(point[1]))
        for point in points
        if isinstance(point, (list, tuple)) and len(point) == 2
    ]

    if not valid_points:
        return []

    if fit_coords:
        raise RuntimeError("fit_coords should be handled at stroke level.")

    if coord_size is None:
        coord_width, coord_height = width, height
    else:
        coord_width, coord_height = coord_size

    scale_x = width / max(coord_width, 1)
    scale_y = height / max(coord_height, 1)
    return [(x * scale_x, y * scale_y) for x, y in valid_points]


def transform_strokes(strokes, image_size, coord_size=None, fit_coords=False, padding=12):
    width, height = image_size

    if fit_coords:
        min_x, min_y, max_x, max_y = point_bounds(strokes)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)

        transformed = []
        for stroke in strokes:
            points = []
            for point in stroke["points"]:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    x = padding + (float(point[0]) - min_x) * scale
                    y = padding + (float(point[1]) - min_y) * scale
                    points.append((x, y))
            transformed.append({"label": stroke["label"], "points": points})
        return transformed

    return [
        {
            "label": stroke["label"],
            "points": transform_points(stroke["points"], image_size, coord_size=coord_size),
        }
        for stroke in strokes
    ]


def region_box(image_size, region):
    width, height = image_size
    if region == "full":
        return 0, 0, width, height
    if region == "left-half":
        return 0, 0, width // 2, height
    if region == "right-half":
        return width // 2, 0, width, height
    raise ValueError(f"Unsupported region: {region}")


def draw_label(draw, xy, label, font, color):
    x, y = xy
    text = str(label) if label != "" else "?"
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    background = (
        bbox[0] - pad,
        bbox[1] - pad,
        bbox[2] + pad,
        bbox[3] + pad,
    )
    draw.rectangle(background, fill=(255, 255, 255, 210))
    draw.text((x, y), text, font=font, fill=color)


def draw_strokes(draw, strokes, args, font):
    for index, stroke in enumerate(strokes):
        color = PALETTE[index % len(PALETTE)]
        rgba = (*color, args.alpha)
        points = stroke["points"]
        if len(points) >= 2:
            draw.line(points, fill=rgba, width=args.line_width, joint="curve")
        for x, y in points[:: max(1, len(points) // args.max_points_drawn)]:
            radius = args.point_radius
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgba)
        if args.labels and points:
            draw_label(draw, points[0], stroke["label"], font, color)


def render_overlay(args):
    json_path = Path(args.json)
    output_path = Path(args.output)

    with json_path.open("r", encoding="utf-8") as f:
        strokes = normalize_strokes(json.load(f))

    font = load_font(args.label_size)

    if args.json_only:
        base = Image.new("RGBA", (args.canvas_width, args.canvas_height), "white")
        region = base
        left, top, right, bottom = region_box(base.size, "full")
    else:
        if not args.image:
            raise ValueError("--image is required unless --json-only is used.")
        image_path = Path(args.image)
        base = Image.open(image_path).convert("RGBA")
        left, top, right, bottom = region_box(base.size, args.region)
        region = base.crop((left, top, right, bottom))

    coord_size = None
    if args.coord_width and args.coord_height:
        coord_size = (args.coord_width, args.coord_height)

    transformed = transform_strokes(
        strokes,
        region.size,
        coord_size=coord_size,
        fit_coords=args.fit_coords,
        padding=args.padding,
    )

    if args.json_only:
        result = Image.new("RGBA", region.size, "white")
        draw = ImageDraw.Draw(result)
        draw_strokes(draw, transformed, args, font)
        result = result.convert("RGB")
    elif args.split:
        width, height = region.size
        divider_height = 8
        result = Image.new("RGBA", (width, height * 2 + divider_height), "white")
        result.paste(region, (0, 0))

        divider = ImageDraw.Draw(result)
        divider.rectangle((0, height, width, height + divider_height), fill=(24, 24, 24, 255))

        bottom = Image.new("RGBA", region.size, "white")
        bottom_draw = ImageDraw.Draw(bottom)
        draw_strokes(bottom_draw, transformed, args, font)
        result.paste(bottom, (0, height + divider_height))
        result = result.convert("RGB")
    else:
        region_overlay = Image.new("RGBA", region.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(region_overlay)
        draw_strokes(draw, transformed, args, font)

        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        overlay.paste(region_overlay, (left, top))
        draw = ImageDraw.Draw(overlay)
        if args.region != "full" and args.show_region_boundary:
            draw.rectangle((left, top, right - 1, bottom - 1), outline=(24, 24, 24, 255), width=2)
        result = Image.alpha_composite(base, overlay).convert("RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved overlay: {output_path}")
    print(f"Image size: {base.size[0]}x{base.size[1]}")
    print(f"Render region: {'json-only' if args.json_only else args.region} ({region.size[0]}x{region.size[1]})")
    print(f"Rendered strokes: {len(strokes)}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Render JSON stroke trajectories on top of an image sample.")
    parser.add_argument("--image", help="Original sample image path. Not needed with --json-only.")
    parser.add_argument("--json", required=True, help="Stroke JSON path.")
    parser.add_argument("--output", required=True, help="Output overlay PNG/JPG path.")
    parser.add_argument("--json-only", action="store_true", help="Render only JSON strokes on a blank canvas.")
    parser.add_argument("--canvas-width", type=int, default=300, help="Canvas width for --json-only.")
    parser.add_argument("--canvas-height", type=int, default=300, help="Canvas height for --json-only.")
    parser.add_argument("--coord-width", type=float, help="Original JSON coordinate-space width.")
    parser.add_argument("--coord-height", type=float, help="Original JSON coordinate-space height.")
    parser.add_argument("--fit-coords", action="store_true", help="Fit JSON point bounds into the image canvas.")
    parser.add_argument(
        "--region",
        choices=["full", "left-half", "right-half"],
        default="full",
        help="Image region where JSON coordinates should be rendered.",
    )
    parser.add_argument(
        "--show-region-boundary",
        action="store_true",
        help="Draw a boundary around the selected render region in overlay mode.",
    )
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--alpha", type=int, default=210)
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--label-size", type=int, default=14)
    parser.add_argument("--max-points-drawn", type=int, default=64)
    parser.add_argument(
        "--split",
        action="store_true",
        help="Create a two-panel image: original/detected letters on top, JSON strokes on bottom.",
    )
    return parser


if __name__ == "__main__":
    render_overlay(build_arg_parser().parse_args())
