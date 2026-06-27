import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean


def _stroke_points(stroke):
    """A stroke is either a bare list of points, or one of our Sedrah-style
    wrappers: {"points": [...]} or the single-key {label: [...]} form."""
    if isinstance(stroke, dict):
        if "points" in stroke:
            return stroke["points"]
        return next(iter(stroke.values()))
    return stroke


def extract_points(strokes_json):
    """Flatten a strokes JSON structure into a single [N, 2] list of [x, y] points,
    in stroke/point order. Accepts a bare list of strokes (each a list of [x,y] or
    [x,y,t] points), or our {"strokes": [...]} wrapper."""
    if isinstance(strokes_json, dict) and isinstance(strokes_json.get("strokes"), list):
        strokes_json = strokes_json["strokes"]

    points = []
    for stroke in strokes_json:
        for point in _stroke_points(stroke):
            points.append([float(point[0]), float(point[1])])
    return points


def normalize_points(points, canvas_size=None):
    """Normalize a flat [N, 2] point list to [0, 1]. Uses canvas_size=(width, height)
    if given, otherwise falls back to the trajectory's own bounding box."""
    if not points:
        return []

    arr = np.asarray(points, dtype=np.float64)

    if canvas_size is not None:
        width, height = canvas_size
        origin = np.zeros(2)
        span = np.array([max(width, 1e-9), max(height, 1e-9)])
    else:
        origin = arr.min(axis=0)
        span = arr.max(axis=0) - origin
        span[span == 0] = 1e-9

    return ((arr - origin) / span).tolist()


def flatten_trajectory(strokes_json, canvas_size=None):
    """Extract and normalize a strokes JSON structure into a flat [N, 2] point list."""
    return normalize_points(extract_points(strokes_json), canvas_size=canvas_size)


def dtw_distance(pred_points, gt_points):
    """Dynamic Time Warping distance between two point sequences. DTW handles the
    temporal alignment itself (sequence order is the time axis), so no explicit
    timestamp is needed once points are flattened to [x, y]."""
    if not pred_points or not gt_points:
        return float("inf")
    distance, _ = fastdtw(pred_points, gt_points, dist=euclidean)
    return distance


def precision_recall(pred_points, gt_points, threshold=0.05):
    """Precision: fraction of predicted points with a ground-truth neighbor within
    `threshold`. Recall: fraction of ground-truth points with a predicted neighbor
    within `threshold`. Both computed via nearest-neighbor lookup (KD-tree)."""
    if not pred_points or not gt_points:
        return 0.0, 0.0

    pred_arr = np.asarray(pred_points, dtype=np.float64)
    gt_arr = np.asarray(gt_points, dtype=np.float64)

    gt_tree = cKDTree(gt_arr)
    pred_to_gt_dist, _ = gt_tree.query(pred_arr)
    precision = float(np.mean(pred_to_gt_dist <= threshold))

    pred_tree = cKDTree(pred_arr)
    gt_to_pred_dist, _ = pred_tree.query(gt_arr)
    recall = float(np.mean(gt_to_pred_dist <= threshold))

    return precision, recall


def evaluate_trajectory(pred_json, gt_json, canvas_size=None, threshold=0.05):
    """Compare a predicted stroke trajectory against ground truth.

    pred_json / gt_json: parsed JSON (list of strokes, each a list of [x,y] or
    [x,y,t] points; our {"strokes": [...]} / {label: points} wrappers also work).
    Returns {"dtw_distance": float, "precision": float, "recall": float}.
    """
    pred_points = flatten_trajectory(pred_json, canvas_size=canvas_size)
    gt_points = flatten_trajectory(gt_json, canvas_size=canvas_size)

    return {
        "dtw_distance": dtw_distance(pred_points, gt_points),
        **dict(zip(("precision", "recall"), precision_recall(pred_points, gt_points, threshold=threshold))),
    }


def log_evaluation(metrics, log_path="evaluation_log.csv", extra_fields=None):
    """Append a row of metrics (plus optional extra context fields) to a CSV log,
    writing a header row the first time the file is created."""
    log_path = Path(log_path)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics, **(extra_fields or {})}

    file_exists = log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a predicted stroke trajectory against ground truth.")
    parser.add_argument("--pred", required=True, help="Path to predicted strokes JSON.")
    parser.add_argument("--gt", required=True, help="Path to ground-truth strokes JSON.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Spatial distance threshold for precision/recall.")
    parser.add_argument("--canvas-width", type=float, help="Canvas width for normalization (defaults to each trajectory's own bounding box).")
    parser.add_argument("--canvas-height", type=float, help="Canvas height for normalization.")
    parser.add_argument("--log-path", default="evaluation_log.csv", help="CSV file to append results to.")
    parser.add_argument("--tag", default="", help="Optional label for this run (e.g. adapter/checkpoint name) recorded in the log.")
    return parser


def main():
    args = build_arg_parser().parse_args()

    pred_json = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gt_json = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    canvas_size = (args.canvas_width, args.canvas_height) if args.canvas_width and args.canvas_height else None

    metrics = evaluate_trajectory(pred_json, gt_json, canvas_size=canvas_size, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))

    log_evaluation(metrics, log_path=args.log_path, extra_fields={"pred_path": str(args.pred), "gt_path": str(args.gt), "tag": args.tag})
    print(f"Logged to {args.log_path}")


if __name__ == "__main__":
    main()
