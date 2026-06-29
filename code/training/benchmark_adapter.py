import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

_CODE_ROOT = Path(__file__).resolve().parent.parent  # .../code
sys.path.insert(0, str(_CODE_ROOT / "inference"))
sys.path.insert(0, str(_CODE_ROOT / "training"))

from infer_stroke import extract_json_object, generate_stroke_text, load_model  # noqa: E402
from train_sandbox import SPLIT_DIRS, load_image_lookup, load_stroke_json, parse_sample_id  # noqa: E402
from trajectory_eval import evaluate_trajectory  # noqa: E402


def sample_ground_truth_files(data_dir, split, num_samples, seed):
    split_dir = Path(data_dir) / SPLIT_DIRS[split]
    files = sorted(split_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {split_dir}")

    if num_samples is not None and num_samples < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, num_samples)
    return files


def evaluate_one(model, processor, device, path, image_lookup, args):
    text, sample_index = parse_sample_id(path)
    gt_strokes = load_stroke_json(path, decimals=2, prune_epsilon=0.0)
    gt_json = {"text": text, "sample_index": sample_index, "strokes": gt_strokes}

    image = None
    image_path = image_lookup.get(path.name)
    if image_path:
        image = Image.open(image_path).convert("RGB")

    generated_text = generate_stroke_text(
        model,
        processor,
        text,
        device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        image=image,
    )

    try:
        pred_json = json.loads(extract_json_object(generated_text))
        metrics = evaluate_trajectory(pred_json, gt_json, threshold=args.threshold)
        metrics["parse_ok"] = True
    except (ValueError, json.JSONDecodeError):
        metrics = {"dtw_distance": None, "precision": 0.0, "recall": 0.0, "parse_ok": False}

    metrics["sample"] = path.name
    metrics["has_image"] = image is not None
    return metrics


def summarize(results, args, elapsed):
    valid = [r for r in results if r["parse_ok"]]
    n = len(results)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adapter_dir": str(args.adapter_dir),
        "split": args.split,
        "num_samples": n,
        "parse_success_rate": len(valid) / n if n else 0.0,
        "mean_dtw_distance": sum(r["dtw_distance"] for r in valid) / len(valid) if valid else "",
        "mean_precision": sum(r["precision"] for r in valid) / len(valid) if valid else 0.0,
        "mean_recall": sum(r["recall"] for r in valid) / len(valid) if valid else 0.0,
        "elapsed_seconds": elapsed,
        "tag": args.tag,
    }


def append_csv(path, row):
    path = Path(path)
    file_exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def plot_benchmark_history(log_path, plot_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(log_path).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {log_path} to plot.")

    labels = [f"{Path(r['adapter_dir']).name}\n{r['timestamp'][:10]}" for r in rows]
    dtw = [float(r["mean_dtw_distance"]) if r["mean_dtw_distance"] else float("nan") for r in rows]
    precision = [float(r["mean_precision"]) for r in rows]
    recall = [float(r["mean_recall"]) for r in rows]
    parse_rate = [float(r["parse_success_rate"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(max(6, len(rows) * 1.3), 8), sharex=True)

    axes[0].plot(labels, precision, marker="o", label="precision")
    axes[0].plot(labels, recall, marker="o", label="recall")
    axes[0].plot(labels, parse_rate, marker="o", label="parse success rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("score")
    axes[0].set_title("Adapter benchmark history")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(labels, dtw, marker="o", color="tab:red", label="mean DTW distance")
    axes[1].set_ylabel("DTW distance")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def run_benchmark(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading base model {args.model_id} + adapter {args.adapter_dir} on {device} ...")
    model, processor = load_model(args.model_id, args.adapter_dir, device)
    print("Model ready.")

    image_lookup = load_image_lookup(args.image_manifest) if args.image_manifest else {}

    files = sample_ground_truth_files(args.data_dir, args.split, args.num_samples, args.seed)
    print(f"Evaluating {len(files)} samples from [{args.split}] in {args.data_dir} ...")

    results = []
    start = time.perf_counter()
    for i, path in enumerate(files, start=1):
        metrics = evaluate_one(model, processor, device, path, image_lookup, args)
        results.append(metrics)
        dtw_str = f"{metrics['dtw_distance']:.3f}" if metrics["dtw_distance"] is not None else "n/a"
        print(
            f"[{i}/{len(files)}] {path.name}: parse_ok={metrics['parse_ok']} "
            f"dtw={dtw_str} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"has_image={metrics['has_image']}"
        )
    elapsed = time.perf_counter() - start

    summary = summarize(results, args, elapsed)
    print("\n=== Benchmark summary ===")
    print(json.dumps(summary, indent=2, default=str))

    append_csv(args.log_path, summary)
    print(f"Logged summary to {args.log_path}")

    if args.per_sample_log:
        for r in results:
            append_csv(args.per_sample_log, {"timestamp": summary["timestamp"], "adapter_dir": summary["adapter_dir"], **r})
        print(f"Logged {len(results)} per-sample rows to {args.per_sample_log}")

    if args.plot:
        plot_path = plot_benchmark_history(args.log_path, args.plot)
        print(f"Saved chart to {plot_path}")

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark a trained Sedrah LoRA adapter: sample ground-truth files, generate "
        "predictions, score them (DTW/precision/recall via trajectory_eval), and log + plot the result."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--data-dir", default="sedrah_pipeline/calliar_combined_dataset/json")
    parser.add_argument("--image-manifest", help="Optional image_manifest.jsonl for multimodal evaluation.")
    parser.add_argument("--split", choices=sorted(SPLIT_DIRS), default="validation")
    parser.add_argument("--num-samples", type=int, default=30, help="Random subset size; omit/exceed to use the whole split.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed, for reproducible benchmark sets across runs.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)

    parser.add_argument("--threshold", type=float, default=0.05, help="Spatial distance threshold for precision/recall.")
    parser.add_argument("--log-path", default="adapter_benchmark_log.csv", help="CSV of one aggregated row per benchmark run.")
    parser.add_argument("--per-sample-log", default="adapter_benchmark_per_sample.csv", help="CSV of one row per evaluated sample. Set to '' to disable.")
    parser.add_argument("--plot", default="adapter_benchmark_history.png", help="PNG chart of metrics across all logged runs. Set to '' to disable.")
    parser.add_argument("--tag", default="", help="Optional label for this run (e.g. adapter version) recorded in the log.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not args.tag:
        args.tag = Path(args.adapter_dir).name
    run_benchmark(args)


if __name__ == "__main__":
    main()
