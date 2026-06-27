import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

from train_sandbox import JsonOnlyCollator, SedrahJsonStrokeDataset, detect_runtime


def extract_json_object(text):
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError("Unterminated JSON object in model output.")


def stroke_labels(strokes):
    return [stroke["label"] for stroke in strokes]


def score_sample(predicted_labels, target_labels):
    overlap = Counter(predicted_labels) & Counter(target_labels)
    return {
        "true_positive": sum(overlap.values()),
        "predicted_count": len(predicted_labels),
        "target_count": len(target_labels),
        "exact_match": predicted_labels == target_labels,
    }


def generate_prediction(model, tokenizer, collator, text, device, max_new_tokens):
    prompt = collator.build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_evaluation(args):
    device, compute_dtype = detect_runtime(args.device)
    print(f"Using device={device} dtype={compute_dtype}")

    dataset = SedrahJsonStrokeDataset(args.data_dir, split=args.split, max_samples=args.max_samples)

    print(f"Loading tokenizer from {args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = JsonOnlyCollator(tokenizer, max_length=args.max_length)

    print(f"Loading base model from {args.model_id}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )

    print(f"Loading LoRA adapter from {args.adapter_dir}")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.to(device)
    model.eval()

    totals = Counter()
    exact_matches = 0
    parse_failures = 0
    eval_start = time.perf_counter()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        target = json.loads(sample["target_text"])
        target_labels = stroke_labels(target["strokes"])

        generated_text = generate_prediction(
            model, tokenizer, collator, sample["text"], device, args.max_new_tokens
        )

        try:
            predicted = extract_json_object(generated_text)
            predicted_labels = stroke_labels(predicted["strokes"])
        except (ValueError, KeyError, TypeError):
            parse_failures += 1
            predicted_labels = []

        result = score_sample(predicted_labels, target_labels)
        totals["true_positive"] += result["true_positive"]
        totals["predicted_count"] += result["predicted_count"]
        totals["target_count"] += result["target_count"]
        exact_matches += int(result["exact_match"])

        if (idx + 1) % args.log_every == 0 or idx == 0:
            print(f"[{idx + 1}/{len(dataset)}] text={sample['text']!r} exact_match={result['exact_match']}")

    eval_duration = time.perf_counter() - eval_start
    precision = totals["true_positive"] / totals["predicted_count"] if totals["predicted_count"] else 0.0
    recall = totals["true_positive"] / totals["target_count"] if totals["target_count"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = exact_matches / len(dataset)

    metrics = {
        "split": args.split,
        "num_samples": len(dataset),
        "parse_failures": parse_failures,
        "stroke_label_precision": precision,
        "stroke_label_recall": recall,
        "stroke_label_f1": f1,
        "exact_match_accuracy": accuracy,
        "eval_seconds": eval_duration,
    }
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    metrics_path = Path(args.metrics_output) if args.metrics_output else Path(args.adapter_dir) / f"eval_metrics_{args.split}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Sedrah LoRA adapter by generating stroke JSON and scoring the predicted "
            "stroke label sequence against ground truth (precision/recall on stroke labels, "
            "exact-match accuracy on the full label sequence)."
        )
    )
    parser.add_argument("--data-dir", default="sedrah_pipeline/calliar_dataset/json")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter-dir", default="outputs/qwen2vl-sedrah-stroke-lora_v1")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="cuda")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--metrics-output")
    return parser


if __name__ == "__main__":
    run_evaluation(build_arg_parser().parse_args())
