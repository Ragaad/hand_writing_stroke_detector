import argparse
import json
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


SPLIT_DIRS = {
    "train": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "test": "test",
}

def resolve_runtime(accelerator, preferred_device="auto"):
    """Resolve device/dtype from the Accelerator, which already handles per-process
    GPU assignment under `accelerate launch`/`torchrun`. --device only validates or
    forces a backend; it no longer picks the device by hand."""
    device = accelerator.device

    if preferred_device == "mps" and device.type != "mps":
        raise RuntimeError("Requested --device mps, but the resolved Accelerate device is not mps.")
    if preferred_device == "cuda" and device.type != "cuda":
        raise RuntimeError("Requested --device cuda, but the resolved Accelerate device is not cuda.")

    if device.type == "cuda":
        compute_dtype = torch.bfloat16
    elif device.type == "mps":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    return device, compute_dtype



def load_image_lookup(manifest_path):
    """Load a {json_filename: image_path} lookup from an image_manifest.jsonl
    (rows like {"json": "...", "image": "..."}). Keyed by filename only, since
    callers already scope datasets by split directory."""
    lookup = {}
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lookup[Path(row["json"]).name] = row["image"]
    return lookup


def parse_sample_id(path):
    stem = path.stem
    if "_" not in stem:
        return stem.strip(), None

    text, sample_index = stem.rsplit("_", 1)
    if sample_index.isdigit():
        return text.strip(), int(sample_index)
    return stem.strip(), None


def normalize_point(point, decimals):
    if not (
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
    ):
        raise ValueError(f"Invalid trajectory point: {point!r}")

    return [round(float(point[0]), decimals), round(float(point[1]), decimals)]


def _perpendicular_distance(point, start, end):
    if start == end:
        return math.hypot(point[0] - start[0], point[1] - start[1])

    x1, y1 = start
    x2, y2 = end
    x0, y0 = point
    num = abs((x2 - x1) * (y0 - y1) - (x0 - x1) * (y2 - y1))
    den = math.hypot(x2 - x1, y2 - y1)
    return num / den


def rdp_simplify(points, epsilon):
    """Ramer-Douglas-Peucker simplification, same epsilon convention as the Calliar dataset generator."""
    if epsilon <= 0 or len(points) < 3:
        return points

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start_idx, end_idx = stack.pop()
        start, end = points[start_idx], points[end_idx]
        max_dist, max_idx = -1.0, -1
        for i in range(start_idx + 1, end_idx):
            dist = _perpendicular_distance(points[i], start, end)
            if dist > max_dist:
                max_dist, max_idx = dist, i

        if max_dist > epsilon:
            keep[max_idx] = True
            stack.append((start_idx, max_idx))
            stack.append((max_idx, end_idx))

    return [point for point, kept in zip(points, keep) if kept]


def load_stroke_json(path, decimals=2, prune_epsilon=0.0):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a list of stroke objects.")

    strokes = []
    for stroke_index, stroke in enumerate(raw):
        if not isinstance(stroke, dict) or len(stroke) != 1:
            raise ValueError(f"{path} stroke {stroke_index} must be a one-key object.")

        label, points = next(iter(stroke.items()))
        if not isinstance(points, list):
            raise ValueError(f"{path} stroke {stroke_index} points must be a list.")

        points = rdp_simplify(points, prune_epsilon)

        strokes.append(
            {
                "label": label,
                "points": [normalize_point(point, decimals) for point in points],
            }
        )

    return strokes


class SedrahJsonStrokeDataset(Dataset):
    def __init__(self, data_dir, split="train", max_samples=None, decimals=2, prune_epsilon=0.0, image_lookup=None):
        if split not in SPLIT_DIRS:
            raise ValueError(f"Invalid split {split!r}. Expected one of {sorted(SPLIT_DIRS)}.")

        self.split = split
        self.json_dir = Path(data_dir) / SPLIT_DIRS[split]
        self.decimals = decimals
        self.prune_epsilon = prune_epsilon
        self.image_lookup = image_lookup or {}
        self.files = sorted(self.json_dir.glob("*.json"))

        if max_samples is not None:
            self.files = self.files[:max_samples]

        if not self.files:
            raise FileNotFoundError(f"No JSON files found in {self.json_dir}")

        print(f"Registered {len(self.files)} JSON stroke files for split [{split}] from {self.json_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        text, sample_index = parse_sample_id(path)
        strokes = load_stroke_json(path, decimals=self.decimals, prune_epsilon=self.prune_epsilon)

        target = {
            "text": text,
            "sample_index": sample_index,
            "strokes": strokes,
        }

        return {
            "path": str(path),
            "text": text,
            "target_text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            "image_path": self.image_lookup.get(path.name),
        }


class JsonOnlyCollator:
    def __init__(self, processor_or_tokenizer, max_length=8192):
        # Accepts either a full AutoProcessor (for image+text batches) or a bare
        # tokenizer (text-only; kept for backward compatibility with infer_stroke.py).
        if hasattr(processor_or_tokenizer, "tokenizer"):
            self.processor = processor_or_tokenizer
            self.tokenizer = processor_or_tokenizer.tokenizer
        else:
            self.processor = processor_or_tokenizer
            self.tokenizer = processor_or_tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, text, has_image=False):
        user_content = []
        if has_image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": f"Generate the stroke trajectory JSON for this text: {text}"})

        messages = [
            {
                "role": "system",
                "content": (
                    "You generate compact, valid JSON for Arabic calligraphy stroke trajectories. "
                    "Return only JSON with keys text, sample_index, and strokes."
                ),
            },
            {
                "role": "user",
                "content": user_content if has_image else user_content[-1]["text"],
            },
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _encode_one(self, item):
        image_path = item.get("image_path")
        image = Image.open(image_path).convert("RGB") if image_path else None
        prompt = self.build_prompt(item["text"], has_image=image is not None)
        answer = item["target_text"] + self.tokenizer.eos_token
        full_text = prompt + answer

        if image is not None:
            encoded = self.processor(text=[full_text], images=[image], return_tensors="pt")
            prompt_len = self.processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
        else:
            encoded = self.tokenizer(full_text, return_tensors="pt")
            prompt_len = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]

        return encoded, prompt_len

    def __call__(self, batch):
        encoded_items = [self._encode_one(item) for item in batch]

        max_len = min(max(enc["input_ids"].shape[1] for enc, _ in encoded_items), self.max_length)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full((len(encoded_items), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded_items), max_len), dtype=torch.long)
        labels = torch.full((len(encoded_items), max_len), -100, dtype=torch.long)
        mm_token_type_ids = torch.zeros((len(encoded_items), max_len), dtype=torch.long)
        pixel_values_list, image_grid_thw_list = [], []
        has_image = False

        for row_index, (encoded, prompt_len) in enumerate(encoded_items):
            ids = encoded["input_ids"][0][:max_len]
            length = ids.shape[0]
            input_ids[row_index, :length] = ids
            attention_mask[row_index, :length] = 1

            label_row = ids.clone()
            label_row[: min(prompt_len, length)] = -100
            labels[row_index, :length] = label_row

            if "pixel_values" in encoded:
                has_image = True
                pixel_values_list.append(encoded["pixel_values"])
                image_grid_thw_list.append(encoded["image_grid_thw"])
                mm_token_type_ids[row_index, :length] = encoded["mm_token_type_ids"][0][:max_len]

        result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if has_image:
            result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
            result["mm_token_type_ids"] = mm_token_type_ids
        return result


def evaluate_loss(model, loader, accelerator):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
    local_avg = total_loss / max(steps, 1)
    reduced = accelerator.reduce(torch.tensor(local_avg, device=accelerator.device), reduction="mean")
    return reduced.item()


def run_sandbox_pipeline(args):
    accelerator = Accelerator(cpu=(args.device == "cpu"))
    device, compute_dtype = resolve_runtime(accelerator, args.device)
    if accelerator.is_main_process:
        print(
            f"Using device={device} dtype={compute_dtype} "
            f"num_processes={accelerator.num_processes}"
        )
        print(
            "MPS status: "
            f"built={torch.backends.mps.is_built()} "
            f"available={torch.backends.mps.is_available()}"
        )

    image_lookup = load_image_lookup(args.image_manifest) if args.image_manifest else {}
    if accelerator.is_main_process and args.image_manifest:
        print(f"Loaded image manifest with {len(image_lookup)} entries from {args.image_manifest}")

    train_dataset = SedrahJsonStrokeDataset(
        args.data_dir,
        split="train",
        max_samples=args.max_train_samples,
        decimals=args.decimals,
        prune_epsilon=args.prune_epsilon,
        image_lookup=image_lookup,
    )
    val_dataset = SedrahJsonStrokeDataset(
        args.data_dir,
        split="validation",
        max_samples=args.max_val_samples,
        decimals=args.decimals,
        prune_epsilon=args.prune_epsilon,
        image_lookup=image_lookup,
    )

    if args.dry_run:
        if accelerator.is_main_process:
            sample = train_dataset[0]
            print("Sample input text:")
            print(sample["text"])
            print("Sample target JSON prefix:")
            print(sample["target_text"][:1200])
            print(f"Target character length: {len(sample['target_text'])}")
        return

    if accelerator.is_main_process:
        print(f"Loading tokenizer/processor from {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer

    if accelerator.is_main_process:
        print(f"Loading model from {args.model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    collator = JsonOnlyCollator(processor, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    epoch_durations = []
    samples_seen = 0
    run_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        epoch_start = time.perf_counter()
        step_start = epoch_start

        for step, batch in enumerate(train_loader, start=1):
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            accelerator.backward(loss)

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum_steps
            samples_seen += batch["input_ids"].shape[0]
            if accelerator.is_main_process and (step % args.log_every == 0 or step == 1):
                step_duration = time.perf_counter() - step_start
                local_samples = (args.log_every if step != 1 else 1) * args.batch_size
                samples_per_sec = local_samples * accelerator.num_processes / max(step_duration, 1e-9)
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={loss.item() * args.grad_accum_steps:.4f} "
                    f"samples_per_sec={samples_per_sec:.2f}"
                )
                step_start = time.perf_counter()

        epoch_duration = time.perf_counter() - epoch_start
        epoch_durations.append(epoch_duration)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_loss = accelerator.reduce(
            torch.tensor(avg_train_loss, device=accelerator.device), reduction="mean"
        ).item()
        avg_val_loss = evaluate_loss(model, val_loader, accelerator)
        if accelerator.is_main_process:
            print(
                f"epoch={epoch + 1} complete "
                f"train_loss={avg_train_loss:.4f} "
                f"val_loss={avg_val_loss:.4f} "
                f"epoch_time={epoch_duration:.2f}s"
            )

    total_duration = time.perf_counter() - run_start
    total_samples_seen = accelerator.reduce(
        torch.tensor(float(samples_seen), device=accelerator.device), reduction="sum"
    ).item()

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        benchmark = {
            "device": str(device),
            "compute_dtype": str(compute_dtype),
            "num_processes": accelerator.num_processes,
            "epochs": args.epochs,
            "train_samples": len(train_dataset),
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "total_seconds": total_duration,
            "epoch_seconds": epoch_durations,
            "avg_epoch_seconds": sum(epoch_durations) / len(epoch_durations),
            "samples_per_sec_avg": total_samples_seen / total_duration,
        }
        print(f"Benchmark: {json.dumps(benchmark, indent=2)}")
        with (output_dir / "benchmark.json").open("w", encoding="utf-8") as f:
            json.dump(benchmark, f, indent=2)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        print(f"Saved LoRA adapter and tokenizer to {output_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="JSON-only LoRA fine-tuning for Sedrah stroke trajectories.")
    parser.add_argument("--data-dir", default="sedrah_pipeline/calliar_dataset/json")
    parser.add_argument(
        "--image-manifest",
        help="Optional path to an image_manifest.jsonl ({\"json\":..., \"image\":...} rows). "
        "Samples with a matching entry train multimodally (image+text -> stroke JSON); "
        "samples without one fall back to text-only.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output-dir", default="outputs/qwen2vl-json-strokes-lora")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument(
        "--prune-epsilon",
        type=float,
        default=2.0,
        help="RDP simplification epsilon applied to each stroke's points before tokenization. "
        "Same technique and default as the Calliar dataset's own .npz generator; cuts target "
        "JSON length drastically and fixes most --max-length truncation. Set to 0 to disable.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run_sandbox_pipeline(build_arg_parser().parse_args())
