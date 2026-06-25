import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


SPLIT_DIRS = {
    "train": "train",
    "validation": "valid",
    "valid": "valid",
    "val": "valid",
    "test": "test",
}


def detect_runtime(preferred_device="auto"):
    if preferred_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but PyTorch MPS is not available in this process.")
        return torch.device("mps"), torch.float16

    if preferred_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available in this process.")
        return torch.device("cuda"), torch.bfloat16

    if preferred_device == "cpu":
        return torch.device("cpu"), torch.float32

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


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


def load_stroke_json(path, decimals=2):
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

        strokes.append(
            {
                "label": label,
                "points": [normalize_point(point, decimals) for point in points],
            }
        )

    return strokes


class SedrahJsonStrokeDataset(Dataset):
    def __init__(self, data_dir, split="train", max_samples=None, decimals=2):
        if split not in SPLIT_DIRS:
            raise ValueError(f"Invalid split {split!r}. Expected one of {sorted(SPLIT_DIRS)}.")

        self.split = split
        self.json_dir = Path(data_dir) / SPLIT_DIRS[split]
        self.decimals = decimals
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
        strokes = load_stroke_json(path, decimals=self.decimals)

        target = {
            "text": text,
            "sample_index": sample_index,
            "strokes": strokes,
        }

        return {
            "path": str(path),
            "text": text,
            "target_text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        }


class JsonOnlyCollator:
    def __init__(self, tokenizer, max_length=8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_prompt(self, text):
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
                "content": f"Generate the stroke trajectory JSON for this text: {text}",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __call__(self, batch):
        full_texts = []
        prompt_lengths = []

        for item in batch:
            prompt = self.build_prompt(item["text"])
            answer = item["target_text"] + self.tokenizer.eos_token
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_texts.append(prompt + answer)
            prompt_lengths.append(len(prompt_ids))

        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = encoded["input_ids"].clone()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : min(prompt_length, labels.shape[1])] = -100
            labels[row_index, encoded["attention_mask"][row_index] == 0] = -100

        encoded["labels"] = labels
        return encoded


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            steps += 1
    return total_loss / max(steps, 1)


def run_sandbox_pipeline(args):
    device, compute_dtype = detect_runtime(args.device)
    print(f"Using device={device} dtype={compute_dtype}")
    print(
        "MPS status: "
        f"built={torch.backends.mps.is_built()} "
        f"available={torch.backends.mps.is_available()}"
    )

    train_dataset = SedrahJsonStrokeDataset(
        args.data_dir,
        split="train",
        max_samples=args.max_train_samples,
        decimals=args.decimals,
    )
    val_dataset = SedrahJsonStrokeDataset(
        args.data_dir,
        split="validation",
        max_samples=args.max_val_samples,
        decimals=args.decimals,
    )

    if args.dry_run:
        sample = train_dataset[0]
        print("Sample input text:")
        print(sample["text"])
        print("Sample target JSON prefix:")
        print(sample["target_text"][:1200])
        print(f"Target character length: {len(sample['target_text'])}")
        return

    print(f"Loading tokenizer/processor from {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer

    print(f"Loading model from {args.model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
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
    model.print_trainable_parameters()

    collator = JsonOnlyCollator(tokenizer, max_length=args.max_length)
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

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum_steps
            if step % args.log_every == 0 or step == 1:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={loss.item() * args.grad_accum_steps:.4f}"
                )

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate_loss(model, val_loader, device)
        print(
            f"epoch={epoch + 1} complete "
            f"train_loss={avg_train_loss:.4f} "
            f"val_loss={avg_val_loss:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter and tokenizer to {output_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="JSON-only LoRA fine-tuning for Sedrah stroke trajectories.")
    parser.add_argument("--data-dir", default="sedrah_pipeline/sandbox_data/json")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output-dir", default="outputs/qwen2vl-json-strokes-lora")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--decimals", type=int, default=2)
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
