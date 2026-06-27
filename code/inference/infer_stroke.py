import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

_CODE_ROOT = Path(__file__).resolve().parent.parent  # .../code
sys.path.insert(0, str(_CODE_ROOT / "data_augmentation"))
sys.path.insert(0, str(_CODE_ROOT / "training"))

import render_json_overlay as rjo
from train_sandbox import JsonOnlyCollator


def extract_json_object(text):
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError(
        "Generation looks truncated: no closing brace found. Try a larger --max-new-tokens."
    )


def load_model(model_id, adapter_dir, device):
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_stroke_text(
    model, tokenizer, text, device, max_new_tokens, do_sample, temperature, top_p, repetition_penalty, no_repeat_ngram_size
):
    collator = JsonOnlyCollator(tokenizer)
    prompt = collator.build_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def classify_text(text):
    stripped = text.strip()
    if " " in stripped or "\t" in stripped:
        return "sentences"
    if len(stripped) <= 1:
        return "letters"
    return "words"


def render_strokes(strokes, output_path, canvas_size, fit_coords, labels):
    render_args = argparse.Namespace(
        alpha=210,
        line_width=2,
        point_radius=2,
        labels=labels,
        max_points_drawn=64,
    )
    font = rjo.load_font(14)
    image_size = (canvas_size, canvas_size)
    transformed = rjo.transform_strokes(strokes, image_size, fit_coords=fit_coords, padding=16)

    canvas = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(canvas)
    rjo.draw_strokes(draw, transformed, render_args, font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_one(model, tokenizer, device, text, args, index):
    print(f"\n>>> Generating strokes for: {text!r}")
    generated_text = generate_stroke_text(
        model,
        tokenizer,
        text,
        device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = "".join(c if c.isalnum() else "_" for c in text)[:40] or "sample"
    base_name = f"{index:03d}_{safe_text}_{timestamp}"
    category_dir = Path(args.output_dir) / classify_text(text)

    raw_path = category_dir / f"{base_name}_raw.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(generated_text, encoding="utf-8")

    try:
        json_str = extract_json_object(generated_text)
        parsed = json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"FAILED to parse model output as JSON: {exc}")
        print(f"Raw output saved to: {raw_path}")
        return

    if not (isinstance(parsed, dict) and isinstance(parsed.get("strokes"), list)):
        print("Parsed JSON does not contain a 'strokes' list.")
        print(f"Raw output saved to: {raw_path}")
        return

    strokes = rjo.normalize_strokes(parsed)

    json_path = category_dir / f"{base_name}.json"
    json_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    png_path = category_dir / f"{base_name}.png"
    render_strokes(
        strokes,
        png_path,
        canvas_size=args.canvas_size,
        fit_coords=not args.no_fit_coords,
        labels=not args.no_labels,
    )

    total_points = sum(len(stroke["points"]) for stroke in strokes)
    print(f"Model echoed text: {parsed.get('text', '')!r}")
    print(f"Strokes: {len(strokes)}  Total points: {total_points}")
    print(f"Saved rendering: {png_path}")
    print(f"Saved raw JSON: {json_path}")


def main():
    args = build_arg_parser().parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading base model {args.model_id} + adapter {args.adapter_dir} on {device} ...")
    model, tokenizer = load_model(args.model_id, args.adapter_dir, device)
    print("Model ready.")

    if args.text:
        run_one(model, tokenizer, device, args.text, args, index=0)
        return

    print("Interactive mode. Type Arabic text (a letter, word, or phrase) and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")
    index = 0
    while True:
        try:
            text = input("Arabic text> ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        index += 1
        run_one(model, tokenizer, device, text, args, index=index)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Test stroke-JSON inference for a trained Sedrah LoRA adapter and render the result."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter-dir", default="outputs/qwen2vl-sedrah-stroke-lora_v4")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--text", help="Single Arabic text to render once, instead of entering interactive mode.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalize repeated tokens. WARNING: JSON coordinate output is inherently repetitive "
        "(brackets, commas, similar numbers); values above 1.0 tend to corrupt valid syntax "
        "(e.g. spurious spaces inside numbers). Only raise this if you hit a genuine generation "
        "loop, and prefer --do-sample first.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Hard-block repeating n-grams of this size; 0 disables.",
    )
    parser.add_argument(
        "--output-dir",
        default="inference-results",
        help="Root folder; results are split into letters/words/sentences subfolders by input type.",
    )
    parser.add_argument("--canvas-size", type=int, default=600, help="Square canvas size in pixels for rendering.")
    parser.add_argument(
        "--no-fit-coords",
        action="store_true",
        help="Disable auto-scaling strokes to fit the canvas; render raw model coordinates.",
    )
    parser.add_argument("--no-labels", action="store_true", help="Disable drawing character labels next to each stroke.")
    return parser


if __name__ == "__main__":
    main()
