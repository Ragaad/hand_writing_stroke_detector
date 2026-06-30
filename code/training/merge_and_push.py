"""Merge a Sedrah LoRA adapter into the base model and optionally push to HuggingFace Hub."""

import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def merge_adapter(model_id, adapter_dir, output_dir, dtype_str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    print(f"Loading base model: {model_id}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )

    print(f"Loading adapter: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("Merging adapter weights into base model ...")
    merged = peft_model.merge_and_unload()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)

    print(f"Saving processor to: {output_dir}")
    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(output_dir)

    print("Done.")
    return merged, processor


def push_to_hub(output_dir, repo_id, token, private):
    from huggingface_hub import HfApi

    output_dir = Path(output_dir)
    api = HfApi(token=token)

    print(f"Creating repo: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"Uploading {output_dir} → {repo_id} ...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add Sedrah merged model from {Path(output_dir).name}",
    )
    print(f"Pushed to https://huggingface.co/{repo_id}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Merge a Sedrah LoRA adapter into Qwen2-VL-2B and (optionally) push to HF Hub."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct", help="Base model HF repo ID.")
    parser.add_argument("--adapter-dir", required=True, help="Path to the trained LoRA adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Local path to save the merged model.")
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="dtype for loading (bf16 recommended on A100/H100).",
    )

    hub = parser.add_argument_group("HuggingFace Hub (optional)")
    hub.add_argument("--push-to-hub", action="store_true", help="Upload the merged model to HF Hub after saving.")
    hub.add_argument("--repo-id", default="", help="HF Hub repo, e.g. 'Ragaad/sedrah-arabic-stroke-v7'.")
    hub.add_argument(
        "--hf-token",
        default="",
        help="HF token. Falls back to HF_TOKEN env var, then $HOME/.huggingface/token.",
    )
    hub.add_argument("--private", action="store_true", help="Create the HF Hub repo as private.")
    return parser


def main():
    args = build_parser().parse_args()

    # Resolve HF token (arg > env > file) — never echo it
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HF_TOCKEN", "")
    if not token:
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()

    if token:
        os.environ["HF_TOKEN"] = token

    merge_adapter(args.model_id, args.adapter_dir, args.output_dir, args.dtype)

    if args.push_to_hub:
        if not args.repo_id:
            print("ERROR: --repo-id is required when --push-to-hub is set.")
            sys.exit(1)
        if not token:
            print("ERROR: No HF token found. Pass --hf-token, set HF_TOKEN env var, or run `huggingface-cli login`.")
            sys.exit(1)
        push_to_hub(args.output_dir, args.repo_id, token, args.private)


if __name__ == "__main__":
    main()
