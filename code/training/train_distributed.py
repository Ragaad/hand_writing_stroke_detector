import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from train_sandbox import JsonOnlyCollator, SedrahJsonStrokeDataset


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    compute_dtype: torch.dtype


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_dtype(dtype_name, device):
    if dtype_name == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    if dtype_name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {dtype_name!r}")


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        if distributed:
            raise RuntimeError("MPS does not support this torch.distributed training path.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    compute_dtype = resolve_dtype(args.dtype, device)
    if distributed:
        backend = args.dist_backend
        if backend == "auto":
            backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method=args.dist_url)

    return DistributedContext(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        compute_dtype=compute_dtype,
    )


def cleanup_distributed(ctx):
    if ctx.distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(ctx):
    return ctx.rank == 0


def rank_zero_print(ctx, message):
    if is_main_process(ctx):
        print(message, flush=True)


def barrier(ctx):
    if ctx.distributed:
        dist.barrier()


def reduce_sum(ctx, values):
    tensor = torch.tensor(values, device=ctx.device, dtype=torch.float64)
    if ctx.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def seed_everything(seed, rank):
    final_seed = seed + rank
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(final_seed)


def load_tokenizer(args):
    try:
        processor = AutoProcessor.from_pretrained(args.model_id)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer
    except Exception:
        if args.model_family == "qwen2vl":
            raise
    return AutoTokenizer.from_pretrained(args.model_id)


def load_base_model(args, ctx):
    model_kwargs = {
        "dtype": ctx.compute_dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.model_family == "qwen2vl":
        return Qwen2VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    if args.model_family == "causal-lm":
        return AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    raise ValueError(f"Unsupported --model-family {args.model_family!r}")


def build_model(args, ctx):
    base_model = load_base_model(args, ctx)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(base_model, args.resume_from_checkpoint, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=parse_csv(args.target_modules),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, peft_config)

    model.to(ctx.device)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def build_loader(dataset, collator, args, ctx, split):
    is_train = split == "train"
    sampler = None
    shuffle = is_train
    if ctx.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=is_train,
            drop_last=args.drop_last if is_train else False,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=ctx.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    return loader, sampler


def move_batch_to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def evaluate_loss(model, loader, ctx):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, ctx.device)
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_steps += 1

    reduced_loss, reduced_steps = reduce_sum(ctx, [total_loss, total_steps])
    model.train()
    return reduced_loss / max(reduced_steps, 1.0)


def load_trainer_state(checkpoint_dir, ctx):
    if not checkpoint_dir:
        return 0, 0, 0, None

    state_path = Path(checkpoint_dir) / "trainer_state.pt"
    if not state_path.exists():
        rank_zero_print(ctx, f"No trainer_state.pt found in {checkpoint_dir}; loading adapter weights only.")
        return 0, 0, 0, None

    state = torch.load(state_path, map_location="cpu")
    return (
        int(state.get("epoch", 0)),
        int(state.get("step_in_epoch", 0)),
        int(state.get("global_step", 0)),
        state.get("optimizer"),
    )


def save_checkpoint(model, tokenizer, optimizer, args, ctx, epoch, step_in_epoch, global_step, name):
    if not is_main_process(ctx):
        return

    checkpoint_root = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.output_dir) / "checkpoints"
    checkpoint_dir = checkpoint_root / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = unwrap_model(model)
    unwrapped.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "world_size": ctx.world_size,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_dir / "trainer_state.pt",
    )


def save_final_model(model, tokenizer, args, ctx, benchmark):
    if not is_main_process(ctx):
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    unwrapped = unwrap_model(model)
    unwrapped.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    rank_zero_print(ctx, f"Saved LoRA adapter, tokenizer, and benchmark to {output_dir}")


def train(args):
    ctx = setup_distributed(args)
    try:
        seed_everything(args.seed, ctx.rank)
        rank_zero_print(
            ctx,
            (
                f"distributed={ctx.distributed} rank={ctx.rank} local_rank={ctx.local_rank} "
                f"world_size={ctx.world_size} device={ctx.device} dtype={ctx.compute_dtype}"
            ),
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

        tokenizer = load_tokenizer(args)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        collator = JsonOnlyCollator(tokenizer, max_length=args.max_length)

        if args.dry_run:
            if is_main_process(ctx):
                sample = train_dataset[0]
                print("Sample input text:")
                print(sample["text"])
                print("Sample target JSON prefix:")
                print(sample["target_text"][:1200])
                print(f"Target character length: {len(sample['target_text'])}")
            return

        model = build_model(args, ctx)
        if is_main_process(ctx) and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        if ctx.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                output_device=ctx.local_rank if ctx.device.type == "cuda" else None,
                find_unused_parameters=args.ddp_find_unused_parameters,
            )

        train_loader, train_sampler = build_loader(train_dataset, collator, args, ctx, "train")
        val_loader, _ = build_loader(val_dataset, collator, args, ctx, "validation")
        optimizer = torch.optim.AdamW(unwrap_model(model).parameters(), lr=args.lr, weight_decay=args.weight_decay)

        start_epoch, resume_step, global_step, optimizer_state = load_trainer_state(args.resume_from_checkpoint, ctx)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            rank_zero_print(
                ctx,
                (
                    f"Resumed optimizer/trainer state from {args.resume_from_checkpoint} "
                    f"at epoch={start_epoch} step={resume_step} global_step={global_step}"
                ),
            )

        effective_global_batch = args.batch_size * args.grad_accum_steps * ctx.world_size
        benchmark = {
            "run_id": args.run_id,
            "model_id": args.model_id,
            "model_family": args.model_family,
            "distributed": ctx.distributed,
            "world_size": ctx.world_size,
            "device": str(ctx.device),
            "compute_dtype": str(ctx.compute_dtype),
            "epochs": args.epochs,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "batch_size_per_gpu": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_global_batch_size": effective_global_batch,
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "epoch_metrics": [],
        }

        samples_seen = 0
        run_start = time.perf_counter()
        stop_training = False

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            total_steps = 0
            epoch_start = time.perf_counter()
            step_start = epoch_start
            window_steps = 0

            for step, batch in enumerate(train_loader, start=1):
                if epoch == start_epoch and step <= resume_step:
                    continue

                batch = move_batch_to_device(batch, ctx.device)
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()

                did_update = step % args.grad_accum_steps == 0 or step == len(train_loader)
                if did_update:
                    nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                loss_value = loss.item() * args.grad_accum_steps
                total_loss += loss_value
                total_steps += 1
                samples_seen += batch["input_ids"].shape[0]
                window_steps += 1

                if step % args.log_every == 0 or step == 1:
                    elapsed = time.perf_counter() - step_start
                    window_samples = window_steps * args.batch_size * ctx.world_size
                    samples_per_sec = window_samples / max(elapsed, 1e-9)
                    rank_zero_print(
                        ctx,
                        (
                            f"epoch={epoch + 1}/{args.epochs} step={step}/{len(train_loader)} "
                            f"global_step={global_step} loss={loss_value:.4f} "
                            f"samples_per_sec={samples_per_sec:.2f}"
                        ),
                    )
                    step_start = time.perf_counter()
                    window_steps = 0

                if args.save_every_n_steps and did_update and global_step % args.save_every_n_steps == 0:
                    save_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        args,
                        ctx,
                        epoch,
                        step,
                        global_step,
                        f"step-{global_step:08d}",
                    )

                if args.max_train_steps and global_step >= args.max_train_steps:
                    stop_training = True
                    break

            resume_step = 0
            reduced_loss, reduced_steps = reduce_sum(ctx, [total_loss, total_steps])
            avg_train_loss = reduced_loss / max(reduced_steps, 1.0)
            avg_val_loss = evaluate_loss(model, val_loader, ctx)
            epoch_seconds = time.perf_counter() - epoch_start
            epoch_metric = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "epoch_seconds": epoch_seconds,
                "global_step": global_step,
            }
            benchmark["epoch_metrics"].append(epoch_metric)
            rank_zero_print(
                ctx,
                (
                    f"epoch={epoch + 1} complete train_loss={avg_train_loss:.4f} "
                    f"val_loss={avg_val_loss:.4f} epoch_time={epoch_seconds:.2f}s"
                ),
            )

            if args.save_every_epoch:
                save_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    args,
                    ctx,
                    epoch + 1,
                    0,
                    global_step,
                    f"epoch-{epoch + 1:04d}",
                )

            if stop_training:
                break

        total_seconds = time.perf_counter() - run_start
        local_samples = torch.tensor([samples_seen], device=ctx.device, dtype=torch.float64)
        if ctx.distributed:
            dist.all_reduce(local_samples, op=dist.ReduceOp.SUM)
        benchmark.update(
            {
                "total_seconds": total_seconds,
                "samples_seen": int(local_samples.item()),
                "samples_per_sec_avg": float(local_samples.item()) / max(total_seconds, 1e-9),
                "avg_epoch_seconds": (
                    sum(item["epoch_seconds"] for item in benchmark["epoch_metrics"])
                    / max(len(benchmark["epoch_metrics"]), 1)
                ),
                "global_steps": global_step,
            }
        )
        rank_zero_print(ctx, f"Benchmark: {json.dumps(benchmark, indent=2)}")
        save_final_model(model, tokenizer, args, ctx, benchmark)
        barrier(ctx)
    finally:
        cleanup_distributed(ctx)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Distributed JSON-only LoRA fine-tuning.")
    parser.add_argument("--data-dir", default="sedrah_pipeline/calliar_dataset/json")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--model-family", choices=["qwen2vl", "causal-lm"], default="qwen2vl")
    parser.add_argument("--output-dir", default="outputs/qwen2vl-json-strokes-lora-ddp")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="auto")
    parser.add_argument("--dist-backend", choices=["auto", "nccl", "gloo"], default="auto")
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every-epoch", type=str_to_bool, default=True)
    parser.add_argument("--save-every-n-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--ddp-find-unused-parameters", type=str_to_bool, default=True)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
