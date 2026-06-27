import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_gpu_counts(value):
    counts = []
    for item in value.split(","):
        item = item.strip()
        if item:
            counts.append(int(item))
    if not counts:
        raise argparse.ArgumentTypeError("At least one GPU count is required.")
    return counts


def detect_gpu_count():
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def default_torchrun():
    sibling = Path(sys.executable).with_name("torchrun")
    if sibling.exists():
        return str(sibling)
    return "torchrun"


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def init_db(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_group TEXT NOT NULL,
                gpu_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                command_json TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                log_path TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                return_code INTEGER,
                total_seconds REAL,
                train_samples INTEGER,
                val_samples INTEGER,
                effective_global_batch_size INTEGER,
                samples_per_sec_avg REAL,
                avg_epoch_seconds REAL,
                final_train_loss REAL,
                final_val_loss REAL,
                benchmark_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                val_loss REAL,
                epoch_seconds REAL,
                global_step INTEGER,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
            """
        )


def insert_run(db_path, run_group, gpu_count, command, output_dir, log_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs (
                run_group, gpu_count, status, command_json, output_dir, log_path, started_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_group,
                gpu_count,
                "running",
                json.dumps(command),
                str(output_dir),
                str(log_path),
                utc_now(),
            ),
        )
        return cursor.lastrowid


def update_run(db_path, run_id, status, return_code, benchmark):
    epoch_metrics = benchmark.get("epoch_metrics") or []
    final_epoch = epoch_metrics[-1] if epoch_metrics else {}
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM epochs WHERE run_id = ?", (run_id,))
        for item in epoch_metrics:
            conn.execute(
                """
                INSERT INTO epochs (
                    run_id, epoch, train_loss, val_loss, epoch_seconds, global_step
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    item.get("epoch"),
                    item.get("train_loss"),
                    item.get("val_loss"),
                    item.get("epoch_seconds"),
                    item.get("global_step"),
                ),
            )

        conn.execute(
            """
            UPDATE runs
            SET status = ?,
                finished_at = ?,
                return_code = ?,
                total_seconds = ?,
                train_samples = ?,
                val_samples = ?,
                effective_global_batch_size = ?,
                samples_per_sec_avg = ?,
                avg_epoch_seconds = ?,
                final_train_loss = ?,
                final_val_loss = ?,
                benchmark_json = ?
            WHERE id = ?
            """,
            (
                status,
                utc_now(),
                return_code,
                benchmark.get("total_seconds"),
                benchmark.get("train_samples"),
                benchmark.get("val_samples"),
                benchmark.get("effective_global_batch_size"),
                benchmark.get("samples_per_sec_avg"),
                benchmark.get("avg_epoch_seconds"),
                final_epoch.get("train_loss"),
                final_epoch.get("val_loss"),
                json.dumps(benchmark, ensure_ascii=False),
                run_id,
            ),
        )


def update_failed_run(db_path, run_id, status, return_code):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = ?, finished_at = ?, return_code = ?
            WHERE id = ?
            """,
            (status, utc_now(), return_code, run_id),
        )


def read_benchmark(output_dir):
    benchmark_path = Path(output_dir) / "benchmark.json"
    if not benchmark_path.exists():
        return {}
    with benchmark_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_grad_accum(args, gpu_count):
    if args.grad_accum_steps:
        return args.grad_accum_steps

    denom = gpu_count * args.batch_size
    if args.target_effective_batch % denom != 0:
        raise ValueError(
            f"target effective batch {args.target_effective_batch} is not divisible by "
            f"gpu_count * batch_size ({gpu_count} * {args.batch_size} = {denom})."
        )
    return max(args.target_effective_batch // denom, 1)


def build_train_command(args, gpu_count, grad_accum, output_dir, run_id):
    master_port = find_free_port()
    command = [
        args.torchrun,
        "--nnodes=1",
        f"--nproc-per-node={gpu_count}",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        args.train_script,
        "--run-id",
        run_id,
    ]
    command.extend(
        [
            "--data-dir",
            args.data_dir,
            "--model-id",
            args.model_id,
            "--model-family",
            args.model_family,
            "--output-dir",
            str(output_dir),
            "--checkpoint-dir",
            str(output_dir / "checkpoints"),
            "--device",
            "cuda",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(grad_accum),
            "--lr",
            str(args.lr),
            "--max-length",
            str(args.max_length),
            "--decimals",
            str(args.decimals),
            "--lora-rank",
            str(args.lora_rank),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--target-modules",
            args.target_modules,
            "--log-every",
            str(args.log_every),
            "--save-every-epoch",
            "true",
            "--ddp-find-unused-parameters",
            str(args.ddp_find_unused_parameters).lower(),
            "--seed",
            str(args.seed),
        ]
    )

    if args.max_train_samples is not None:
        command.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_val_samples is not None:
        command.extend(["--max-val-samples", str(args.max_val_samples)])
    if args.max_train_steps:
        command.extend(["--max-train-steps", str(args.max_train_steps)])
    if args.num_workers:
        command.extend(["--num-workers", str(args.num_workers)])
    return command


def run_one(args, gpu_count):
    grad_accum = compute_grad_accum(args, gpu_count)
    run_id = f"{args.run_group}-gpus-{gpu_count}"
    output_dir = Path(args.base_output_dir) / args.run_group / f"gpus_{gpu_count}"
    log_path = Path(args.log_dir) / args.run_group / f"gpus_{gpu_count}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_train_command(args, gpu_count, grad_accum, output_dir, run_id)
    run_db_id = insert_run(args.db_path, args.run_group, gpu_count, command, output_dir, log_path)

    env = os.environ.copy()
    if args.set_cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))

    print(
        f"[{utc_now()}] starting gpu_count={gpu_count} grad_accum={grad_accum} "
        f"effective_batch={gpu_count * args.batch_size * grad_accum}"
    )
    print(" ".join(command))

    if args.dry_run:
        update_failed_run(args.db_path, run_db_id, "dry_run", 0)
        return 0

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=args.workdir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = process.wait()
    elapsed = time.perf_counter() - start

    benchmark = read_benchmark(output_dir)
    if benchmark:
        benchmark["launcher_elapsed_seconds"] = elapsed
        status = "succeeded" if return_code == 0 else "failed"
        update_run(args.db_path, run_db_id, status, return_code, benchmark)
    else:
        status = "failed_no_benchmark" if return_code == 0 else "failed"
        update_failed_run(args.db_path, run_db_id, status, return_code)

    print(f"[{utc_now()}] finished gpu_count={gpu_count} status={status} return_code={return_code}")
    print(f"log: {log_path}")
    print(f"output: {output_dir}")
    return return_code


def build_arg_parser():
    default_group = datetime.now(timezone.utc).strftime("scaling-%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Run gradual distributed training benchmarks and store them in SQLite.")
    parser.add_argument("--gpu-counts", type=parse_gpu_counts, default=parse_gpu_counts("1,2,4,8"))
    parser.add_argument("--target-effective-batch", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--run-group", default=default_group)
    parser.add_argument("--db-path", type=Path, default=Path("benchmarks/training_benchmarks.sqlite"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/distributed_scaling"))
    parser.add_argument("--base-output-dir", type=Path, default=Path("outputs/distributed_scaling"))
    parser.add_argument("--workdir", default=str(Path.cwd()))
    parser.add_argument("--torchrun", default=default_torchrun())
    parser.add_argument(
        "--train-script",
        default=str(Path(__file__).resolve().parent / "train_distributed.py"),
    )
    parser.add_argument("--set-cuda-visible-devices", action="store_true", default=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--data-dir", default="sedrah_pipeline/calliar_dataset/json")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--model-family", choices=["qwen2vl", "causal-lm"], default="qwen2vl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp-find-unused-parameters", action="store_true", default=True)
    return parser


def main():
    args = build_arg_parser().parse_args()
    available_gpus = detect_gpu_count()
    requested = args.gpu_counts
    if available_gpus and max(requested) > available_gpus:
        raise RuntimeError(f"Requested {max(requested)} GPUs, but only {available_gpus} are visible.")

    init_db(args.db_path)
    print(f"database: {args.db_path}")
    print(f"run_group: {args.run_group}")

    final_code = 0
    for gpu_count in requested:
        return_code = run_one(args, gpu_count)
        if return_code != 0:
            final_code = return_code
            if not args.continue_on_failure:
                break
    return final_code


if __name__ == "__main__":
    raise SystemExit(main())
