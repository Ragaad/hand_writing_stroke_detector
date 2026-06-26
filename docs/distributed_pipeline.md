# Distributed Training Pipeline

This document describes the single-node distributed training pipeline for the Sedrah stroke JSON fine-tuning workflow. The implementation is model-agnostic enough to support Qwen2-VL style models and standard causal language models through a small model-family switch.

## Components

- `train_distributed.py`: DDP training entrypoint with LoRA, checkpoint/resume, rank-0 logging, distributed validation, and final benchmark export.
- `benchmark_distributed_scaling.py`: gradual scaling launcher for 1, 2, 4, and 8 GPU runs. It stores benchmark results in SQLite.
- `run_scaling_benchmark_background.sh`: background launcher for the scaling benchmark.

## Effective Batch Size

The pipeline keeps training behavior comparable while scaling GPU count by preserving the effective global batch size:

```text
effective_global_batch = batch_size_per_gpu * grad_accum_steps * gpu_count
```

For the previous single-GPU run, the effective batch was:

```text
1 * 8 * 1 = 8
```

The gradual benchmark keeps that same effective batch:

```text
1 GPU -> batch_size=1, grad_accum_steps=8
2 GPU -> batch_size=1, grad_accum_steps=4
4 GPU -> batch_size=1, grad_accum_steps=2
8 GPU -> batch_size=1, grad_accum_steps=1
```

## Smoke Test

Run a tiny one-GPU test before a full benchmark:

```bash
.venv/bin/torchrun --nnodes=1 --nproc-per-node=1 \
  --master-addr=127.0.0.1 --master-port=29500 \
  train_distributed.py \
  --epochs 1 \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --max-train-steps 2 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --output-dir outputs/ddp_smoke_1gpu
```

## Run Gradual Scaling

Foreground:

```bash
.venv/bin/python benchmark_distributed_scaling.py \
  --gpu-counts 1,2,4,8 \
  --target-effective-batch 8 \
  --epochs 1 \
  --run-group sedrah-ddp-scale-v1
```

Background:

```bash
./run_scaling_benchmark_background.sh sedrah-ddp-scale-v1
```

The benchmark runner writes:

```text
benchmarks/training_benchmarks.sqlite
outputs/distributed_scaling/<run_group>/
logs/distributed_scaling/<run_group>/
```

## Query Results

```bash
sqlite3 benchmarks/training_benchmarks.sqlite \
  "select run_group, gpu_count, status, effective_global_batch_size, samples_per_sec_avg, avg_epoch_seconds, final_val_loss from runs order by id;"
```

Epoch-level metrics are in the `epochs` table:

```bash
sqlite3 benchmarks/training_benchmarks.sqlite \
  "select run_id, epoch, train_loss, val_loss, epoch_seconds, global_step from epochs order by run_id, epoch;"
```

## Resume

Each distributed run saves epoch checkpoints under the configured checkpoint directory. Resume an 8-GPU run like this:

```bash
.venv/bin/torchrun --nnodes=1 --nproc-per-node=8 \
  --master-addr=127.0.0.1 --master-port=29508 \
  train_distributed.py \
  --resume-from-checkpoint outputs/distributed_scaling/sedrah-ddp-scale-v1/gpus_8/checkpoints/epoch-0001 \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --output-dir outputs/distributed_scaling/sedrah-ddp-scale-v1/gpus_8_resume
```

For mid-epoch step checkpoints, resume with the same GPU count, per-GPU batch size, and gradient accumulation as the original run.

## Notes

- The scaling launcher uses `--master-addr=127.0.0.1` and a free local port for each stage to avoid hostname rendezvous issues on this machine.
- Start with 1 GPU, then 2, 4, and 8. Do not jump straight to a larger cluster until the validation loss and throughput are sane.
- If another training job is already using GPU 0, wait for it to finish before running the default scaling launcher.
