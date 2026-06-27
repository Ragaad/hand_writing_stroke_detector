# Inference & Evaluation Pipeline

This document describes how to run a trained Sedrah LoRA adapter on new text (and
optionally an image) to generate stroke trajectories, render the result, and score
predicted strokes against ground truth.

## Components

- `code/inference/infer_stroke.py`: loads the base model + a LoRA adapter, generates
  stroke-trajectory JSON for a given text, parses/validates the model output, and
  renders it to a PNG via the data-augmentation rendering helpers.
- `code/training/trajectory_eval.py`: compares a predicted stroke trajectory against
  a ground-truth one (DTW distance, precision, recall) and logs results to a CSV for
  tracking quality over time as augmentation/training parameters change.

## Running Inference

One-shot:

```bash
python code/inference/infer_stroke.py \
  --text "بسم الله" \
  --adapter-dir outputs/qwen2vl-calliar-aug-stroke-lora_v6
```

Interactive (keeps the model loaded, type repeated inputs):

```bash
python code/inference/infer_stroke.py --adapter-dir outputs/qwen2vl-calliar-aug-stroke-lora_v6
```

Each call writes three files under `inference-results/{letters,words,sentences}/`
(category chosen automatically from the input — single character, single word, or
multi-word):

```text
<NNN>_<text>_<timestamp>.png        rendered stroke image
<NNN>_<text>_<timestamp>.json       parsed stroke JSON (only if parsing succeeded)
<NNN>_<text>_<timestamp>_raw.txt    raw model output (always written; useful when parsing fails)
```

### Parameters

`--model-id` (default `Qwen/Qwen2-VL-2B-Instruct`), `--adapter-dir` (default
`outputs/qwen2vl-sedrah-stroke-lora_v4`), `--device` (`auto`/`cuda`/`cpu`), `--text`
(single-shot input; omit for interactive mode).

`--max-new-tokens` (default `2048`): raise this for longer phrases — target JSON for
out-of-distribution or long inputs can run into the thousands of tokens.

`--do-sample`, `--temperature` (`0.7`), `--top-p` (`0.9`): enable/configure sampling
instead of greedy decoding.

`--repetition-penalty` (default `1.0`, i.e. disabled), `--no-repeat-ngram-size`
(default `0`, disabled): **leave these at their defaults unless generation is stuck
in a genuine repetition loop.** Stroke JSON is inherently repetitive (brackets,
commas, similar coordinates), so a repetition penalty above `1.0` reliably corrupts
otherwise-valid output (e.g. spurious spaces inserted mid-number). If you do hit a
real loop (most likely for out-of-distribution prompts, e.g. a single letter for a
model trained mostly on full lines), prefer `--do-sample` first before reaching for
these.

`--canvas-size` (default `600`, square), `--no-fit-coords` (render raw model
coordinates instead of auto-scaling to fit the canvas), `--no-labels` (disable
per-stroke character labels in the render).

## Evaluating Predictions

`trajectory_eval.py` accepts a predicted and a ground-truth stroke JSON file —
either the bare `[{label: points}, ...]` Sedrah/Calliar format, our
`{"strokes": [...]}` wrapper, or a plain list of strokes where each stroke is a list
of `[x, y]` / `[x, y, t]` points.

```bash
python code/training/trajectory_eval.py \
  --pred inference-results/words/000_ما_20260627_201748.json \
  --gt sedrah_pipeline/calliar_dataset/json/train/ما_0.json \
  --tag v6-smoke-test
```

This prints:

```json
{
  "dtw_distance": 2.9696249664833467,
  "precision": 1.0,
  "recall": 0.7865168539325843
}
```

and appends a row (timestamp, metrics, `--pred`/`--gt` paths, `--tag`) to
`evaluation_log.csv` (header written once, on first run) — run this after each
training/augmentation change and compare rows over time.

### How the metrics are computed

1. **Normalize**: both trajectories' points are scaled to `[0, 1]`, using
   `--canvas-width`/`--canvas-height` if given, otherwise each trajectory's own
   bounding box.
2. **Flatten**: strokes are concatenated into one `[N, 2]` point sequence (the third
   `t` coordinate, if present, is dropped — sequence order already encodes time for
   DTW purposes).
3. **DTW distance**: `fastdtw` with Euclidean point distance — aligns the two
   sequences temporally even if they differ in length or local speed.
4. **Precision / recall** (spatial threshold `--threshold`, default `0.05`): for each
   predicted point, find its nearest ground-truth point (KD-tree) — precision is the
   fraction within the threshold. Recall is the same lookup in the other direction
   (each ground-truth point's nearest predicted point).

## Notes

- Inference can safely run alongside an active multi-GPU training job — it only
  needs a small amount of free memory on whichever single GPU it defaults to, and
  loads its own short-lived copy of the base model.
- `infer_stroke.py` and `trajectory_eval.py` both fall back gracefully when fed
  text-only vs. image-bearing samples, or our different stroke JSON shapes — no
  separate code path is needed depending on which adapter/dataset you're testing.
