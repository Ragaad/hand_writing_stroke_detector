# Arabic Stroke Data Pipelines

This repo currently has three practical pipelines:

1. `synthesize_arabic_from_pdf.py` - extract Arabic text from selectable-text PDFs and generate synthetic letter-level pseudo-stroke JSON.
2. `render_json_overlay.py` - visualize an image and its JSON coordinates together.
3. `train_sandbox.py` - train a JSON-only LoRA model from text to stroke JSON.

The JSON target format matches the original dataset shape:

```json
[
  {"م": [[245.0, 77.0], [247.0, 73.0]]},
  {"ح": [[211.0, 88.0], [214.0, 91.0]]}
]
```

Each item is one stroke component. In synthetic data, a letter may produce multiple components when dots or diacritics are kept separately.

## Setup

From the repo root:

```bash
cd /Users/shahrukhhumayoun/Desktop/sedrah_ai_github/Qwen2-VL-2B
source venv/bin/activate
pip install -r requirements.txt
```

If you are training on Apple Silicon, force MPS:

```bash
python train_sandbox.py --device mps --dry-run --max-train-samples 1 --max-val-samples 1
```

## Pipeline 1: PDF To Synthetic Arabic Data

Script:

```bash
python synthesize_arabic_from_pdf.py
```

Example used for `chess2.pdf`:

```bash
python synthesize_arabic_from_pdf.py \
  --pdf chess2.pdf \
  --output-dir synthesis_arabic_data/chess2_test_filtered \
  --split train \
  --max-letters-per-sample 16 \
  --points-per-letter 96 \
  --columns 8 \
  --fix-rtl-order \
  --mask-threshold 64 \
  --min-component-pixels 8 \
  --smooth-window 3
```

Output:

```text
synthesis_arabic_data/chess2_test_filtered/
  images/train/*.png
  json/train/*.json
  manifest.jsonl
```

### Parameters

`--pdf`
: One or more Arabic PDF paths. The PDFs must contain selectable text. Scanned image-only PDFs need OCR first.

`--output-dir`
: Folder where generated images, JSON files, and `manifest.jsonl` are written.

`--split`
: Dataset split folder name. One of `train`, `valid`, or `test`.

`--font`
: Optional Arabic-capable font file. If omitted, the script tries macOS Arabic fonts such as `GeezaPro.ttc`.

`--canvas-size`
: Width and height of each generated square image. Default is `300`, giving a `300x300` coordinate canvas.

`--max-letters-per-sample`
: Number of Arabic letters per generated sample image. Larger values create denser images.

`--points-per-letter`
: Maximum number of sampled coordinate points per stroke component.

`--columns`
: Number of letter cells per row in the rendered image. Letters are placed right-to-left.

`--fix-rtl-order`
: Reverses each extracted Arabic line. Use this when generated filenames/text look backwards.

`--mask-threshold`
: Pixel intensity threshold for glyph masks. Higher values remove faint anti-aliased edge pixels.

`--min-component-pixels`
: Removes connected components smaller than this number of pixels. This removes specks while keeping real Arabic dots when they are large enough.

`--smooth-window`
: Moving-average smoothing window for sampled paths. Use `1` to disable smoothing. Odd values like `3` or `5` are best.

`--merge-components`
: Merges a letter body, dots, and diacritics into one stroke entry. Without it, separate components become separate JSON strokes with the same letter label.

### Notes

This pipeline creates synthetic glyph-boundary pseudo-strokes, not true handwriting pen trajectories. It is useful for controlled pretraining and debugging, then you should fine-tune on real stroke JSON.

## Pipeline 2: Render Image And JSON Coordinates

Script:

```bash
python render_json_overlay.py
```

### JSON-Only View For Synthetic Data

For synthetic data, render only the JSON coordinates on a blank canvas:

```bash
python render_json_overlay.py \
  --json synthesis_arabic_data/chess2_test_filtered/json/train/sample_000001_أحدثتضجةعالميةمم.json \
  --output synthesis_arabic_data/chess2_test_filtered/overlays/sample_000001_json_only.png \
  --json-only \
  --canvas-width 300 \
  --canvas-height 300 \
  --line-width 2 \
  --point-radius 1
```

### Split View

Use `--split` to create a two-part image:

```text
top: original or detected/rendered letters
bottom: JSON-derived coordinate paths
```

Example:

```bash
python render_json_overlay.py \
  --image synthesis_arabic_data/chess2_test_filtered/images/train/sample_000001_أحدثتضجةعالميةمم.png \
  --json synthesis_arabic_data/chess2_test_filtered/json/train/sample_000001_أحدثتضجةعالميةمم.json \
  --output synthesis_arabic_data/chess2_test_filtered/overlays/sample_000001_split.png \
  --split \
  --line-width 2 \
  --point-radius 1
```

### Overlay View

Without `--split`, paths are drawn directly on top of the image:

```bash
python render_json_overlay.py \
  --image path/to/image.png \
  --json path/to/strokes.json \
  --output outputs/overlays/sample_overlay.png \
  --labels
```

For original Sedrah samples, the image is usually two panels:

```text
left half:  the writer's handwritten attempt, where the JSON coordinates belong
right half: the reference/model image the writer was imitating
```

Use `--region left-half --fit-coords` so the JSON is rendered only on the handwritten side:

```bash
python render_json_overlay.py \
  --image 'sedrah_pipeline/sandbox_data/pix2pix/train/Shokot_شوقوط.jpg' \
  --json 'sedrah_pipeline/sandbox_data/json/train/شوقوط_0.json' \
  --output outputs/overlays/shokot_original_overlay.png \
  --region left-half \
  --fit-coords \
  --labels
```

For a top/bottom comparison of only the handwritten side:

```bash
python render_json_overlay.py \
  --image 'sedrah_pipeline/sandbox_data/pix2pix/train/Shokot_شوقوط.jpg' \
  --json 'sedrah_pipeline/sandbox_data/json/train/شوقوط_0.json' \
  --output outputs/overlays/shokot_left_half_split.png \
  --region left-half \
  --fit-coords \
  --split
```

### Parameters

`--image`
: Source image path.

`--json`
: Stroke JSON path. Supports the original `[{letter: points}]` format and object-style `{"strokes": [...]}`.

`--output`
: Output image path.

`--json-only`
: Render only the JSON coordinates on a blank canvas. This is recommended for synthetic data.

`--canvas-width`, `--canvas-height`
: Blank canvas size for `--json-only`. Use `300x300` for default synthetic samples.

`--coord-width`, `--coord-height`
: Original JSON coordinate-space size. Use these when coordinates are known to come from a fixed canvas different from the image size.

`--fit-coords`
: Fits the min/max JSON point bounds into the image canvas. Useful for original dataset samples when exact coordinate canvas size is unknown.

`--region`
: Region where JSON coordinates should be rendered. Use `full` for normal images, `left-half` for original Sedrah paired images, and `right-half` only if a dataset stores coordinates for the reference side.

`--show-region-boundary`
: Draws a boundary around the selected render region in overlay mode.

`--padding`
: Padding used with `--fit-coords`.

`--line-width`
: Width of rendered stroke lines.

`--point-radius`
: Radius of point markers drawn along each stroke.

`--alpha`
: Transparency of rendered paths. `255` is fully opaque.

`--labels`
: Draws the letter label near each stroke start.

`--label-size`
: Font size for labels.

`--max-points-drawn`
: Maximum number of point markers per stroke. Lines still use all transformed points.

`--split`
: Creates the two-panel image: original/detected letters on top, JSON paths on bottom.

## Pipeline 3: JSON-Only LoRA Training

Script:

```bash
python train_sandbox.py
```

This trains a model to map:

```text
input: Arabic text from filename
output: stroke JSON
```

It does not train image-based stroke detection yet.

### Dry Run

This checks local JSON data without loading the model:

```bash
python train_sandbox.py \
  --dry-run \
  --max-train-samples 1 \
  --max-val-samples 1
```

### Apple M4 Smoke Test

```bash
python train_sandbox.py \
  --device mps \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --max-length 4096 \
  --output-dir outputs/local-m4-smoke-lora
```

### Slightly Larger Local Test

```bash
python train_sandbox.py \
  --device mps \
  --max-train-samples 8 \
  --max-val-samples 2 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --max-length 4096 \
  --output-dir outputs/local-m4-smoke-lora
```

### Parameters

`--data-dir`
: Root JSON data folder. Default is `sedrah_pipeline/sandbox_data/json`.

`--model-id`
: Hugging Face model id. Default is `Qwen/Qwen2-VL-2B-Instruct`.

`--output-dir`
: Folder for saved LoRA adapter and tokenizer.

`--device`
: `auto`, `mps`, `cuda`, or `cpu`. Use `mps` for Apple M4 local tests.

`--epochs`
: Number of training epochs.

`--batch-size`
: Per-step batch size. Keep `1` locally.

`--grad-accum-steps`
: Number of steps to accumulate gradients before optimizer update. Effective batch size is `batch-size * grad-accum-steps`.

`--lr`
: Learning rate for LoRA parameters.

`--max-length`
: Maximum token sequence length. Lower values train faster but truncate long stroke JSON.

`--decimals`
: Number of decimal places kept for coordinate values in target JSON.

`--lora-rank`
: LoRA rank. Higher means more trainable capacity and more memory.

`--lora-alpha`
: LoRA scaling value.

`--lora-dropout`
: Dropout applied inside LoRA layers.

`--max-train-samples`
: Limits the number of training JSON files. Good for smoke tests.

`--max-val-samples`
: Limits validation JSON files.

`--log-every`
: Print training loss every N steps.

`--dry-run`
: Load and print a sample from the dataset without loading the model or training.

## Recommended Workflow

1. Generate synthetic data from a PDF:

```bash
python synthesize_arabic_from_pdf.py \
  --pdf chess2.pdf \
  --output-dir synthesis_arabic_data/chess2_test_filtered \
  --fix-rtl-order \
  --mask-threshold 64 \
  --min-component-pixels 8 \
  --smooth-window 3
```

2. Visualize one generated synthetic JSON sample:

```bash
python render_json_overlay.py \
  --json synthesis_arabic_data/chess2_test_filtered/json/train/sample_000001_أحدثتضجةعالميةمم.json \
  --output synthesis_arabic_data/chess2_test_filtered/overlays/sample_000001_json_only.png \
  --json-only
```

3. Run a local model smoke test:

```bash
python train_sandbox.py \
  --device mps \
  --max-train-samples 2 \
  --max-val-samples 2 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --max-length 4096
```

4. Use H100/Brev for real experiments after local smoke tests pass.
