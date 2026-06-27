# Sedrah Pipeline — Commands Log

## 1. Dataset extraction and placement

Source archives: `Calliar/calliar_dataset/dataset.zip` and `Calliar/calliar_dataset/dataset_imgs.zip`.

```bash
# Extract dataset.zip (contains dataset/{train,test,valid} json + pix2pix/{train,test,val} jpg)
unzip -q -o Calliar/calliar_dataset/dataset.zip -d <scratch>/extract_dataset

# Move stroke JSON files into the json split dirs
mv <scratch>/extract_dataset/dataset/train/*   sedrah_pipeline/calliar_dataset/json/train/
mv <scratch>/extract_dataset/dataset/test/*    sedrah_pipeline/calliar_dataset/json/test/
mv <scratch>/extract_dataset/dataset/valid/*   sedrah_pipeline/calliar_dataset/json/validation/

# Move pix2pix name-signature renders into the pix2pix split dirs
mv <scratch>/extract_dataset/pix2pix/train/*   sedrah_pipeline/calliar_dataset/pix2pix/train/
mv <scratch>/extract_dataset/pix2pix/test/*    sedrah_pipeline/calliar_dataset/pix2pix/test/
mv <scratch>/extract_dataset/pix2pix/val/*     sedrah_pipeline/calliar_dataset/pix2pix/validation/

# Extract dataset_imgs.zip (rendered sentence images + text labels) into a new imgs/ folder
unzip -q -o Calliar/calliar_dataset/dataset_imgs.zip -d <scratch>/extract_imgs
mkdir -p sedrah_pipeline/calliar_dataset/imgs/{train,test,validation}
mv <scratch>/extract_imgs/dataset_imgs/train/* sedrah_pipeline/calliar_dataset/imgs/train/
mv <scratch>/extract_imgs/dataset_imgs/test/*  sedrah_pipeline/calliar_dataset/imgs/test/
mv <scratch>/extract_imgs/dataset_imgs/valid/* sedrah_pipeline/calliar_dataset/imgs/validation/
```

Resulting counts: `json` 2000/250/250 (train/test/validation), `pix2pix` 400/132/100, `imgs` 2000/250/250 (png+txt pairs).

## 2. `code/training/train_sandbox.py` changes to match the new layout

- `SPLIT_DIRS["validation"/"valid"/"val"]` now resolve to `"validation"` (the folder created above), not the old `"valid"`.
- Default `--data-dir` changed from the stale `sedrah_pipeline/sandbox_data/json` to `sedrah_pipeline/calliar_dataset/json`.
- Added wall-clock benchmarking: per-step `samples_per_sec`, per-epoch duration, and a `benchmark.json` written into the output dir alongside the LoRA adapter.

## 3. Smoke test (dry run, no model weights loaded for inference)

```bash
python3 code/training/train_sandbox.py --dry-run
```

Confirms dataset discovery (2000 train / 250 validation json files) and stroke JSON parsing/serialization.

## 4. Training run — `qwen2vl-sedrah-stroke-lora_v1`

```bash
CUDA_VISIBLE_DEVICES=0 python3 code/training/train_sandbox.py \
  --epochs 10 \
  --output-dir outputs/qwen2vl-sedrah-stroke-lora_v1 \
  > train_run_v1.log 2>&1
```

- Base model: `Qwen/Qwen2-VL-2B-Instruct` (LoRA, rank 16, alpha 32, dropout 0.05, targeting `q_proj/k_proj/v_proj/o_proj`).
- Data: `sedrah_pipeline/calliar_dataset/json/train` (2000 samples), validated against `json/validation` (250 samples).
- Device: single GPU (`cuda:0`, bf16), batch size 1, grad accumulation 8.
- Run in background; progress logged to `train_run_v1.log`; final adapter + tokenizer + `benchmark.json` saved to `outputs/qwen2vl-sedrah-stroke-lora_v1/`.

## 5. Evaluation — `code/training/evaluate_sandbox.py`

Generates the stroke JSON for each sample in a split and scores the predicted stroke **label** sequence (the character/primitive-stroke tags, not the point trajectories) against ground truth:

- `stroke_label_precision` / `stroke_label_recall` / `stroke_label_f1` — micro-averaged over a multiset (`Counter`) intersection of predicted vs. target stroke labels across the whole split.
- `exact_match_accuracy` — fraction of samples where the predicted label sequence exactly equals the target sequence, in order.
- `parse_failures` — generations where no valid JSON object (or no `strokes` key) could be extracted, e.g. due to truncation or schema mismatch.

```bash
python3 code/training/evaluate_sandbox.py \
  --adapter-dir outputs/qwen2vl-sedrah-stroke-lora_v1 \
  --split test \
  --max-new-tokens 3072
```

Metrics are saved to `outputs/qwen2vl-sedrah-stroke-lora_v1/eval_metrics_test.json`.

Smoke-tested against the pre-existing `outputs/h100-smoke-lora` adapter on an idle GPU (`CUDA_VISIBLE_DEVICES=1`) while the v1 run trained on GPU 0: load/generate/parse/score/save-metrics all completed without error. That adapter reported 0% scores because it was trained on an older `start`/`end` span JSON schema, not the current `text`/`strokes` schema — the parser correctly flagged this as a schema mismatch rather than crashing.


python -u code/training/train_sandbox.py \
  --epochs 10 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 2e-5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --output-dir outputs/qwen2vl-sedrah-stroke-lora_v2 \
  > train_run_v2.log 2>&1 & echo $! > train_run_v2.pid


  #distributed 
  accelerate launch --num_processes=8 code/training/train_sandbox.py \
  --epochs 5 \
  --batch-size 4 \
  --grad-accum-steps 2 \
  --lr 2e-4 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --output-dir outputs/qwen2vl-sedrah-stroke-lora_v3 \
  > train_run_v3.log 2>&1 & echo $! > train_run_v3.pid


  cd /home/nvidia/Qwen2-VL-2B
rm -rf outputs/inference_renders
.venv/bin/python code/inference/infer_stroke.py --text "ب" --adapter-dir outputs/qwen2vl-pohdb-stroke-lora_v5-farsi  2>&1 | tail -10
echo "---"
.venv/bin/python code/inference/infer_stroke.py --text "قلم" --adapter-dir outputs/qwen2vl-pohdb-stroke-lora_v5-farsi  2>&1 | tail -10
echo "---"
.venv/bin/python code/inference/infer_stroke.py --text "بسم الله" --adapter-dir outputs/qwen2vl-pohdb-stroke-lora_v5-farsi  2>&1 | tail -10
echo "---"
find inference-results -type f


#Farsi adaptar
cd /home/nvidia/Qwen2-VL-2B
.venv/bin/python code/inference/infer_stroke.py --text "پ" --adapter-dir outputs/qwen2vl-pohdb-stroke-lora_v5-farsi 2>&1 | tail -8


cd /home/nvidia/Qwen2-VL-2B
.venv/bin/python code/training/train_sandbox.py \
  --data-dir sedrah_pipeline/calliar_combined_dataset/json \
  --image-manifest sedrah_pipeline/calliar_dataset/image_manifest.jsonl \
  --max-train-samples 20 --max-val-samples 4 \
  --epochs 1 --batch-size 1 --grad-accum-steps 2 --log-every 2 \
  --output-dir outputs/multimodal-smoke-test 2>&1 | tail -40

  cd /home/nvidia/Qwen2-VL-2B
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --num_processes=8 code/training/train_sandbox.py \
  --data-dir sedrah_pipeline/calliar_combined_dataset/json \
  --image-manifest sedrah_pipeline/calliar_dataset/image_manifest.jsonl \
  --epochs 15 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 2e-4 \
  --lora-rank 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --output-dir outputs/qwen2vl-calliar-aug-stroke-lora_v6 \
  > train_run_v6.log 2>&1

  v6 (multimodal, 15 epochs, combined original+augmented Arabic data, 12000 train samples) is running on all 8 GPUs without errors. Given the dataset size and image-processing overhead, this will likely take a few hours — I'll notify you the moment it finishes or crashes, no need to check in.

  Used in training (v6, currently running): 2,482 images — these are the ones in image_manifest.jsonl, each paired with a stroke-JSON sample.

Breaking down all image sources in the repo:

Source	train	validation	test	total
imgs/ (raw, stroke renderings)	2000	250	250	2500
pix2pix/ (raw, two-panel)	400	100	132	632
pix2pix_left/ (cropped, left panel only)	—	—	—	335
image_manifest.jsonl (actually used)	—	—	—	2482
The manifest count (2482) is slightly below the raw imgs/ count (2500) because ~18 samples couldn't be text-matched to a JSON file. The 335 pix2pix_left crops were generated but contributed zero additional entries to the manifest — all of them turned out to already be covered by an imgs/ match. The remaining ~12,000 - 2,482 ≈ 9,500 augmented samples in the combined training set train text-only (no accurate image exists for their transformed coordinates).


cd /home/nvidia/Qwen2-VL-2B
.venv/bin/python code/data_augmentation/render_augmented_images.py \
  --input-dir sedrah_pipeline/calliar_dataset_aug/json \
  --output-dir sedrah_pipeline/calliar_dataset_aug/imgs_rendered \
  --manifest-out sedrah_pipeline/calliar_dataset_aug/image_manifest.jsonl \
  --workers 8 2>&1 | tail -20