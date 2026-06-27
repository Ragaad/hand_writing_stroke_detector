import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


INKML_NS = "{http://www.w3.org/2003/InkML}"
DEFAULT_TYPES = ("Text", "WordGroup", "WordGroup2", "WordGroup3", "NumberGroup")


@dataclass
class PohSample:
    source_path: Path
    writepad_type: str
    writer_id: str
    truth_id: str
    text: str
    strokes: list


def local_name(tag):
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalized_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def trim_utf8(value, max_bytes):
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value

    trimmed = encoded[:max_bytes]
    while trimmed:
        try:
            return trimmed.decode("utf-8").rstrip()
        except UnicodeDecodeError:
            trimmed = trimmed[:-1]
    return "sample"


def safe_filename_text(text, max_bytes=180):
    cleaned = text.replace("/", " ").replace("\\", " ")
    cleaned = "".join(" " if ord(ch) < 32 else ch for ch in cleaned)
    cleaned = normalized_text(cleaned)
    cleaned = cleaned.rstrip(". ")
    return trim_utf8(cleaned, max_bytes) or "sample"


def format_number(value, decimals):
    rounded = round(float(value), decimals)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def extract_content(text_el):
    content = None
    for child in text_el:
        if local_name(child.tag) == "content":
            content = child
            break
    if content is None:
        return normalized_text(" ".join(text_el.itertext()))

    lines = []
    for child in content:
        if local_name(child.tag) == "line":
            line = normalized_text(" ".join(child.itertext()))
            if line:
                lines.append(line)

    if lines:
        return normalized_text(" ".join(lines))
    return normalized_text(" ".join(content.itertext()))


def load_ground_truths(raw_dir):
    by_type = {}
    by_id = defaultdict(set)

    for xml_path in sorted((raw_dir / "GroundTruths").glob("*/*.xml")):
        truth_type = xml_path.parent.name
        root = ET.parse(xml_path).getroot()
        for text_el in root.iter():
            if local_name(text_el.tag) != "text":
                continue
            truth_id = text_el.get("id")
            text = extract_content(text_el)
            if not truth_id or not text:
                continue
            by_type[(truth_type, truth_id)] = text
            by_id[truth_id].add(text)

    unique_by_id = {
        truth_id: next(iter(values))
        for truth_id, values in by_id.items()
        if len(values) == 1
    }
    return by_type, unique_by_id


def parse_annotation(root):
    annotation = {}
    for el in root.iter():
        if local_name(el.tag) != "annotationXML":
            continue
        for child in el:
            annotation[local_name(child.tag)] = normalized_text(child.text)
        break
    return annotation


def trace_channel_indices(root):
    for trace_format in root.iter(f"{INKML_NS}traceFormat"):
        channels = [
            channel.get("name")
            for channel in trace_format
            if local_name(channel.tag) == "channel"
        ]
        if "X" in channels and "Y" in channels:
            return channels.index("X"), channels.index("Y")
    return 0, 1


def parse_trace_points(trace_el, x_index, y_index, decimals):
    points = []
    raw_trace = trace_el.text or ""
    required_len = max(x_index, y_index) + 1

    for raw_point in raw_trace.replace("\n", " ").split(","):
        values = raw_point.strip().split()
        if len(values) < required_len:
            continue
        try:
            x = format_number(values[x_index], decimals)
            y = format_number(values[y_index], decimals)
        except ValueError:
            continue
        points.append([x, y])

    return points


def load_writepad_sample(path, ground_truths_by_type, unique_ground_truths_by_id, decimals):
    root = ET.parse(path).getroot()
    annotation = parse_annotation(root)
    writepad_type = annotation.get("type") or path.parent.name
    writer_id = annotation.get("writerId") or f"missing-writer:{path.parent.name}:{path.stem}"
    truth_id = annotation.get("truthId")
    if not truth_id:
        return None, "missing_truth_id"

    text = ground_truths_by_type.get((writepad_type, truth_id)) or unique_ground_truths_by_id.get(truth_id)
    if not text:
        return None, "missing_ground_truth"

    x_index, y_index = trace_channel_indices(root)
    strokes = []
    for trace_el in root.iter(f"{INKML_NS}trace"):
        points = parse_trace_points(trace_el, x_index, y_index, decimals)
        if points:
            strokes.append({text: points})

    if not strokes:
        return None, "empty_traces"

    return (
        PohSample(
            source_path=path,
            writepad_type=writepad_type,
            writer_id=writer_id,
            truth_id=truth_id,
            text=text,
            strokes=strokes,
        ),
        None,
    )


def collect_samples(raw_dir, writepad_types, decimals, max_samples=None):
    ground_truths_by_type, unique_ground_truths_by_id = load_ground_truths(raw_dir)
    samples = []
    skipped = Counter()

    for writepad_type in writepad_types:
        writepad_dir = raw_dir / "Writepads" / writepad_type
        if not writepad_dir.exists():
            skipped[f"missing_writepad_dir:{writepad_type}"] += 1
            continue
        for path in sorted(writepad_dir.glob("*.inkml")):
            sample, skip_reason = load_writepad_sample(
                path,
                ground_truths_by_type,
                unique_ground_truths_by_id,
                decimals,
            )
            if skip_reason:
                skipped[skip_reason] += 1
                continue
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                return samples, skipped, len(ground_truths_by_type)

    return samples, skipped, len(ground_truths_by_type)


def split_by_writer(samples, train_ratio, validation_ratio, seed):
    by_writer = defaultdict(list)
    for sample in samples:
        by_writer[sample.writer_id].append(sample)

    writers = sorted(by_writer)
    rng = random.Random(seed)
    rng.shuffle(writers)

    train_writer_count = int(round(len(writers) * train_ratio))
    validation_writer_count = int(round(len(writers) * validation_ratio))
    train_writer_count = min(max(train_writer_count, 1), len(writers))
    validation_writer_count = min(max(validation_writer_count, 1), max(len(writers) - train_writer_count, 0))

    train_writers = set(writers[:train_writer_count])
    validation_writers = set(writers[train_writer_count:train_writer_count + validation_writer_count])

    splits = {"train": [], "validation": [], "test": []}
    for writer_id in writers:
        if writer_id in train_writers:
            split = "train"
        elif writer_id in validation_writers:
            split = "validation"
        else:
            split = "test"
        splits[split].extend(by_writer[writer_id])

    for split_samples in splits.values():
        split_samples.sort(key=lambda item: (item.writepad_type, int(item.source_path.stem), item.source_path.name))

    return splits


def write_json_dataset(splits, out_dir, filename_max_bytes):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    filename_counts = defaultdict(Counter)
    truncated_filenames = 0
    written_counts = {}

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for split, samples in splits.items():
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for sample in samples:
                filename_text = safe_filename_text(sample.text, filename_max_bytes)
                if filename_text != sample.text:
                    truncated_filenames += 1

                sample_index = filename_counts[split][filename_text]
                filename_counts[split][filename_text] += 1
                out_path = split_dir / f"{filename_text}_{sample_index}.json"

                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(sample.strokes, f, ensure_ascii=False, separators=(",", ":"))

                manifest.write(
                    json.dumps(
                        {
                            "split": split,
                            "json_path": str(out_path),
                            "source_path": str(sample.source_path),
                            "writepad_type": sample.writepad_type,
                            "writer_id": sample.writer_id,
                            "truth_id": sample.truth_id,
                            "text": sample.text,
                            "filename_text": filename_text,
                            "stroke_count": len(sample.strokes),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            written_counts[split] = len(samples)

    return written_counts, truncated_filenames, manifest_path


def parse_writepad_types(value):
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or list(DEFAULT_TYPES)


def main():
    parser = argparse.ArgumentParser(description="Convert POH-Db InkML files into Sedrah-style stroke JSON.")
    parser.add_argument("--raw-dir", default="sedrah_pipeline/poh_db_dataset/raw")
    parser.add_argument("--out-dir", default="sedrah_pipeline/poh_db_dataset/json")
    parser.add_argument("--writepad-types", default=",".join(DEFAULT_TYPES))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--filename-max-bytes", type=int, default=180)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"POH-Db raw directory not found: {raw_dir}")
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")
    if args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("--train-ratio + --validation-ratio must leave room for a test split")

    samples, skipped, ground_truth_count = collect_samples(
        raw_dir,
        parse_writepad_types(args.writepad_types),
        decimals=args.decimals,
        max_samples=args.max_samples,
    )
    if not samples:
        raise RuntimeError("No convertible POH-Db samples were found.")

    splits = split_by_writer(samples, args.train_ratio, args.validation_ratio, args.seed)
    written_counts, truncated_filenames, manifest_path = write_json_dataset(
        splits,
        out_dir,
        filename_max_bytes=args.filename_max_bytes,
    )

    writer_counts = {
        split: len({sample.writer_id for sample in split_samples})
        for split, split_samples in splits.items()
    }
    print(
        json.dumps(
            {
                "raw_dir": str(raw_dir),
                "out_dir": str(out_dir),
                "ground_truth_entries": ground_truth_count,
                "converted_samples": len(samples),
                "written_samples": written_counts,
                "writers_by_split": writer_counts,
                "skipped": dict(skipped),
                "truncated_filename_texts": truncated_filenames,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
