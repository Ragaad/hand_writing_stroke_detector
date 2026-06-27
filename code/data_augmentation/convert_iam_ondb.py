import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

SPLIT_FILES = {
    "train": "trainset.txt",
    "validation": "testset_v.txt",
    "test": "testset_t.txt",
}

SAMPLE_ID_RE = re.compile(r"^(((\w+)-\d+)\w?)$")


def sample_base_path(sample_id):
    match = SAMPLE_ID_RE.match(sample_id)
    if not match:
        raise ValueError(f"Unrecognized IAM-OnDB sample id format: {sample_id!r}")
    full, form, writer = match.group(1), match.group(2), match.group(3)
    return f"{writer}/{form}/{full}"


def read_ascii_lines(ascii_path):
    text = ascii_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r".*[\r\n]+CSR:\s*[\r\n]+", "", text, count=1, flags=re.DOTALL)
    lines = [line.strip() for line in re.split(r"[\r\n]+", text.strip())]
    return [line for line in lines if line]


def read_strokes(xml_path):
    root = ET.parse(xml_path).getroot()
    strokes = []
    for stroke_el in root.iter("Stroke"):
        points = [
            [float(point_el.get("x")), float(point_el.get("y"))]
            for point_el in stroke_el.iter("Point")
        ]
        if points:
            strokes.append(points)
    return strokes


def safe_filename(text, max_len=80):
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "_", text).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] or "sample"


def convert_split(raw_dir, split_file, out_dir):
    sample_ids = [
        line.strip()
        for line in (raw_dir / split_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    name_counts = {}
    written, missing = 0, 0

    for sample_id in sample_ids:
        base = sample_base_path(sample_id)
        ascii_path = raw_dir / "ascii" / f"{base}.txt"
        if not ascii_path.exists():
            print(f"WARNING: missing ascii file for {sample_id}: {ascii_path}")
            missing += 1
            continue

        for line_index, line_text in enumerate(read_ascii_lines(ascii_path), start=1):
            xml_path = raw_dir / "lineStrokes" / f"{base}-{line_index:02d}.xml"
            if not xml_path.exists():
                print(f"WARNING: missing strokes for {sample_id} line {line_index}: {xml_path}")
                missing += 1
                continue

            strokes = read_strokes(xml_path)
            if not strokes:
                continue

            # No per-character alignment is available in IAM-OnDB, so every stroke in
            # this line shares the same label: the full transcribed line.
            entries = [{line_text: points} for points in strokes]

            safe_text = safe_filename(line_text)
            sample_index = name_counts.get(safe_text, 0)
            name_counts[safe_text] = sample_index + 1

            out_path = out_dir / f"{safe_text}_{sample_index}.json"
            out_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
            written += 1

    print(f"[{split_file}] wrote {written} files, {missing} missing references")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw IAM-OnDB ascii/lineStrokes files into Sedrah-style stroke JSON."
    )
    parser.add_argument("--raw-dir", default="sedrah_pipeline/iam_ondb_dataset/raw")
    parser.add_argument("--out-dir", default="sedrah_pipeline/iam_ondb_dataset/json")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    for split, split_file in SPLIT_FILES.items():
        if not (raw_dir / split_file).exists():
            print(f"SKIP: {split_file} not found in {raw_dir}")
            continue
        convert_split(raw_dir, split_file, out_dir / split)


if __name__ == "__main__":
    main()
