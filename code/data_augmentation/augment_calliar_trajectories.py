import argparse
import copy
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keeps the script usable in minimal envs.
    def tqdm(iterable, **_kwargs):
        return iterable


TRANSFORM_NAMES = ("scaling", "rotation", "shear", "noise", "time_shift")


@dataclass(frozen=True)
class AugmentConfig:
    scale_min: float = 0.85
    scale_max: float = 1.15
    rotation_min_deg: float = -5.0
    rotation_max_deg: float = 5.0
    shear_min: float = -0.2
    shear_max: float = 0.2
    noise_mean: float = 0.0
    noise_std: float = 0.5
    time_scale_min: float = 0.8
    time_scale_max: float = 1.2
    decimals: int = 3
    time_decimals: int = 3
    min_transforms: int = 2
    max_transforms: int = 3
    preserve_ints: bool = True


def stable_seed(base_seed, path, generation_index):
    raw = f"{base_seed}:{path.as_posix()}:{generation_index}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32)


def parse_point_sequence(sequence, source_path):
    if not isinstance(sequence, list):
        raise ValueError(f"{source_path}: stroke sequence must be a list")
    if not sequence:
        return None

    dims = None
    rows = []
    for point in sequence:
        if not isinstance(point, list) or len(point) < 2:
            raise ValueError(f"{source_path}: each point must be a list with at least [x, y]")
        if dims is None:
            dims = len(point)
        elif len(point) != dims:
            raise ValueError(f"{source_path}: all points in a stroke must have the same dimensionality")
        try:
            rows.append([float(value) for value in point])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source_path}: point contains a non-numeric value: {point!r}") from exc

    return np.asarray(rows, dtype=np.float64)


def collect_sequences(data, source_path):
    """Return mutable point-sequence refs plus an object/list copy to write back."""
    augmented = copy.deepcopy(data)
    sequence_refs = []

    if isinstance(augmented, dict):
        strokes = augmented.get("strokes")
        if not isinstance(strokes, list):
            raise ValueError(f"{source_path}: object JSON must contain a list-valued 'strokes' key")

        for stroke in strokes:
            if isinstance(stroke, list):
                sequence_refs.append(stroke)
            elif isinstance(stroke, dict) and isinstance(stroke.get("points"), list):
                sequence_refs.append(stroke["points"])
            elif isinstance(stroke, dict) and len(stroke) == 1:
                sequence_refs.append(next(iter(stroke.values())))
            else:
                raise ValueError(f"{source_path}: unsupported stroke entry inside 'strokes'")
        return augmented, sequence_refs, "object"

    if isinstance(augmented, list):
        for item in augmented:
            if isinstance(item, list):
                sequence_refs.append(item)
            elif isinstance(item, dict) and isinstance(item.get("points"), list):
                sequence_refs.append(item["points"])
            elif isinstance(item, dict) and len(item) == 1:
                sequence_refs.append(next(iter(item.values())))
            else:
                raise ValueError(f"{source_path}: unsupported list-style stroke entry")
        return augmented, sequence_refs, "list"

    raise ValueError(f"{source_path}: expected top-level object or list JSON")


def geometric_center(arrays):
    xy_arrays = [array[:, :2] for array in arrays if array is not None and len(array)]
    if not xy_arrays:
        raise ValueError("sample has no non-empty point sequences")
    return np.concatenate(xy_arrays, axis=0).mean(axis=0)


def apply_scaling(arrays, center, rng, config):
    factor = float(rng.uniform(config.scale_min, config.scale_max))
    for array in arrays:
        array[:, :2] = center + (array[:, :2] - center) * factor
    return {"type": "scaling", "factor": factor}


def apply_rotation(arrays, center, rng, config):
    angle_deg = float(rng.uniform(config.rotation_min_deg, config.rotation_max_deg))
    angle_rad = math.radians(angle_deg)
    rotation = np.asarray(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=np.float64,
    )
    for array in arrays:
        array[:, :2] = (array[:, :2] - center) @ rotation.T + center
    return {"type": "rotation", "angle_deg": angle_deg}


def apply_shear(arrays, center, rng, config):
    factor = float(rng.uniform(config.shear_min, config.shear_max))
    for array in arrays:
        array[:, 0] = array[:, 0] + factor * (array[:, 1] - center[1])
    return {"type": "shear", "factor": factor}


def apply_noise(arrays, _center, rng, config):
    for array in arrays:
        array[:, :2] += rng.normal(config.noise_mean, config.noise_std, size=(len(array), 2))
    return {"type": "noise", "mean": config.noise_mean, "std": config.noise_std}


def apply_time_shift(arrays, _center, rng, config):
    factor = float(rng.uniform(config.time_scale_min, config.time_scale_max))
    changed = 0
    for array in arrays:
        if array.shape[1] >= 3:
            array[:, 2] *= factor
            changed += len(array)
    return {"type": "time_shift", "factor": factor, "points_changed": changed}


TRANSFORM_FUNCS = {
    "scaling": apply_scaling,
    "rotation": apply_rotation,
    "shear": apply_shear,
    "noise": apply_noise,
    "time_shift": apply_time_shift,
}


def choose_transforms(arrays, rng, config):
    available = list(TRANSFORM_NAMES)
    if not any(array.shape[1] >= 3 for array in arrays):
        available.remove("time_shift")

    count = int(rng.integers(config.min_transforms, config.max_transforms + 1))
    count = min(count, len(available))
    return list(rng.choice(available, size=count, replace=False))


def clean_number(value, decimals, preserve_ints):
    rounded = round(float(value), decimals)
    if preserve_ints and rounded.is_integer():
        return int(rounded)
    return rounded


def array_to_points(array, config):
    points = []
    for row in array:
        point = []
        for index, value in enumerate(row):
            decimals = config.time_decimals if index == 2 else config.decimals
            point.append(clean_number(value, decimals, config.preserve_ints))
        points.append(point)
    return points


def augment_data(data, source_path, rng, config):
    augmented, sequence_refs, schema = collect_sequences(data, source_path)
    arrays = []
    live_refs = []

    for sequence in sequence_refs:
        array = parse_point_sequence(sequence, source_path)
        if array is None:
            continue
        arrays.append(array)
        live_refs.append(sequence)

    if not arrays:
        raise ValueError(f"{source_path}: no non-empty stroke sequences found")

    center = geometric_center(arrays)
    transform_names = choose_transforms(arrays, rng, config)
    transform_params = []

    for name in transform_names:
        transform_params.append(TRANSFORM_FUNCS[name](arrays, center, rng, config))

    for sequence, array in zip(live_refs, arrays):
        sequence[:] = array_to_points(array, config)

    if isinstance(augmented, dict):
        existing_types = augmented.get("augmented_types", [])
        if not isinstance(existing_types, list):
            existing_types = [existing_types]
        augmented["augmented_types"] = existing_types + transform_names
        augmented["augmentation_params"] = transform_params
    elif schema == "list":
        # Keep list-style Calliar/Sedrah JSON loadable by existing training code.
        # Per-file augmentation metadata is still recorded in manifest.jsonl.
        pass

    return augmented, transform_names, transform_params


def output_name(input_path, generation_index, transform_names):
    suffix = "_".join(transform_names)
    return f"{input_path.stem}_aug{generation_index:03d}_{suffix}{input_path.suffix}"


def process_file(task):
    input_path, input_root, output_root, multiplier, config_dict, base_seed, overwrite = task
    input_path = Path(input_path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    config = AugmentConfig(**config_dict)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    relative_parent = input_path.relative_to(input_root).parent
    out_dir = output_root / relative_parent
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    written = 0
    for generation_index in range(1, multiplier + 1):
        rng = np.random.default_rng(stable_seed(base_seed, input_path, generation_index))
        augmented, transform_names, transform_params = augment_data(data, input_path, rng, config)
        out_path = out_dir / output_name(input_path, generation_index, transform_names)
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {out_path}")

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(augmented, f, ensure_ascii=False, separators=(",", ":"))

        manifest_rows.append(
            {
                "source": str(input_path),
                "output": str(out_path),
                "generation_index": generation_index,
                "augmented_types": transform_names,
                "augmentation_params": transform_params,
            }
        )
        written += 1

    return {"source": str(input_path), "written": written, "manifest_rows": manifest_rows}


def iter_json_files(input_dir, recursive):
    pattern = "**/*.json" if recursive else "*.json"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def write_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Synthetically augment Calliar-style handwriting trajectory JSON with NumPy transforms."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory where augmented JSON files will be written.")
    parser.add_argument("--multiplier", type=int, required=True, help="Number of augmented variants per input file.")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() or 1, 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, help="Optional cap for quick smoke tests.")
    parser.add_argument("--manifest-name", default="augmentation_manifest.jsonl")

    parser.add_argument("--scale-min", type=float, default=0.85)
    parser.add_argument("--scale-max", type=float, default=1.15)
    parser.add_argument("--rotation-min-deg", type=float, default=-5.0)
    parser.add_argument("--rotation-max-deg", type=float, default=5.0)
    parser.add_argument("--shear-min", type=float, default=-0.2)
    parser.add_argument("--shear-max", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--time-scale-min", type=float, default=0.8)
    parser.add_argument("--time-scale-max", type=float, default=1.2)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--time-decimals", type=int, default=3)
    parser.add_argument(
        "--min-transforms",
        type=int,
        default=2,
        help="Minimum number of stacked transforms per generated copy. Use 0 to allow clean, "
        "near-unaltered passthrough copies in the mix.",
    )
    parser.add_argument(
        "--max-transforms",
        type=int,
        default=3,
        help="Maximum number of stacked transforms per generated copy.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.multiplier < 1:
        raise ValueError("--multiplier must be >= 1")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("--output-dir must be different from --input-dir")

    config = AugmentConfig(
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        rotation_min_deg=args.rotation_min_deg,
        rotation_max_deg=args.rotation_max_deg,
        shear_min=args.shear_min,
        shear_max=args.shear_max,
        noise_std=args.noise_std,
        time_scale_min=args.time_scale_min,
        time_scale_max=args.time_scale_max,
        decimals=args.decimals,
        time_decimals=args.time_decimals,
        min_transforms=args.min_transforms,
        max_transforms=args.max_transforms,
    )

    files = iter_json_files(input_dir, args.recursive)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    tasks = [
        (path, input_dir, output_dir, args.multiplier, asdict(config), args.seed, args.overwrite)
        for path in files
    ]

    manifest_rows = []
    total_written = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Augmenting JSON files"):
            result = future.result()
            total_written += result["written"]
            manifest_rows.extend(result["manifest_rows"])

    manifest_path = output_dir / args.manifest_name
    write_manifest(manifest_path, manifest_rows)
    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "input_files": len(files),
                "multiplier": args.multiplier,
                "written_files": total_written,
                "manifest": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
