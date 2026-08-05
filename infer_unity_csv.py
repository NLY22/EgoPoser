"""Run EgoPoser offline on a CSV recorded by NLY22/EGO_UNITY.

CSV rows are converted to the same trackers messages used by the live
WebSocket path. Offline and live inference therefore share coordinate
conversion, 54D encoding, model inference, SMPL-H FK, and NPZ recording.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np

from vision_pro_receiver import (
    EgoPoserFeatureEncoder,
    HeadTrackingLost,
    TrackerFrameError,
    VisionProSessionRecorder,
)


TRACKERS = ("head", "left", "right")
REQUIRED_COLUMNS = (
    "sequence",
    "timestamp",
    *(
        column
        for tracker in TRACKERS
        for column in (
            f"{tracker}_tracked",
            f"{tracker}_px",
            f"{tracker}_py",
            f"{tracker}_pz",
            f"{tracker}_qx",
            f"{tracker}_qy",
            f"{tracker}_qz",
            f"{tracker}_qw",
        )
    ),
)


def _default_output_path(input_path: str) -> str:
    stem, _ = os.path.splitext(os.path.abspath(input_path))
    return stem + "_egoposer.npz"


def _parse_float(row: Dict[str, str], column: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{column} must be a number") from error
    if not np.isfinite(value):
        raise ValueError(f"{column} contains NaN or Inf")
    return value


def _parse_int(row: Dict[str, str], column: str) -> int:
    value = _parse_float(row, column)
    integer = int(value)
    if value != integer:
        raise ValueError(f"{column} must be an integer")
    return integer


def _parse_bool(row: Dict[str, str], column: str) -> bool:
    try:
        value = str(row[column]).strip().lower()
    except KeyError as error:
        raise ValueError(f"missing column: {column}") from error
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise ValueError(f"{column} must be one of: 0, 1, false, true")


def _tracker_node(row: Dict[str, str], prefix: str) -> Dict[str, Any]:
    return {
        "position": [
            _parse_float(row, f"{prefix}_px"),
            _parse_float(row, f"{prefix}_py"),
            _parse_float(row, f"{prefix}_pz"),
        ],
        "rotation": [
            _parse_float(row, f"{prefix}_qx"),
            _parse_float(row, f"{prefix}_qy"),
            _parse_float(row, f"{prefix}_qz"),
            _parse_float(row, f"{prefix}_qw"),
        ],
        "tracked": _parse_bool(row, f"{prefix}_tracked"),
    }


def row_to_tracker_message(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert one EGO_UNITY CSV row to the live WebSocket contract."""

    return {
        "type": "trackers",
        "sequence": _parse_int(row, "sequence"),
        "timestamp": _parse_float(row, "timestamp"),
        "head": _tracker_node(row, "head"),
        "left_wrist": _tracker_node(row, "left"),
        "right_wrist": _tracker_node(row, "right"),
    }


def _head_has_zero_position(message: Dict[str, Any], epsilon: float) -> bool:
    position = np.asarray(message["head"]["position"], dtype=np.float64)
    return float(np.linalg.norm(position)) <= epsilon


def _validate_header(fieldnames: Optional[Iterable[str]]) -> None:
    actual = set(fieldnames or ())
    missing = [column for column in REQUIRED_COLUMNS if column not in actual]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))


def run_csv(
    input_path: str,
    output_path: str,
    yaml_path: str,
    checkpoint: Optional[str],
    device: str,
    window_size: int,
    reset_gap: float,
    start_sequence: Optional[int],
    allow_zero_head: bool,
    zero_head_epsilon: float,
    validate_only: bool,
) -> Dict[str, Any]:
    """Encode or infer all usable rows and save a frame-aligned NPZ."""

    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if reset_gap <= 0:
        raise ValueError("reset_gap must be greater than zero")
    if zero_head_epsilon < 0:
        raise ValueError("zero_head_epsilon cannot be negative")

    model = None
    runner = None
    build_body_message = None
    if not validate_only:
        from egoposer_server import (
            EgoPoserModelRunner,
            load_model,
            prediction_to_unity_message,
        )

        model = load_model(
            yaml_path=yaml_path,
            pretrained_path=checkpoint,
            device=device,
        )
        runner = EgoPoserModelRunner(model, window_size=window_size)
        build_body_message = prediction_to_unity_message

    encoder = EgoPoserFeatureEncoder()
    recorder = VisionProSessionRecorder(output_path, flush_every=0)

    stats: Dict[str, Any] = {
        "rows_read": 0,
        "rows_before_start": 0,
        "rows_nonmonotonic": 0,
        "rows_head_untracked": 0,
        "rows_zero_head": 0,
        "discontinuities": 0,
        "encoded_frames": 0,
        "predictions": 0,
        "first_prediction_sequence": None,
        "inference_failures": 0,
    }
    last_sequence: Optional[int] = None
    last_timestamp: Optional[float] = None

    with open(input_path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        _validate_header(reader.fieldnames)

        for line_number, row in enumerate(reader, start=2):
            stats["rows_read"] += 1
            try:
                message = row_to_tracker_message(row)
            except ValueError as error:
                raise ValueError(f"{input_path}:{line_number}: {error}") from error

            sequence = int(message["sequence"])
            timestamp = float(message["timestamp"])

            if start_sequence is not None and sequence < start_sequence:
                stats["rows_before_start"] += 1
                continue
            if last_sequence is not None and sequence <= last_sequence:
                stats["rows_nonmonotonic"] += 1
                continue

            if last_timestamp is not None:
                delta = timestamp - last_timestamp
                if delta <= 0.0 or delta > reset_gap:
                    encoder.reset()
                    if runner is not None:
                        runner.reset()
                    stats["discontinuities"] += 1

            last_sequence = sequence
            last_timestamp = timestamp

            if not bool(message["head"]["tracked"]):
                encoder.reset()
                if runner is not None:
                    runner.reset()
                stats["rows_head_untracked"] += 1
                continue

            if not allow_zero_head and _head_has_zero_position(
                message, zero_head_epsilon
            ):
                encoder.reset()
                if runner is not None:
                    runner.reset()
                stats["rows_zero_head"] += 1
                continue

            try:
                encoded = encoder.encode(message)
            except (HeadTrackingLost, TrackerFrameError) as error:
                raise ValueError(f"{input_path}:{line_number}: {error}") from error

            stats["encoded_frames"] += 1
            body_message = None

            if runner is not None:
                started = time.perf_counter()
                try:
                    prediction = runner.step(
                        encoded.sparse_input,
                        encoded.head_position_smpl,
                        encoded.fov_l,
                        encoded.fov_r,
                    )
                except (FloatingPointError, RuntimeError) as error:
                    runner.reset()
                    stats["inference_failures"] += 1
                    print(
                        f"[warning] inference failed at sequence {sequence}: {error}"
                    )
                    prediction = None
                inference_ms = (time.perf_counter() - started) * 1000.0

                if prediction is not None:
                    body_message = build_body_message(
                        prediction,
                        sequence=sequence,
                        inference_ms=inference_ms,
                    )
                    stats["predictions"] += 1
                    if stats["first_prediction_sequence"] is None:
                        stats["first_prediction_sequence"] = sequence

            recorder.append(encoded, body_message)

    saved_path = recorder.save()
    if saved_path is None:
        raise RuntimeError(
            "No usable frames were encoded; check head tracking and --start-sequence"
        )

    return {
        "input": os.path.abspath(input_path),
        "output": saved_path,
        **stats,
        "validate_only": validate_only,
        "device": None if model is None else str(model.device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EgoPoser offline on an EGO_UNITY tracking CSV"
    )
    parser.add_argument("input", help="CSV exported by EGO_UNITY")
    parser.add_argument(
        "--output",
        default=None,
        help="output NPZ (default: <input>_egoposer.npz)",
    )
    parser.add_argument(
        "--yaml",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "options",
            "test_egoposer.yaml",
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--window-size", type=int, default=80)
    parser.add_argument("--reset-gap", type=float, default=0.5)
    parser.add_argument(
        "--start-sequence",
        type=int,
        default=None,
        help="ignore rows before this sequence",
    )
    parser.add_argument(
        "--allow-zero-head",
        action="store_true",
        help="accept a tracked head whose position is at the world origin",
    )
    parser.add_argument(
        "--zero-head-epsilon",
        type=float,
        default=1e-8,
        help="head-position norm treated as an invalid all-zero sentinel",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="encode 54D features without loading PyTorch, checkpoint, or SMPL-H",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or _default_output_path(args.input)
    try:
        summary = run_csv(
            input_path=args.input,
            output_path=output_path,
            yaml_path=args.yaml,
            checkpoint=args.checkpoint,
            device=args.device,
            window_size=args.window_size,
            reset_gap=args.reset_gap,
            start_sequence=args.start_sequence,
            allow_zero_head=args.allow_zero_head,
            zero_head_epsilon=args.zero_head_epsilon,
            validate_only=args.validate_only,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error

    print("=" * 64)
    print("EGO_UNITY CSV offline inference")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 64)
    if summary["validate_only"]:
        print("Validation complete; no model predictions were requested.")
    else:
        print("Use has_prediction to select valid rows from joints_unity [N, 22, 3].")


if __name__ == "__main__":
    main()
