"""
Unity / Apple Vision Pro tracker input utilities for EgoPoser.

This module is the single source of truth for the realtime input pipeline:

1. validate the JSON message emitted by EGO_UNITY;
2. convert Unity world-space poses to the AMASS / SMPL coordinate system;
3. reproduce the 54-dimensional feature layout used by ``prepare_data.py``;
4. optionally record raw and encoded frames to a compressed NPZ file.

It can also run as a recorder-only WebSocket server.  Point Unity's
``server.json`` at ``ws://<computer-ip>:8889/ws`` while collecting data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


TRACKER_NAMES = ("head", "left_wrist", "right_wrist")
FEATURE_DIM = 54
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8889
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "support_data",
    "github_data",
    "visionpro_track.npz",
)

# Existing project convention:
#   SMPL (x, y, z) -> Unity (x, z, -y)
# Positions are column vectors.  The matrix is a proper rotation, so the same
# change of basis can be used for orientations.
SMPL_TO_UNITY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
UNITY_TO_SMPL = SMPL_TO_UNITY.T


class TrackerFrameError(ValueError):
    """The Unity tracker message is malformed."""


class HeadTrackingLost(TrackerFrameError):
    """The head pose is unavailable, so global body placement is impossible."""


@dataclass(frozen=True)
class EncodedTrackerFrame:
    """One Unity tracker message after validation and EgoPoser encoding."""

    sequence: int
    timestamp: float
    sparse_input: np.ndarray
    fov_l: bool
    fov_r: bool
    positions_unity: np.ndarray
    rotations_unity_xyzw: np.ndarray
    positions_smpl: np.ndarray
    rotations_smpl: np.ndarray
    tracked: np.ndarray

    @property
    def head_position_smpl(self) -> np.ndarray:
        return self.positions_smpl[0]


def _finite_array(value: Any, shape: Tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise TrackerFrameError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise TrackerFrameError(f"{name} contains NaN or Inf")
    return array


def quaternion_xyzw_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    """Convert Unity's JSON quaternion ``[x, y, z, w]`` to a 3x3 matrix."""

    x, y, z, w = _finite_array(quaternion, (4,), "quaternion")
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-8:
        raise TrackerFrameError("quaternion has near-zero length")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to normalized ``[x, y, z, w]``."""

    matrix = _finite_array(rotation, (3, 3), "rotation")
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return quaternion.astype(np.float32)


def rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    """EgoPoser 6D order: first matrix column, then second matrix column."""

    rotation = _finite_array(rotation, (3, 3), "rotation")
    return np.concatenate((rotation[:, 0], rotation[:, 1])).astype(np.float32)


def sixd_to_rotation_matrix(rotation_6d: Iterable[float]) -> np.ndarray:
    """Convert 6D rotation to a proper matrix using Gram-Schmidt."""

    rotation_6d = _finite_array(rotation_6d, (6,), "rotation_6d")
    first = rotation_6d[:3]
    second = rotation_6d[3:]
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = second - np.dot(first, second) * first
    second /= max(float(np.linalg.norm(second)), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=1)


def unity_to_smpl_positions(positions: np.ndarray) -> np.ndarray:
    """Map one or more Unity positions to the project's SMPL coordinates."""

    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[-1] != 3 or not np.all(np.isfinite(positions)):
        raise TrackerFrameError("positions must be finite and end in dimension 3")
    return np.matmul(positions, UNITY_TO_SMPL.T).astype(np.float32)


def smpl_to_unity_positions(positions: np.ndarray) -> np.ndarray:
    """Map one or more SMPL positions back to Unity world-space axes."""

    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[-1] != 3 or not np.all(np.isfinite(positions)):
        raise TrackerFrameError("positions must be finite and end in dimension 3")
    return np.matmul(positions, SMPL_TO_UNITY.T).astype(np.float32)


def unity_to_smpl_rotation(rotation_unity: np.ndarray) -> np.ndarray:
    """Change the basis of a Unity world rotation into SMPL coordinates."""

    rotation_unity = _finite_array(rotation_unity, (3, 3), "rotation_unity")
    return UNITY_TO_SMPL @ rotation_unity @ SMPL_TO_UNITY


def smpl_to_unity_rotation(rotation_smpl: np.ndarray) -> np.ndarray:
    """Change the basis of an SMPL world rotation back into Unity coordinates."""

    rotation_smpl = _finite_array(rotation_smpl, (3, 3), "rotation_smpl")
    return SMPL_TO_UNITY @ rotation_smpl @ UNITY_TO_SMPL


class EgoPoserFeatureEncoder:
    """
    Stateful encoder matching ``prepare_data.py`` exactly.

    Output layout:
      [0:18]   current global rotation 6D (head, left wrist, right wrist)
      [18:36]  relative rotation 6D: R[t-1]^T @ R[t]
      [36:45]  current global positions
      [45:54]  position difference: p[t] - p[t-1]

    An unavailable wrist is held at its last valid pose and marked out of FOV.
    The network mask removes its features.  Head tracking is mandatory because
    EgoPoser uses the head position to place the predicted body in the world.
    """

    def __init__(self) -> None:
        self.previous_rotations_smpl: Optional[np.ndarray] = None
        self.previous_positions_smpl: Optional[np.ndarray] = None
        self.previous_tracked: Optional[np.ndarray] = None
        self.last_valid_unity: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def reset(self) -> None:
        self.previous_rotations_smpl = None
        self.previous_positions_smpl = None
        self.previous_tracked = None
        self.last_valid_unity.clear()

    def _read_valid_node(self, message: Dict[str, Any], name: str) -> Tuple[np.ndarray, np.ndarray]:
        node = message.get(name)
        if not isinstance(node, dict):
            raise TrackerFrameError(f"{name} must be an object")
        position = _finite_array(node.get("position"), (3,), f"{name}.position")
        rotation = quaternion_xyzw_to_matrix(node.get("rotation"))
        return position, rotation

    def encode(self, message: Dict[str, Any]) -> EncodedTrackerFrame:
        if not isinstance(message, dict) or message.get("type") != "trackers":
            raise TrackerFrameError("message.type must be 'trackers'")

        head_node = message.get("head")
        if not isinstance(head_node, dict) or not bool(head_node.get("tracked", False)):
            raise HeadTrackingLost("head.tracked is false")

        head_position, head_rotation = self._read_valid_node(message, "head")
        self.last_valid_unity["head"] = (head_position.copy(), head_rotation.copy())

        positions_unity = [head_position]
        rotations_unity = [head_rotation]
        tracked = [True]

        for name in TRACKER_NAMES[1:]:
            node = message.get(name)
            is_tracked = isinstance(node, dict) and bool(node.get("tracked", False))
            if is_tracked:
                position, rotation = self._read_valid_node(message, name)
                self.last_valid_unity[name] = (position.copy(), rotation.copy())
            else:
                position, rotation = self.last_valid_unity.get(
                    name,
                    (head_position.copy(), head_rotation.copy()),
                )
            positions_unity.append(position)
            rotations_unity.append(rotation)
            tracked.append(is_tracked)

        positions_unity_array = np.stack(positions_unity).astype(np.float32)
        rotations_unity_array = np.stack(rotations_unity)
        positions_smpl = unity_to_smpl_positions(positions_unity_array)
        rotations_smpl = np.stack(
            [unity_to_smpl_rotation(rotation) for rotation in rotations_unity_array]
        )

        tracked_array = np.asarray(tracked, dtype=np.bool_)
        sparse_input = self.encode_smpl(
            rotations_smpl,
            positions_smpl,
            tracked=tracked_array,
        )
        rotations_unity_xyzw = np.stack(
            [matrix_to_quaternion_xyzw(rotation) for rotation in rotations_unity_array]
        )

        sequence = int(message.get("sequence", -1))
        timestamp = float(message.get("timestamp", 0.0))
        if not np.isfinite(timestamp):
            raise TrackerFrameError("timestamp must be finite")

        return EncodedTrackerFrame(
            sequence=sequence,
            timestamp=timestamp,
            sparse_input=sparse_input,
            fov_l=bool(tracked[1]),
            fov_r=bool(tracked[2]),
            positions_unity=positions_unity_array,
            rotations_unity_xyzw=rotations_unity_xyzw,
            positions_smpl=positions_smpl,
            rotations_smpl=rotations_smpl.astype(np.float32),
            tracked=tracked_array,
        )

    def encode_smpl(
        self,
        rotations_smpl: np.ndarray,
        positions_smpl: np.ndarray,
        tracked: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode three SMPL tracker poses.

        This public method is also used by the AMASS bridge so offline and live
        paths share the exact same feature definition.
        """

        rotations_smpl = np.asarray(rotations_smpl, dtype=np.float64)
        positions_smpl = np.asarray(positions_smpl, dtype=np.float64)
        if rotations_smpl.shape != (3, 3, 3):
            raise TrackerFrameError(
                f"rotations_smpl must have shape (3, 3, 3), got {rotations_smpl.shape}"
            )
        if positions_smpl.shape != (3, 3):
            raise TrackerFrameError(
                f"positions_smpl must have shape (3, 3), got {positions_smpl.shape}"
            )
        if not np.all(np.isfinite(rotations_smpl)) or not np.all(np.isfinite(positions_smpl)):
            raise TrackerFrameError("SMPL tracker pose contains NaN or Inf")
        if tracked is None:
            tracked = np.ones(3, dtype=np.bool_)
        tracked = np.asarray(tracked, dtype=np.bool_)
        if tracked.shape != (3,):
            raise TrackerFrameError(f"tracked must have shape (3,), got {tracked.shape}")

        current_rotation_6d = np.concatenate(
            [rotation_matrix_to_6d(rotation) for rotation in rotations_smpl]
        )

        if self.previous_rotations_smpl is None:
            relative_rotations = np.repeat(np.eye(3)[None, ...], 3, axis=0)
            position_velocity = np.zeros((3, 3), dtype=np.float64)
        else:
            relative_rotations = np.matmul(
                np.transpose(self.previous_rotations_smpl, (0, 2, 1)),
                rotations_smpl,
            )
            position_velocity = positions_smpl - self.previous_positions_smpl

            # A wrist can be absent for many frames.  On the first reacquired
            # frame there is no one-frame velocity observation, so use the
            # neutral relative rotation/translation instead of a large jump.
            reacquired = tracked & ~self.previous_tracked
            relative_rotations[reacquired] = np.eye(3)
            position_velocity[reacquired] = 0.0

        relative_rotation_6d = np.concatenate(
            [rotation_matrix_to_6d(rotation) for rotation in relative_rotations]
        )

        sparse_input = np.concatenate(
            (
                current_rotation_6d,
                relative_rotation_6d,
                positions_smpl.reshape(-1),
                position_velocity.reshape(-1),
            )
        ).astype(np.float32)

        if sparse_input.shape != (FEATURE_DIM,):
            raise AssertionError(f"internal feature shape is {sparse_input.shape}, expected (54,)")

        self.previous_rotations_smpl = rotations_smpl.copy()
        self.previous_positions_smpl = positions_smpl.copy()
        self.previous_tracked = tracked.copy()
        return sparse_input


class VisionProSessionRecorder:
    """Accumulate raw/effective tracker poses and model-ready features."""

    def __init__(self, output_path: str = DEFAULT_OUTPUT, flush_every: int = 300) -> None:
        self.output_path = os.path.abspath(output_path)
        self.flush_every = max(int(flush_every), 0)
        self.frames: list[EncodedTrackerFrame] = []
        self.body_messages: list[Optional[Dict[str, Any]]] = []

    def append(
        self,
        frame: EncodedTrackerFrame,
        body_message: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.frames.append(frame)
        self.body_messages.append(body_message)
        if self.flush_every and len(self.frames) % self.flush_every == 0:
            self.save()

    def save(self) -> Optional[str]:
        if not self.frames:
            return None

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        frame_count = len(self.frames)
        has_prediction = np.zeros(frame_count, dtype=np.bool_)
        inference_ms = np.full(frame_count, np.nan, dtype=np.float32)
        root_position_unity = np.full((frame_count, 3), np.nan, dtype=np.float32)
        root_rotation_unity_xyzw = np.full((frame_count, 4), np.nan, dtype=np.float32)
        joints_unity = np.full((frame_count, 22, 3), np.nan, dtype=np.float32)

        for index, body_message in enumerate(self.body_messages):
            if not body_message or body_message.get("type") != "body_pose":
                continue
            joints = np.asarray(body_message.get("joints_world"), dtype=np.float32)
            root_position = np.asarray(body_message.get("root_position"), dtype=np.float32)
            root_rotation = np.asarray(body_message.get("root_rotation"), dtype=np.float32)
            if joints.shape != (22, 3) or root_position.shape != (3,) or root_rotation.shape != (4,):
                continue
            if not (
                np.all(np.isfinite(joints))
                and np.all(np.isfinite(root_position))
                and np.all(np.isfinite(root_rotation))
            ):
                continue
            has_prediction[index] = True
            inference_ms[index] = float(body_message.get("inference_ms", np.nan))
            root_position_unity[index] = root_position
            root_rotation_unity_xyzw[index] = root_rotation
            joints_unity[index] = joints

        arrays = {
            "sequence": np.asarray([frame.sequence for frame in self.frames], dtype=np.int64),
            "timestamp": np.asarray([frame.timestamp for frame in self.frames], dtype=np.float64),
            "sparse_input": np.stack([frame.sparse_input for frame in self.frames]),
            "fov_l": np.asarray([frame.fov_l for frame in self.frames], dtype=np.bool_),
            "fov_r": np.asarray([frame.fov_r for frame in self.frames], dtype=np.bool_),
            "tracked": np.stack([frame.tracked for frame in self.frames]),
            "positions_unity": np.stack([frame.positions_unity for frame in self.frames]),
            "rotations_unity_xyzw": np.stack(
                [frame.rotations_unity_xyzw for frame in self.frames]
            ),
            "positions_smpl": np.stack([frame.positions_smpl for frame in self.frames]),
            "rotations_smpl": np.stack([frame.rotations_smpl for frame in self.frames]),
            "has_prediction": has_prediction,
            "inference_ms": inference_ms,
            "root_position_unity": root_position_unity,
            "root_rotation_unity_xyzw": root_rotation_unity_xyzw,
            "joints_unity": joints_unity,
        }

        temporary_path = self.output_path + ".tmp.npz"
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, self.output_path)
        return self.output_path


def run_self_test() -> None:
    """Pure NumPy checks for coordinate and feature consistency."""

    rng = np.random.default_rng(22)
    positions_unity = rng.normal(size=(32, 3))
    positions_roundtrip = smpl_to_unity_positions(
        unity_to_smpl_positions(positions_unity)
    )
    np.testing.assert_allclose(positions_roundtrip, positions_unity, atol=1e-6)

    encoder = EgoPoserFeatureEncoder()
    previous_rotations = np.repeat(np.eye(3)[None, ...], 3, axis=0)
    previous_positions = rng.normal(size=(3, 3))
    first = encoder.encode_smpl(previous_rotations, previous_positions)
    np.testing.assert_allclose(first[18:36], np.tile([1, 0, 0, 0, 1, 0], 3), atol=1e-6)
    np.testing.assert_allclose(first[45:54], 0.0, atol=1e-6)

    angle = 0.15
    rotation_delta = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    current_rotations = np.matmul(previous_rotations, rotation_delta)
    current_positions = previous_positions + 0.01
    second = encoder.encode_smpl(current_rotations, current_positions)
    expected_relative = np.matmul(
        np.transpose(previous_rotations, (0, 2, 1)),
        current_rotations,
    )
    expected_relative_6d = np.concatenate(
        [rotation_matrix_to_6d(rotation) for rotation in expected_relative]
    )
    np.testing.assert_allclose(second[18:36], expected_relative_6d, atol=1e-6)
    np.testing.assert_allclose(second[45:54], 0.01, atol=1e-6)

    test_rotation = quaternion_xyzw_to_matrix([0.2, -0.1, 0.3, 0.9])
    rotation_roundtrip = unity_to_smpl_rotation(
        smpl_to_unity_rotation(test_rotation)
    )
    np.testing.assert_allclose(rotation_roundtrip, test_rotation, atol=1e-6)
    print("[self-test] coordinate transforms and 54D encoding passed")


async def run_recorder_server(
    host: str,
    port: int,
    output: str,
    flush_every: int,
) -> None:
    """Run the Unity-compatible recorder-only WebSocket endpoint."""

    import websockets

    recorder = VisionProSessionRecorder(output, flush_every)

    async def handle(websocket: Any) -> None:
        encoder = EgoPoserFeatureEncoder()
        peer = getattr(websocket, "remote_address", "?")
        print(f"[recorder] Unity connected: {peer}")
        try:
            async for payload in websocket:
                try:
                    message = json.loads(payload)
                    if message.get("type") != "trackers":
                        continue
                    encoded = encoder.encode(message)
                    recorder.append(encoded)
                except HeadTrackingLost as error:
                    encoder.reset()
                    print(f"[recorder] tracking paused: {error}")
                except (json.JSONDecodeError, TrackerFrameError) as error:
                    print(f"[recorder] rejected frame: {error}")
        finally:
            saved = recorder.save()
            print(f"[recorder] Unity disconnected: {peer}; saved={saved}")

    async with websockets.serve(
        handle,
        host,
        port,
        max_size=1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ):
        print(f"[recorder] listening on ws://{host}:{port}/ws")
        print(f"[recorder] output: {os.path.abspath(output)}")
        await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record EGO_UNITY tracker messages")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--flush-every", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    try:
        asyncio.run(
            run_recorder_server(
                host=args.host,
                port=args.port,
                output=args.output,
                flush_every=args.flush_every,
            )
        )
    except KeyboardInterrupt:
        print("\n[recorder] stopped")


if __name__ == "__main__":
    main()
