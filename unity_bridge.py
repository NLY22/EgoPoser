"""
AMASS playback bridge for the EGO_UNITY WebSocket client.

This is an offline diagnostic tool, not the live Vision Pro inference server.
It accepts the same ``trackers`` JSON as ``egoposer_server.py`` and uses each
incoming frame as a clock tick.  It returns either:

* ``raw``: ground-truth AMASS/SMPL-H joints; or
* ``model``: pretrained EgoPoser predictions from correctly encoded AMASS data.

The returned body is translated so joint 15 (head) matches Unity's current
head position.  This isolates Unity rendering/protocol problems from live
tracker-domain problems.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


EGOPOSER_DIR = os.path.dirname(os.path.abspath(__file__))
if EGOPOSER_DIR not in sys.path:
    sys.path.insert(0, EGOPOSER_DIR)

from human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from egoposer_server import (
    DEFAULT_YAML,
    EgoPoserModelRunner,
    ModelPrediction,
    load_model,
)
from utils import utils_transform
from vision_pro_receiver import (
    EgoPoserFeatureEncoder,
    TrackerFrameError,
    matrix_to_quaternion_xyzw,
    smpl_to_unity_positions,
    smpl_to_unity_rotation,
)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_DATA = os.path.join(
    EGOPOSER_DIR,
    "support_data",
    "github_data",
    "dmpl_sample.npz",
)
TRACKER_JOINTS = (15, 20, 21)


@dataclass(frozen=True)
class AmassSequence:
    sparse_input: np.ndarray
    head_positions_smpl: np.ndarray
    joints_smpl: np.ndarray
    root_rotations_smpl: np.ndarray

    @property
    def length(self) -> int:
        return int(self.sparse_input.shape[0])


def load_amass_sequence(
    data_path: str,
    body_model: Any,
    device: torch.device,
    max_frames: int,
) -> AmassSequence:
    """Prepare AMASS with the same frame alignment and features as prepare_data.py."""

    data_path = os.path.abspath(data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"AMASS sample does not exist: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    if "poses" not in data or "trans" not in data:
        raise ValueError("AMASS NPZ must contain 'poses' and 'trans'")

    source_fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else 60.0
    stride = max(1, int(round(source_fps / 60.0)))
    poses_numpy = np.asarray(data["poses"])[::stride]
    trans_numpy = np.asarray(data["trans"])[::stride]
    frame_count = min(int(max_frames), poses_numpy.shape[0], trans_numpy.shape[0])
    if frame_count < 2:
        raise ValueError("AMASS sequence must contain at least two 60 Hz frames")

    poses = torch.as_tensor(
        poses_numpy[:frame_count],
        dtype=torch.float32,
        device=device,
    )
    trans = torch.as_tensor(
        trans_numpy[:frame_count],
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        local_rotations = aa2matrot(poses.reshape(-1, 3)).reshape(frame_count, -1, 9)
        global_rotations = local2global_pose(
            local_rotations,
            body_model.kintree_table[0].long(),
        )
        body = body_model(
            pose_body=poses[:, 3:66],
            root_orient=poses[:, :3],
            trans=trans,
        )
        joints_smpl = body.Jtr[:, :22].detach().cpu().numpy()

    tracker_rotations = (
        global_rotations[:, TRACKER_JOINTS].detach().cpu().numpy()
    )
    tracker_positions = joints_smpl[:, TRACKER_JOINTS]

    # Seed frame 0 as history, then encode frames 1..N-1.  This exactly matches
    # prepare_data.py, which drops the first pose after computing velocities.
    encoder = EgoPoserFeatureEncoder()
    encoder.encode_smpl(tracker_rotations[0], tracker_positions[0])
    sparse_frames = [
        encoder.encode_smpl(tracker_rotations[index], tracker_positions[index])
        for index in range(1, frame_count)
    ]
    sparse_input = np.stack(sparse_frames).astype(np.float32)

    # Independent vectorized equality check against the original implementation.
    rotations_current = global_rotations[1:, TRACKER_JOINTS]
    rotations_relative = torch.matmul(
        torch.inverse(global_rotations[:-1]),
        global_rotations[1:],
    )[:, TRACKER_JOINTS]
    current_6d = utils_transform.matrot2sixd(
        rotations_current.reshape(-1, 3, 3)
    ).reshape(frame_count - 1, -1)
    relative_6d = utils_transform.matrot2sixd(
        rotations_relative.reshape(-1, 3, 3)
    ).reshape(frame_count - 1, -1)
    positions_current = torch.as_tensor(
        tracker_positions[1:].reshape(frame_count - 1, -1),
        device=device,
    )
    positions_velocity = torch.as_tensor(
        (tracker_positions[1:] - tracker_positions[:-1]).reshape(frame_count - 1, -1),
        device=device,
    )
    original_features = torch.cat(
        (current_6d, relative_6d, positions_current, positions_velocity),
        dim=-1,
    ).detach().cpu().numpy()
    np.testing.assert_allclose(sparse_input, original_features, atol=1e-5, rtol=1e-5)

    return AmassSequence(
        sparse_input=sparse_input,
        head_positions_smpl=tracker_positions[1:, 0].astype(np.float32),
        joints_smpl=joints_smpl[1:].astype(np.float32),
        root_rotations_smpl=global_rotations[1:, 0].detach().cpu().numpy().astype(np.float32),
    )


def _unity_head_position(message: dict[str, Any]) -> np.ndarray:
    head = message.get("head")
    if not isinstance(head, dict) or not bool(head.get("tracked", False)):
        raise TrackerFrameError("head.tracked is false")
    position = np.asarray(head.get("position"), dtype=np.float32)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise TrackerFrameError("head.position must contain three finite values")
    return position


def build_playback_message(
    prediction: ModelPrediction,
    unity_head_position: np.ndarray,
    sequence: int,
    inference_ms: float,
) -> dict[str, Any]:
    joints_unity = smpl_to_unity_positions(prediction.joints_smpl)
    joints_unity += unity_head_position.reshape(1, 3) - joints_unity[15]
    root_rotation_unity = smpl_to_unity_rotation(prediction.root_rotation_smpl)
    root_quaternion = matrix_to_quaternion_xyzw(root_rotation_unity)
    return {
        "type": "body_pose",
        "sequence": int(sequence),
        "inference_ms": round(float(inference_ms), 2),
        "root_position": joints_unity[0].tolist(),
        "root_rotation": root_quaternion.tolist(),
        "joints_world": joints_unity.tolist(),
    }


class AmassPlaybackServer:
    def __init__(
        self,
        model: Any,
        sequence: AmassSequence,
        mode: str,
        host: str,
        port: int,
        window_size: int,
    ) -> None:
        self.model = model
        self.sequence = sequence
        self.mode = mode
        self.host = host
        self.port = int(port)
        self.window_size = int(window_size)
        self.active_connection = False

    async def handle_connection(self, websocket: Any) -> None:
        if self.active_connection:
            await websocket.close(code=1013, reason="Only one Unity client is supported")
            return
        self.active_connection = True
        peer = getattr(websocket, "remote_address", "?")
        runner = EgoPoserModelRunner(self.model, self.window_size)
        frame_index = 0
        print(f"[bridge] Unity connected: {peer}")

        try:
            async for payload in websocket:
                try:
                    message = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if message.get("type") != "trackers":
                    continue

                try:
                    head_unity = _unity_head_position(message)
                except TrackerFrameError:
                    continue

                sequence_number = int(message.get("sequence", -1))
                source_index = frame_index % self.sequence.length
                if source_index == 0 and frame_index > 0:
                    runner.reset()

                started = time.perf_counter()
                if self.mode == "raw":
                    prediction = ModelPrediction(
                        joints_smpl=self.sequence.joints_smpl[source_index],
                        root_rotation_smpl=self.sequence.root_rotations_smpl[source_index],
                    )
                else:
                    prediction = runner.step(
                        self.sequence.sparse_input[source_index],
                        self.sequence.head_positions_smpl[source_index],
                        True,
                        True,
                    )
                inference_ms = (time.perf_counter() - started) * 1000.0
                frame_index += 1

                if prediction is None:
                    response = {
                        "type": "warming_up",
                        "sequence": sequence_number,
                        "frames_collected": runner.frames_collected,
                        "frames_needed": runner.window_size,
                    }
                else:
                    response = build_playback_message(
                        prediction,
                        unity_head_position=head_unity,
                        sequence=sequence_number,
                        inference_ms=inference_ms,
                    )
                await websocket.send(json.dumps(response, separators=(",", ":")))
        except Exception as error:
            print(f"[bridge] connection ended: {error}")
        finally:
            self.active_connection = False
            print(f"[bridge] Unity disconnected: {peer}")

    async def run(self) -> None:
        import websockets

        print("=" * 64)
        print(f"AMASS -> EGO_UNITY bridge ({self.mode})")
        print(f"WebSocket: ws://{self.host}:{self.port}/ws")
        print(f"Frames: {self.sequence.length}")
        print("=" * 64)
        async with websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            max_size=1024 * 1024,
            ping_interval=20,
            ping_timeout=60,
        ):
            await asyncio.Future()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play AMASS through EGO_UNITY")
    parser.add_argument("--mode", choices=("raw", "model"), default="raw")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--yaml", default=DEFAULT_YAML)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--window-size", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model = load_model(
            yaml_path=args.yaml,
            pretrained_path=args.checkpoint,
            device=args.device,
        )
        sequence = load_amass_sequence(
            data_path=args.data,
            body_model=model.bm,
            device=model.device,
            max_frames=args.max_frames,
        )
        server = AmassPlaybackServer(
            model=model,
            sequence=sequence,
            mode=args.mode,
            host=args.host,
            port=args.port,
            window_size=args.window_size,
        )
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[bridge] stopped")


if __name__ == "__main__":
    main()
