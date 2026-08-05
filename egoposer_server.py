"""
Realtime EgoPoser inference server for the EGO_UNITY visionOS project.

Unity sends head and wrist world poses as JSON over WebSocket.  This server
converts them to the exact 54D representation used by ``prepare_data.py``,
runs the pretrained 80-frame EgoPoser model, performs SMPL-H forward
kinematics, and returns 22 joints in Unity world-space metres.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


EGOPOSER_DIR = os.path.dirname(os.path.abspath(__file__))
if EGOPOSER_DIR not in sys.path:
    sys.path.insert(0, EGOPOSER_DIR)

from human_body_prior.tools.rotation_tools import aa2matrot
from models.select_model import define_Model
from utils import utils_option as option
from utils import utils_transform
from vision_pro_receiver import (
    EgoPoserFeatureEncoder,
    HeadTrackingLost,
    TrackerFrameError,
    VisionProSessionRecorder,
    matrix_to_quaternion_xyzw,
    smpl_to_unity_positions,
    smpl_to_unity_rotation,
)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8888
DEFAULT_WINDOW_SIZE = 80
DEFAULT_YAML = os.path.join(EGOPOSER_DIR, "options", "test_egoposer.yaml")


@dataclass(frozen=True)
class ModelPrediction:
    joints_smpl: np.ndarray
    root_rotation_smpl: np.ndarray


def _absolute_repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(EGOPOSER_DIR, path)


def load_model(
    yaml_path: str = DEFAULT_YAML,
    pretrained_path: Optional[str] = None,
    device: str = "auto",
) -> Any:
    """Load the original model without depending on the caller's CWD."""

    yaml_path = os.path.abspath(yaml_path)
    opt = option.parse(yaml_path, is_train=True)

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    use_cuda = torch.cuda.is_available() if device == "auto" else device == "cuda"
    opt["gpu_ids"] = opt.get("gpu_ids", [0]) if use_cuda else None
    opt["num_gpu"] = len(opt["gpu_ids"]) if opt["gpu_ids"] is not None else 0

    smpl_path = _absolute_repo_path(opt["body_model"]["smpl_path"])
    opt["body_model"]["smpl_path"] = smpl_path
    body_model_path = os.path.join(smpl_path, "body_models", "smplh", "male", "model.npz")
    if not os.path.isfile(body_model_path):
        raise FileNotFoundError(
            "SMPL-H body model is missing. Expected: "
            f"{body_model_path}. Follow the original EgoPoser README setup steps."
        )

    checkpoint = pretrained_path or opt.get("pretrained_model")
    if not checkpoint:
        raise ValueError("No pretrained EgoPoser checkpoint was configured")
    checkpoint = os.path.abspath(_absolute_repo_path(checkpoint))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            "Pretrained EgoPoser checkpoint is missing. Expected: "
            f"{checkpoint}. Download egoposer.pth into model_zoo/."
        )

    opt["path"]["pretrained"] = checkpoint
    opt = option.dict_to_nonedict(opt)

    model = define_Model(opt)
    model.load(test=True)
    model.net.eval()
    return model


class EgoPoserModelRunner:
    """Maintain one connection's 80-frame window and run model inference."""

    def __init__(self, model: Any, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        self.model = model
        self.body_model = model.bm
        self.device = model.device
        self.window_size = int(window_size)
        self.sparse_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.fov_l_buffer: deque[bool] = deque(maxlen=self.window_size)
        self.fov_r_buffer: deque[bool] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.sparse_buffer.clear()
        self.fov_l_buffer.clear()
        self.fov_r_buffer.clear()

    @property
    def frames_collected(self) -> int:
        return len(self.sparse_buffer)

    def step(
        self,
        sparse_input: np.ndarray,
        head_position_smpl: np.ndarray,
        fov_l: bool,
        fov_r: bool,
    ) -> Optional[ModelPrediction]:
        sparse_input = np.asarray(sparse_input, dtype=np.float32)
        head_position_smpl = np.asarray(head_position_smpl, dtype=np.float32)
        if sparse_input.shape != (54,):
            raise ValueError(f"sparse_input must have shape (54,), got {sparse_input.shape}")
        if head_position_smpl.shape != (3,):
            raise ValueError(
                f"head_position_smpl must have shape (3,), got {head_position_smpl.shape}"
            )

        self.sparse_buffer.append(sparse_input.copy())
        self.fov_l_buffer.append(bool(fov_l))
        self.fov_r_buffer.append(bool(fov_r))
        if len(self.sparse_buffer) < self.window_size:
            return None

        sparse = torch.as_tensor(
            np.stack(self.sparse_buffer),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        fov_l_tensor = torch.as_tensor(
            np.asarray(self.fov_l_buffer),
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        fov_r_tensor = torch.as_tensor(
            np.asarray(self.fov_r_buffer),
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        network_input = {
            "sparse_input": sparse,
            "fov_l": fov_l_tensor,
            "fov_r": fov_r_tensor,
        }

        with torch.inference_mode():
            output = self.model.net(network_input)
            root_orient_6d = output["root_orient"].reshape(-1, 6)
            pose_body_6d = output["pose_body"].reshape(-1, 6)

            # Keep the same conversion path as ModelEgoPoser.test().
            root_orient_aa = utils_transform.sixd2aa(root_orient_6d).reshape(-1, 3).float()
            pose_body_aa = utils_transform.sixd2aa(pose_body_6d).reshape(-1, 63).float()

            local_kwargs = {
                "pose_body": pose_body_aa,
                "root_orient": root_orient_aa,
            }
            betas = output.get("betas")
            if betas is not None:
                local_kwargs["betas"] = betas.reshape(-1, 16).float()

            body_local = self.body_model(**local_kwargs)
            head_relative_to_root = body_local.Jtr[:, 15, :]
            target_head = torch.as_tensor(
                head_position_smpl,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3)
            root_translation = target_head - head_relative_to_root

            world_kwargs = dict(local_kwargs)
            world_kwargs["trans"] = root_translation
            body_world = self.body_model(**world_kwargs)
            joints_smpl = body_world.Jtr[0, :22].detach().cpu().numpy()
            root_rotation_smpl = (
                aa2matrot(root_orient_aa)[0].detach().cpu().numpy()
            )

        if not np.all(np.isfinite(joints_smpl)):
            raise FloatingPointError("model produced NaN or Inf joint positions")

        return ModelPrediction(
            joints_smpl=joints_smpl.astype(np.float32),
            root_rotation_smpl=root_rotation_smpl.astype(np.float32),
        )


def prediction_to_unity_message(
    prediction: ModelPrediction,
    sequence: int,
    inference_ms: float,
) -> dict[str, Any]:
    """Build the exact ``BodyPoseMessage`` JSON contract used by EGO_UNITY."""

    joints_unity = smpl_to_unity_positions(prediction.joints_smpl)
    root_rotation_unity = smpl_to_unity_rotation(prediction.root_rotation_smpl)
    root_quaternion_unity = matrix_to_quaternion_xyzw(root_rotation_unity)

    return {
        "type": "body_pose",
        "sequence": int(sequence),
        "inference_ms": round(float(inference_ms), 2),
        "root_position": joints_unity[0].tolist(),
        "root_rotation": root_quaternion_unity.tolist(),
        "joints_world": joints_unity.tolist(),
    }


def _connection_path(websocket: Any) -> Optional[str]:
    request = getattr(websocket, "request", None)
    if request is not None:
        return getattr(request, "path", None)
    return getattr(websocket, "path", None)


class EgoPoserServer:
    def __init__(
        self,
        model: Any,
        host: str,
        port: int,
        window_size: int,
        reset_gap_seconds: float,
        record_path: Optional[str],
        flush_every: int,
    ) -> None:
        self.model = model
        self.host = host
        self.port = int(port)
        self.window_size = int(window_size)
        self.reset_gap_seconds = float(reset_gap_seconds)
        self.recorder = (
            VisionProSessionRecorder(record_path, flush_every) if record_path else None
        )
        self.active_connection = False
        self.total_inferences = 0

    async def _send(self, websocket: Any, message: dict[str, Any]) -> None:
        await websocket.send(json.dumps(message, separators=(",", ":")))

    async def handle_connection(self, websocket: Any) -> None:
        path = _connection_path(websocket)
        if path not in {None, "/", "/ws"}:
            await websocket.close(code=1008, reason="Use WebSocket path /ws")
            return
        if self.active_connection:
            await websocket.close(code=1013, reason="Only one Unity client is supported")
            return

        self.active_connection = True
        peer = getattr(websocket, "remote_address", "?")
        encoder = EgoPoserFeatureEncoder()
        runner = EgoPoserModelRunner(self.model, self.window_size)
        last_sequence: Optional[int] = None
        last_timestamp: Optional[float] = None
        received_since_log = 0
        log_started = time.perf_counter()
        print(f"[ws] Unity connected: {peer}")

        try:
            async for raw_message in websocket:
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError as error:
                    await self._send(
                        websocket,
                        {"type": "error", "code": "invalid_json", "message": str(error)},
                    )
                    continue

                if message.get("type") != "trackers":
                    continue

                sequence = int(message.get("sequence", -1))
                timestamp = float(message.get("timestamp", 0.0))
                if last_sequence is not None and sequence <= last_sequence:
                    continue

                if last_timestamp is not None:
                    timestamp_delta = timestamp - last_timestamp
                    if timestamp_delta <= 0.0 or timestamp_delta > self.reset_gap_seconds:
                        encoder.reset()
                        runner.reset()
                        print(
                            "[tracking] discontinuity; window reset "
                            f"(dt={timestamp_delta:.3f}s, seq={sequence})"
                        )

                last_sequence = sequence
                last_timestamp = timestamp

                try:
                    encoded = encoder.encode(message)
                except HeadTrackingLost as error:
                    encoder.reset()
                    runner.reset()
                    await self._send(
                        websocket,
                        {
                            "type": "tracking_lost",
                            "sequence": sequence,
                            "reason": str(error),
                        },
                    )
                    continue
                except TrackerFrameError as error:
                    await self._send(
                        websocket,
                        {
                            "type": "error",
                            "sequence": sequence,
                            "code": "invalid_trackers",
                            "message": str(error),
                        },
                    )
                    continue

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
                    if self.recorder is not None:
                        self.recorder.append(encoded)
                    await self._send(
                        websocket,
                        {
                            "type": "error",
                            "sequence": sequence,
                            "code": "inference_failed",
                            "message": str(error),
                        },
                    )
                    continue
                inference_ms = (time.perf_counter() - started) * 1000.0

                if prediction is None:
                    if self.recorder is not None:
                        self.recorder.append(encoded)
                    await self._send(
                        websocket,
                        {
                            "type": "warming_up",
                            "sequence": sequence,
                            "frames_collected": runner.frames_collected,
                            "frames_needed": runner.window_size,
                        },
                    )
                else:
                    body_message = prediction_to_unity_message(
                        prediction,
                        sequence=sequence,
                        inference_ms=inference_ms,
                    )
                    if self.recorder is not None:
                        self.recorder.append(encoded, body_message)
                    await self._send(
                        websocket,
                        body_message,
                    )
                    self.total_inferences += 1

                received_since_log += 1
                now = time.perf_counter()
                if now - log_started >= 5.0:
                    fps = received_since_log / (now - log_started)
                    print(
                        f"[stats] recv={fps:.1f} Hz, seq={sequence}, "
                        f"window={runner.frames_collected}/{runner.window_size}, "
                        f"inferences={self.total_inferences}"
                    )
                    received_since_log = 0
                    log_started = now
        except Exception as error:
            # websockets raises a version-specific ConnectionClosed subclass.
            print(f"[ws] connection ended: {error}")
        finally:
            self.active_connection = False
            if self.recorder is not None:
                self.recorder.save()
            print(f"[ws] Unity disconnected: {peer}")

    async def run(self) -> None:
        import websockets

        print("=" * 64)
        print("EgoPoser realtime server")
        print(f"WebSocket: ws://{self.host}:{self.port}/ws")
        print(f"Device: {self.model.device}")
        print(f"Window: {self.window_size} frames")
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
    parser = argparse.ArgumentParser(description="Run EGO_UNITY + EgoPoser inference")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--yaml", default=DEFAULT_YAML)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--reset-gap",
        type=float,
        default=0.5,
        help="reset temporal state after this many seconds without a frame",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="optional NPZ path for raw and 54D encoded session data",
    )
    parser.add_argument("--flush-every", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        model = load_model(
            yaml_path=args.yaml,
            pretrained_path=args.checkpoint,
            device=args.device,
        )
        server = EgoPoserServer(
            model=model,
            host=args.host,
            port=args.port,
            window_size=args.window_size,
            reset_gap_seconds=args.reset_gap,
            record_path=args.record,
            flush_every=args.flush_every,
        )
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[server] stopped")


if __name__ == "__main__":
    main()
