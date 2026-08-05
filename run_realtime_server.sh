#!/usr/bin/env bash
set -Eeuo pipefail

# Start the Vision Pro -> EgoPoser -> Unity realtime pipeline on Linux or WSL.
# Activate the "egoposer" environment before running this script.

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${EGOPOSER_HOST:-0.0.0.0}"
PORT="${EGOPOSER_PORT:-8888}"
DEVICE="${EGOPOSER_DEVICE:-auto}"
WINDOW_SIZE="${EGOPOSER_WINDOW_SIZE:-80}"
RESET_GAP="${EGOPOSER_RESET_GAP:-0.5}"
YAML_PATH="${EGOPOSER_YAML:-$REPO_DIR/options/test_egoposer.yaml}"
CHECKPOINT_PATH="${EGOPOSER_CHECKPOINT:-$REPO_DIR/model_zoo/egoposer.pth}"
SMPLH_PATH="${EGOPOSER_SMPLH:-$REPO_DIR/support_data/body_models/smplh/male/model.npz}"
DMPL_PATH="${EGOPOSER_DMPL:-$REPO_DIR/support_data/body_models/dmpls/male/model.npz}"

if [[ -v EGOPOSER_RECORD ]]; then
    RECORD_PATH="$EGOPOSER_RECORD"
else
    RECORD_PATH="$REPO_DIR/recordings/visionpro_$(date +%Y%m%d_%H%M%S).npz"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $PYTHON_BIN" >&2
    echo "Activate the egoposer environment or set PYTHON_BIN=/path/to/python." >&2
    exit 1
fi

case "$DEVICE" in
    auto|cpu|cuda) ;;
    *)
        echo "Error: EGOPOSER_DEVICE must be auto, cpu, or cuda." >&2
        exit 1
        ;;
esac

for required_file in "$YAML_PATH" "$CHECKPOINT_PATH" "$SMPLH_PATH" "$DMPL_PATH"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Error: required file is missing: $required_file" >&2
        exit 1
    fi
done

record_args=()
if [[ -n "$RECORD_PATH" ]]; then
    mkdir -p -- "$(dirname -- "$RECORD_PATH")"
    record_args=(--record "$RECORD_PATH")
fi

echo "Starting EgoPoser realtime server"
echo "  Repository: $REPO_DIR"
echo "  WebSocket:  ws://$HOST:$PORT/ws"
echo "  Device:     $DEVICE"
echo "  Window:     $WINDOW_SIZE frames"
if [[ -n "$RECORD_PATH" ]]; then
    echo "  Recording:  $RECORD_PATH"
else
    echo "  Recording:  disabled"
fi
echo
echo "Configure EGO_UNITY with: ws://<server-LAN-IP>:$PORT/ws"

exec "$PYTHON_BIN" "$REPO_DIR/egoposer_server.py" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --window-size "$WINDOW_SIZE" \
    --reset-gap "$RESET_GAP" \
    --yaml "$YAML_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    "${record_args[@]}" \
    "$@"
