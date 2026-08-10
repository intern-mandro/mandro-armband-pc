"""
CLI smoke test for lib/ble_weights.py.

Companion to test_firmware_uploader.py, same step-by-step style. Lets you
validate the BLE weights-transfer pipeline before wiring it into the
Phase 3 UI, including a hardware-free check of the wire protocol itself.

Usage:
    # 1. No hardware, no model needed: verify build_frame()/parse_frame()
    #    round-trip and that a synthetic [132,64,64,6] payload matches the
    #    sizes exo_armband_hybrid.ino's NeuralNet expects.
    python test_ble_weights.py --frame-roundtrip

    # 2. No hardware needed: serialize a real trained model + scaler and
    #    report sizes (needs TensorFlow only).
    python test_ble_weights.py --serialize models/trained/model_KOTA_6cl.keras

    # 3. Full send over BLE (needs the bracelet powered on and paired).
    python test_ble_weights.py --full <address> \
        models/trained/model_KOTA_6cl.keras
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Make `lib/ble_weights.py` importable when running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import ble_weights as bw  # noqa: E402


# Firmware-side constants this test cross-checks against
# (firmware/esp32/exo_armband_hybrid_6clf/exo_armband_hybrid/nn.h /
#  preprocessor.h). If MODEL_TOPOLOGY ever changes, update these too.
TOPOLOGY = [132, 64, 64, 6]
NN_TOTAL_WEIGHTS = 12928
NN_TOTAL_BIASES = 134
N_FEATURES = 132


def print_progress(msg: str) -> None:
    print(f"  {msg}")


def cmd_frame_roundtrip() -> int:
    import numpy as np

    print("── Wire protocol round-trip (no hardware, no model needed) ──")

    rng = np.random.default_rng(0)
    chunks = []
    layer_sizes = []
    for in_d, out_d in zip(TOPOLOGY[:-1], TOPOLOGY[1:]):
        w = rng.standard_normal(in_d * out_d).astype("<f4")
        b = rng.standard_normal(out_d).astype("<f4")
        chunks.append(w)
        chunks.append(b)
        layer_sizes.append((in_d * out_d, out_d))
    means = rng.standard_normal(N_FEATURES).astype("<f4")
    stds = np.abs(rng.standard_normal(N_FEATURES)).astype("<f4") + 0.1
    chunks.append(means)
    chunks.append(stds)
    payload = np.concatenate(chunks).tobytes()

    total_w = sum(w for w, _ in layer_sizes)
    total_b = sum(b for _, b in layer_sizes)
    print(f"  Topology: {TOPOLOGY}")
    print(f"  Per-layer (weights, biases): {layer_sizes}")
    print(f"  Total weights: {total_w} (expect {NN_TOTAL_WEIGHTS})")
    print(f"  Total biases : {total_b} (expect {NN_TOTAL_BIASES})")
    assert total_w == NN_TOTAL_WEIGHTS, "weight count doesn't match firmware's NN_TOTAL_WEIGHTS"
    assert total_b == NN_TOTAL_BIASES, "bias count doesn't match firmware's NN_TOTAL_BIASES"

    expected_payload_bytes = (total_w + total_b + 2 * N_FEATURES) * 4
    print(f"  Payload size: {len(payload)} bytes (expect {expected_payload_bytes})")
    assert len(payload) == expected_payload_bytes

    frame = bw.build_frame(payload)
    expected_frame_bytes = 4 + 4 + len(payload) + 4
    print(f"  Frame size  : {len(frame)} bytes (expect {expected_frame_bytes})")
    assert len(frame) == expected_frame_bytes

    recovered = bw.parse_frame(frame)
    assert recovered == payload, "parse_frame() did not recover the original payload"
    print("  parse_frame(build_frame(payload)) == payload: OK")

    # Corruption must be caught, not silently accepted.
    corrupted = bytearray(frame)
    corrupted[20] ^= 0xFF  # flip a byte inside the payload region
    try:
        bw.parse_frame(bytes(corrupted))
    except bw.BleWeightsError as exc:
        print(f"  Corrupted frame correctly rejected: {exc}")
    else:
        print("  FAIL: corrupted frame was NOT rejected")
        return 1

    n_chunks = (len(frame) + bw.CHUNK_SIZE - 1) // bw.CHUNK_SIZE
    print(f"  Would send in {n_chunks} chunks of {bw.CHUNK_SIZE} bytes "
          f"(write-with-response)")

    print("\nAll checks passed.")
    return 0


def cmd_serialize(keras_path: Path) -> int:
    print(f"── Serializing {keras_path} (no hardware) ──")
    try:
        payload = bw.serialize_weights(keras_path)
    except bw.BleWeightsError as exc:
        print(f"\nERROR: {exc}")
        return 1
    frame = bw.build_frame(payload)
    print(f"  Payload: {len(payload)} bytes")
    print(f"  Frame  : {len(frame)} bytes (+8 header +4 crc)")
    n_chunks = (len(frame) + bw.CHUNK_SIZE - 1) // bw.CHUNK_SIZE
    print(f"  Chunks : {n_chunks} x {bw.CHUNK_SIZE} bytes")

    recovered = bw.parse_frame(frame)
    assert recovered == payload
    print("  Frame round-trips correctly.")
    return 0


def cmd_full(address: str, keras_path: Path) -> int:
    print(f"── Full BLE weights send: {keras_path} -> {address} ──")
    try:
        payload = bw.serialize_weights(keras_path)
        frame = bw.build_frame(payload)
        asyncio.run(bw.send_weights_ble(address, frame, on_progress=print_progress))
    except bw.BleWeightsError as exc:
        print(f"\nERROR: {exc}")
        return 1
    print("\nDone.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test lib/ble_weights.py")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--frame-roundtrip", action="store_true",
                   help="Verify the wire protocol with synthetic weights (no hardware/model)")
    g.add_argument("--serialize", metavar="KERAS_PATH",
                   help="Serialize a real model+scaler and report sizes (no hardware)")
    g.add_argument("--full", nargs=2, metavar=("ADDRESS", "KERAS_PATH"),
                   help="Send a real model to the bracelet over BLE")
    args = parser.parse_args()

    if args.frame_roundtrip:
        return cmd_frame_roundtrip()
    if args.serialize:
        return cmd_serialize(Path(args.serialize))
    if args.full:
        address, keras_path = args.full
        return cmd_full(address, Path(keras_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
