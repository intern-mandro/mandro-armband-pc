"""
Export a Keras model + its scaler as MODEL.h / means.h / stds.h.

Usage:
    # Default (backwards compatible: uses model_causal_concat132)
    python scripts/export_teensy_headers.py

    # Explicit model
    python scripts/export_teensy_headers.py \
        --keras models/trained/model_KOTA_6cl_20260529_morning.keras

    # Custom scaler path (default: auto-deduced from model name)
    python scripts/export_teensy_headers.py \
        --keras models/trained/model_KOTA_6cl.keras \
        --scaler models/scalers/scaler_KOTA_6cl.pkl

    # Custom output directory (default: firmware/teensy)
    python scripts/export_teensy_headers.py \
        --keras models/trained/model_KOTA_6cl.keras \
        --output-dir firmware/esp32/exo_armband_hybrid_6clf/exo_armband_hybrid
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from lib.pipeline import load_pipeline


# ── Defaults (backwards compatible) ─────────────────────────────────
DEFAULT_MODEL_PATH = "models/trained/model_causal_concat132.keras"
DEFAULT_OUTPUT_DIR = "firmware/teensy"
DEFAULT_SCALER_FOLDER = "models/scalers"


def deduce_scaler_path(model_path: Path,
                       scaler_folder: str = DEFAULT_SCALER_FOLDER) -> Path:
    """Given models/trained/model_FOO.keras, return models/scalers/scaler_FOO.pkl"""
    stem = model_path.stem
    if stem.startswith("model_"):
        scaler_name = "scaler_" + stem[len("model_"):]
    else:
        scaler_name = "scaler_" + stem
    return Path(scaler_folder) / f"{scaler_name}.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Keras model + scaler as C++ headers.")
    parser.add_argument(
        "--keras", default=DEFAULT_MODEL_PATH,
        help=f"Path to the .keras model (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument(
        "--scaler", default=None,
        help="Path to the scaler .pkl (default: auto-deduced from model name)")
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write MODEL.h/means.h/stds.h (default: {DEFAULT_OUTPUT_DIR})")
    return parser.parse_args()


def get_activation_code(layer):
    name = layer.activation.__name__.lower()
    return {'sigmoid': 'SIGMOID', 'relu': 'RELU', 'tanh': 'TANH',
            'linear': 'LINEAR', 'softmax': 'SOFTMAX'}.get(name, 'UNKNOWN')


def fmt_float_array(values, per_line=8):
    """Format a flat list of floats as a C++ initializer list."""
    def fmt_one(v):
        s = f"{v:.10g}"
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s + "f"
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i+per_line]
        lines.append("    " + ", ".join(fmt_one(v) for v in chunk))
    return ",\n".join(lines)


def fmt_int_array(values):
    return ", ".join(str(int(v)) for v in values)


def main():
    args = parse_args()
    model_path = Path(args.keras)

    if args.scaler is not None:
        scaler_path = Path(args.scaler)
    else:
        scaler_path = deduce_scaler_path(model_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)
    if not scaler_path.exists():
        print(f"ERROR: scaler not found: {scaler_path}")
        sys.exit(1)

    print(f"── Loading model:  {model_path}")
    model = tf.keras.models.load_model(model_path)

    # The load_pipeline helper takes a name (no .pkl) and a folder
    scaler_folder = str(scaler_path.parent)
    scaler_name   = scaler_path.stem  # "scaler_KOTA_6cl..."
    print(f"── Loading scaler: {scaler_path}")
    scaler = load_pipeline(scaler_name, folder=scaler_folder)['scaler']

    topology    = [model.input_shape[-1]]
    weights     = []
    biases      = []
    activations = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            topology.append(layer.units)
            w, b = layer.get_weights()
            weights.extend(w.flatten())
            biases.extend(b.flatten())
            activations.append(get_activation_code(layer))

    n_params = len(weights) + len(biases)
    print(f"\n── Model summary")
    print(f"   Topology    : {topology}")
    print(f"   Activations : {activations}")
    print(f"   Weights     : {len(weights)}")
    print(f"   Biases      : {len(biases)}")
    print(f"   Total params: {n_params}")

    model_h = output_dir / "MODEL.h"
    with open(model_h, 'w') as f:
        f.write("#ifndef MODEL_H\n#define MODEL_H\n\n")
        f.write("// Auto-generated. Do not edit by hand.\n")
        f.write(f"// Source model: {model_path.name}\n")
        f.write(f"// Topology: {topology}\n")
        f.write(f"// Activations: {activations}\n\n")

        f.write(f"const int MODEL_N_LAYERS = {len(topology)};\n\n")

        f.write(f"const int MODEL_TOPOLOGY[{len(topology)}] = {{\n")
        f.write(f"    {fmt_int_array(topology)}\n")
        f.write("};\n\n")

        f.write(f"const float MODEL_WEIGHTS[{len(weights)}] = {{\n")
        f.write(fmt_float_array(weights))
        f.write("\n};\n\n")

        f.write(f"const float MODEL_BIASES[{len(biases)}] = {{\n")
        f.write(fmt_float_array(biases))
        f.write("\n};\n\n")

        act_map = {'RELU': 0, 'SOFTMAX': 1, 'SIGMOID': 2, 'TANH': 3, 'LINEAR': 4}
        act_ints = [act_map[a] for a in activations]
        f.write("// Activation codes: 0=RELU, 1=SOFTMAX, 2=SIGMOID, 3=TANH, 4=LINEAR\n")
        f.write(f"const int MODEL_ACTIVATIONS[{len(act_ints)}] = {{\n")
        f.write(f"    {fmt_int_array(act_ints)}\n")
        f.write("};\n\n")

        f.write("#endif // MODEL_H\n")

    print(f"\n\u2705 MODEL.h written ({model_h.stat().st_size / 1024:.1f} KB)")

    means = scaler.mean_
    stds  = scaler.scale_

    print(f"\n── Scaler stats")
    print(f"   n_features : {len(means)}")
    print(f"   means range: [{means.min():.4f}, {means.max():.4f}]")
    print(f"   stds  range: [{stds.min():.4f}, {stds.max():.4f}]")

    assert len(means) == 132, f"Expected 132 features, got {len(means)}"

    means_h = output_dir / "means.h"
    with open(means_h, 'w') as f:
        f.write("#ifndef MEANS_H\n#define MEANS_H\n\n")
        f.write("// Auto-generated. Do not edit by hand.\n")
        f.write(f"// Source scaler: {scaler_path.name}\n\n")
        f.write(f"const float STANDARDIZER_MEANS[{len(means)}] = {{\n")
        f.write(fmt_float_array(means.tolist()))
        f.write("\n};\n\n")
        f.write("#endif // MEANS_H\n")

    stds_h = output_dir / "stds.h"
    with open(stds_h, 'w') as f:
        f.write("#ifndef STDS_H\n#define STDS_H\n\n")
        f.write("// Auto-generated. Do not edit by hand.\n")
        f.write(f"// Source scaler: {scaler_path.name}\n\n")
        f.write(f"const float STANDARDIZER_STDS[{len(stds)}] = {{\n")
        f.write(fmt_float_array(stds.tolist()))
        f.write("\n};\n\n")
        f.write("#endif // STDS_H\n")

    print(f"\n\u2705 means.h written ({means_h.stat().st_size / 1024:.1f} KB)")
    print(f"\u2705 stds.h  written ({stds_h.stat().st_size / 1024:.1f} KB)")
    print(f"\nDone. Files in {output_dir}/")


if __name__ == "__main__":
    main()
