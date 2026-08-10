"""
export.py
=========
Model saving and loading utilities (pkl, joblib, ONNX, TFLite, C++ header).
"""

import os
import pickle as pkl
import numpy as np
import joblib as jb


# ─────────────────────────────────────────────────────────────────────────────
# Pickle / Joblib
# ─────────────────────────────────────────────────────────────────────────────

def save_my_model(
    model,
    model_name: str,
    folder_name: str = "models/trained",
    save_type: str = 'pkl',
) -> None:
    model_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}.{save_type}")
    with open(path, 'wb') as f:
        if save_type == 'pkl':
            pkl.dump(model, f)
        elif save_type == 'joblib':
            jb.dump(model, f)
        else:
            raise ValueError(f"Unknown save_type: '{save_type}'")
    print(f"Saved → {path}")


def load_my_model(
    model_name: str,
    folder_name: str = "models/trained",
    save_type: str = 'pkl',
):
    model_dir = os.path.join(os.getcwd(), folder_name)
    path = os.path.join(model_dir, f"{model_name}.{save_type}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'rb') as f:
        if save_type == 'pkl':
            return pkl.load(f)
        elif save_type == 'joblib':
            return jb.load(f)
    raise ValueError(f"Unknown save_type: '{save_type}'")


# ─────────────────────────────────────────────────────────────────────────────
# ONNX
# ─────────────────────────────────────────────────────────────────────────────

def save_as_onnx(
    model,
    model_name: str,
    input_shape: tuple,
    folder_name: str = "models/exported"
) -> None:
    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError:
        raise ImportError("Install tf2onnx: pip install tf2onnx")
    os.makedirs(folder_name, exist_ok=True)
    path = os.path.join(folder_name, f"{model_name}.onnx")
    spec = (tf.TensorSpec((None, *input_shape), tf.float32, name="input"),)
    _, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=path)
    print(f"ONNX saved → {path}")


def load_onnx(
    model_name: str,
    folder_name: str = "models/exported"
):
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Install onnxruntime: pip install onnxruntime")
    path = os.path.join(folder_name, f"{model_name}.onnx")
    return ort.InferenceSession(path)


# ─────────────────────────────────────────────────────────────────────────────
# C++ header for microcontrollers (Teensy 4.0 compatible)
# ─────────────────────────────────────────────────────────────────────────────

def save_nn_model_header(
    model,
    model_name: str = "MODEL",
    folder_name: str = "firmware/teensy"
) -> None:
    """
    Generate a C++ .h file containing neural network weights
    compatible with nn.h / nn.cpp on Teensy 4.0.

    Fixes vs original:
    - float instead of double
    - MODEL_N_LAYERS instead of MODEL_LAYERS
    - activation codes as integers (0=RELU, 1=SOFTMAX, etc.)
    - no PROGMEM (Teensy 4.0 has enough RAM)
    - no avr/pgmspace.h include
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required.")

    # Activation name → integer code matching nn.h defines
    def _get_activation_code(layer):
        name = layer.activation.__name__.lower()
        return {
            'relu':    0,
            'softmax': 1,
            'sigmoid': 2,
            'tanh':    3,
            'linear':  4,
        }.get(name, 4)

    # Format a list of floats as C float literals with f suffix
    def _fmt_floats(numbers):
        vals = ",\n    ".join(f"{float(x):.8g}f" for x in numbers)
        return "{\n    " + vals + "\n}"

    # Format a list of ints as C int literals
    def _fmt_ints(numbers):
        return "{" + ", ".join(str(int(x)) for x in numbers) + "}"

    topology = [model.input_shape[-1]]
    weights = []
    biases = []
    activations = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            topology.append(layer.units)
            w, b = layer.get_weights()
            weights.extend(w.flatten())
            biases.extend(b.flatten())
            activations.append(_get_activation_code(layer))

    os.makedirs(folder_name, exist_ok=True)

    mname = "MODEL"  # always MODEL so nn.h finds it
    path = os.path.join(folder_name, f"{mname}.h")

    n_layers = len(topology)

    with open(path, 'w') as f:
        f.write(f"#ifndef {mname}_H\n")
        f.write(f"#define {mname}_H\n\n")
        f.write(f"// Auto-generated. Do not edit by hand.\n")
        f.write(f"// Topology: {topology}\n")
        f.write(f"// Activations: {['RELU' if a==0 else 'SOFTMAX' if a==1 else str(a) for a in activations]}\n\n")

        f.write(f"const int MODEL_N_LAYERS = {n_layers};\n\n")

        f.write(f"const int MODEL_TOPOLOGY[{n_layers}] = {_fmt_ints(topology)};\n\n")

        f.write(f"const int MODEL_ACTIVATIONS[{len(activations)}] = {_fmt_ints(activations)};\n\n")

        f.write(f"const float MODEL_WEIGHTS[{len(weights)}] = {_fmt_floats(weights)};\n\n")

        f.write(f"const float MODEL_BIASES[{len(biases)}] = {_fmt_floats(biases)};\n\n")

        f.write(f"#endif // {mname}_H\n")

    print(f"C++ header saved → {path}")
    print(f"  Topology   : {topology}")
    print(f"  Activations: {activations}")
    print(f"  Weights    : {len(weights)}")
    print(f"  Biases     : {len(biases)}")


def array_to_C(
    array: np.ndarray,
    name_file: str,
    folder_name: str = "firmware/teensy"
) -> None:
    """
    Export a NumPy array as a C++ header file (float[]).
    """
    os.makedirs(folder_name, exist_ok=True)
    upper = name_file.upper()
    path = os.path.join(folder_name, f"{name_file}.h")
    with open(path, 'w') as f:
        f.write(f"#ifndef {upper}_H\n#define {upper}_H\n\n")
        f.write("// Auto-generated. Do not edit by hand.\n\n")
        f.write(f"const float STANDARDIZER_{upper}[{len(array)}] = {{\n")
        # 8 values per line
        for i in range(0, len(array), 8):
            chunk = array[i:i+8]
            comma = ',' if i + 8 < len(array) else ''
            line = ", ".join(f"{v:.8g}f" for v in chunk)
            f.write(f"    {line}{comma}\n")
        f.write("};\n\n")
        f.write(f"#endif // {upper}_H\n")
    print(f"C header saved → {path}")


def check_sizeof(model) -> int:
    """Return the approximate number of parameters in a Keras model."""
    return int(sum(w.numpy().size for w in model.weights))


# ─────────────────────────────────────────────────────────────────────────────
# TFLite
# ─────────────────────────────────────────────────────────────────────────────

def export_tflite(
    model,
    model_name: str,
    folder_name: str = "models/exported"
) -> None:
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required.")
    os.makedirs(folder_name, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path = os.path.join(folder_name, f"{model_name}.tflite")
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite saved → {path}  ({len(tflite_model)/1024:.1f} KB)")


def quantize_model(
    model,
    model_name: str,
    folder_name: str = "models/exported"
) -> None:
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required.")
    os.makedirs(folder_name, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    path = os.path.join(folder_name, f"{model_name}_quantized.tflite")
    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"Quantized TFLite saved → {path}  ({len(tflite_model)/1024:.1f} KB)")
