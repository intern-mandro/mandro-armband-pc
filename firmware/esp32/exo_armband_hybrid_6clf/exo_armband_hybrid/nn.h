#ifndef NN_H
#define NN_H

#include <Arduino.h>
#include "MODEL.h"

// Activation codes (must match export_teensy_headers.py / lib/ble_weights.py)
#define ACT_RELU     0
#define ACT_SOFTMAX  1
#define ACT_SIGMOID  2
#define ACT_TANH     3
#define ACT_LINEAR   4

// Max layer width — sized for [132, 64, 64, 6], with margin.
#define NN_MAX_WIDTH 128

// Total weight/bias counts for MODEL_TOPOLOGY = [132, 64, 64, 6]:
//   132*64 + 64*64 + 64*6 = 8448 + 4096 + 384 = 12928 weights
//   64 + 64 + 6                              =  134 biases
// If MODEL_TOPOLOGY in MODEL.h ever changes, these must change too (and the
// firmware must be reflashed over USB — see the note in MODEL.h).
#define NN_TOTAL_WEIGHTS 12928
#define NN_TOTAL_BIASES  134

class Preprocessor;  // forward declaration; full type only needed in nn.cpp


class NeuralNet {
public:
    NeuralNet();

    // Forward pass. input must have MODEL_TOPOLOGY[0] elements,
    // output must have MODEL_TOPOLOGY[MODEL_N_LAYERS-1] elements.
    void predict(const float* input, float* output);

    // Return argmax index of a probability vector of length n.
    int  argmax(const float* output, int n);

    // Load weights+biases+standardizer from a LittleFS file written by the
    // BLE weights-receive path (see exo_armband_hybrid.ino). The file layout
    // is [W0 b0 W1 b1 ... means stds], float32, matching MODEL_TOPOLOGY.
    // Reshuffles that per-layer-interleaved layout into the two concatenated
    // arrays predict() expects (all weights, then all biases). Returns false
    // (and leaves isLoaded() == false) on any size mismatch or read error.
    bool loadFromLittleFS(const char* path, Preprocessor& preproc);

    // True once a model has been successfully loaded via loadFromLittleFS().
    bool isLoaded() const { return loaded_; }

    // Debug: number of features expected as input.
    int  inputSize()  const { return MODEL_TOPOLOGY[0]; }
    int  outputSize() const { return MODEL_TOPOLOGY[MODEL_N_LAYERS - 1]; }

private:
    // Two ping-pong buffers for layer outputs.
    float buf_a[NN_MAX_WIDTH];
    float buf_b[NN_MAX_WIDTH];

    // Weights/biases loaded at runtime (were compile-time MODEL_WEIGHTS/
    // MODEL_BIASES before the BLE weights-transfer change).
    float weights_[NN_TOTAL_WEIGHTS];
    float biases_[NN_TOTAL_BIASES];
    bool  loaded_ = false;

    // Apply activation in-place.
    void applyActivation(float* data, int n, int code);

    // Dense layer: out = activation(W^T * in + b)
    // weights are stored row-major with shape (in_dim, out_dim)
    //   weights[i * out_dim + j] = W[i, j]
    // matches Keras Dense layer weight format (kernel shape (in_dim, out_dim)).
    void denseLayer(const float* in, int in_dim,
                    float* out, int out_dim,
                    const float* weights, const float* biases,
                    int activation_code);
};

#endif // NN_H
