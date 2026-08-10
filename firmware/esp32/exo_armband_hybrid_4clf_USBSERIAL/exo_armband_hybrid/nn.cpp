#include "nn.h"
#include <math.h>


NeuralNet::NeuralNet() {
    // Nothing to init; MODEL_* are const globals from MODEL.h
}


// ═══════════════════════════════════════════════════════════════════════
// Activation functions
// ═══════════════════════════════════════════════════════════════════════
void NeuralNet::applyActivation(float* data, int n, int code) {
    switch (code) {
        case ACT_RELU:
            for (int i = 0; i < n; i++) {
                if (data[i] < 0.0f) data[i] = 0.0f;
            }
            break;

        case ACT_SOFTMAX: {
            // Numerically stable softmax: subtract max before exp
            float max_val = data[0];
            for (int i = 1; i < n; i++) {
                if (data[i] > max_val) max_val = data[i];
            }
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                data[i] = expf(data[i] - max_val);
                sum += data[i];
            }
            float inv_sum = 1.0f / sum;
            for (int i = 0; i < n; i++) data[i] *= inv_sum;
            break;
        }

        case ACT_SIGMOID:
            for (int i = 0; i < n; i++) {
                data[i] = 1.0f / (1.0f + expf(-data[i]));
            }
            break;

        case ACT_TANH:
            for (int i = 0; i < n; i++) data[i] = tanhf(data[i]);
            break;

        case ACT_LINEAR:
        default:
            // No-op
            break;
    }
}


// ═══════════════════════════════════════════════════════════════════════
// Dense layer: out[j] = activation( sum_i(in[i] * W[i,j]) + b[j] )
// weights layout: row-major (in_dim, out_dim), matching Keras flatten()
// ═══════════════════════════════════════════════════════════════════════
void NeuralNet::denseLayer(const float* in, int in_dim,
                           float* out, int out_dim,
                           const float* weights, const float* biases,
                           int activation_code) {
    for (int j = 0; j < out_dim; j++) {
        float acc = biases[j];
        for (int i = 0; i < in_dim; i++) {
            acc += in[i] * weights[i * out_dim + j];
        }
        out[j] = acc;
    }
    applyActivation(out, out_dim, activation_code);
}


// ═══════════════════════════════════════════════════════════════════════
// Full forward pass
// ═══════════════════════════════════════════════════════════════════════
void NeuralNet::predict(const float* input, float* output) {
    // Copy input to buf_a
    int in_dim = MODEL_TOPOLOGY[0];
    for (int i = 0; i < in_dim; i++) buf_a[i] = input[i];

    // Ping-pong through hidden layers
    const float* w_ptr = MODEL_WEIGHTS;
    const float* b_ptr = MODEL_BIASES;

    float* cur_in  = buf_a;
    float* cur_out = buf_b;

    for (int L = 0; L < MODEL_N_LAYERS - 1; L++) {
        int in_d  = MODEL_TOPOLOGY[L];
        int out_d = MODEL_TOPOLOGY[L + 1];
        int act   = MODEL_ACTIVATIONS[L];

        denseLayer(cur_in, in_d, cur_out, out_d, w_ptr, b_ptr, act);

        // Advance pointers
        w_ptr += in_d * out_d;
        b_ptr += out_d;

        // Swap buffers
        float* tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
    }

    // Final output is in cur_in (after last swap)
    int out_dim = MODEL_TOPOLOGY[MODEL_N_LAYERS - 1];
    for (int i = 0; i < out_dim; i++) output[i] = cur_in[i];
}


// ═══════════════════════════════════════════════════════════════════════
// Argmax
// ═══════════════════════════════════════════════════════════════════════
int NeuralNet::argmax(const float* output, int n) {
    int best = 0;
    float best_val = output[0];
    for (int i = 1; i < n; i++) {
        if (output[i] > best_val) {
            best_val = output[i];
            best = i;
        }
    }
    return best;
}