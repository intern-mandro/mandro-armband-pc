#include "nn.h"
#include "preprocessor.h"
#include <LittleFS.h>
#include <math.h>


NeuralNet::NeuralNet() {
    loaded_ = false;
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
    const float* w_ptr = weights_;
    const float* b_ptr = biases_;

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


// ═══════════════════════════════════════════════════════════════════════
// Load weights+biases+standardizer from LittleFS.
//
// File layout (written by the BLE weights-receive handler once CRC-checked):
//   [W0][b0][W1][b1]...[W_{n-1}][b_{n-1}][means][stds]   all float32
// i.e. weights and biases interleaved PER LAYER, matching how the PC app
// serializes a Keras model. predict() above instead expects weights_[]/
// biases_[] as two arrays each concatenated ACROSS layers, so this function
// reshuffles while reading.
// ═══════════════════════════════════════════════════════════════════════
bool NeuralNet::loadFromLittleFS(const char* path, Preprocessor& preproc) {
    loaded_ = false;

    File f = LittleFS.open(path, "r");
    if (!f) {
        Serial.printf("[nn] loadFromLittleFS: cannot open %s\n", path);
        return false;
    }

    size_t total_weights = 0, total_biases = 0;
    for (int L = 0; L < MODEL_N_LAYERS - 1; L++) {
        total_weights += (size_t)MODEL_TOPOLOGY[L] * (size_t)MODEL_TOPOLOGY[L + 1];
        total_biases  += (size_t)MODEL_TOPOLOGY[L + 1];
    }
    if (total_weights != (size_t)NN_TOTAL_WEIGHTS || total_biases != (size_t)NN_TOTAL_BIASES) {
        // MODEL_TOPOLOGY and the NN_TOTAL_* constants have drifted apart.
        Serial.println("[nn] loadFromLittleFS: topology/size constants mismatch");
        f.close();
        return false;
    }

    size_t expected_bytes = (total_weights + total_biases + 2 * (size_t)N_FEATURES) * sizeof(float);
    if ((size_t)f.size() != expected_bytes) {
        Serial.printf("[nn] loadFromLittleFS: size mismatch (file=%u expected=%u)\n",
                      (unsigned)f.size(), (unsigned)expected_bytes);
        f.close();
        return false;
    }

    float means[N_FEATURES];
    float stds[N_FEATURES];

    size_t w_off = 0, b_off = 0;
    for (int L = 0; L < MODEL_N_LAYERS - 1; L++) {
        int in_d  = MODEL_TOPOLOGY[L];
        int out_d = MODEL_TOPOLOGY[L + 1];
        size_t w_bytes = (size_t)in_d * (size_t)out_d * sizeof(float);
        size_t b_bytes = (size_t)out_d * sizeof(float);

        if (f.read((uint8_t*)(weights_ + w_off), w_bytes) != w_bytes) {
            Serial.println("[nn] loadFromLittleFS: short read (weights)");
            f.close();
            return false;
        }
        w_off += (size_t)in_d * (size_t)out_d;

        if (f.read((uint8_t*)(biases_ + b_off), b_bytes) != b_bytes) {
            Serial.println("[nn] loadFromLittleFS: short read (biases)");
            f.close();
            return false;
        }
        b_off += out_d;
    }

    if (f.read((uint8_t*)means, N_FEATURES * sizeof(float)) != N_FEATURES * sizeof(float)) {
        Serial.println("[nn] loadFromLittleFS: short read (means)");
        f.close();
        return false;
    }
    if (f.read((uint8_t*)stds, N_FEATURES * sizeof(float)) != N_FEATURES * sizeof(float)) {
        Serial.println("[nn] loadFromLittleFS: short read (stds)");
        f.close();
        return false;
    }

    f.close();

    preproc.setStandardizer(means, stds);
    loaded_ = true;
    Serial.println("[nn] loadFromLittleFS: model loaded OK");
    return true;
}
