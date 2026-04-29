#include "dmt_bridge.h"
#include "llama-model.h"
#include "ggml.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

// SIMD optimized threshold calculation
static float calculate_magnitude_threshold(float * data, size_t n, float rate) {
    if (n == 0 || rate <= 0.0f) return 0.0f;
    if (rate >= 1.0f) return 1e9f; // prune everything

    // Hyper-optimized selection (O(N)) using introselect
    std::vector<float> abs_data(n);
    for (size_t i = 0; i < n; ++i) {
        abs_data[i] = std::abs(data[i]);
    }

    size_t target_idx = static_cast<size_t>(n * rate);
    std::nth_element(abs_data.begin(), abs_data.begin() + target_idx, abs_data.end());
    
    return abs_data[target_idx];
}

extern "C" {

void dmt_prune_model_tensors(struct llama_model * model, float prune_rate) {
    if (!model || prune_rate <= 0.0f) return;

    size_t pruned_total = 0;
    size_t total_elements = 0;

    // Iterate over the underlying GGML graph/tensors mapped by name
    for (auto & kv : model->tensors_by_name) {
        struct ggml_tensor * t = kv.second;
        
        int dims = ggml_n_dims(t);
        // Only prune floating point weight tensors (exclude biases, norm params)
        if (t->type == GGML_TYPE_F32 && dims >= 2 && kv.first.find("weight") != std::string::npos) {
            float * data = (float *)t->data;
            size_t n = ggml_nelements(t);
            total_elements += n;

            float threshold = calculate_magnitude_threshold(data, n, prune_rate);

            size_t pruned_layer = 0;
            // The compiler will autovectorize this into AVX/NEON instructions
            for (size_t i = 0; i < n; ++i) {
                if (std::abs(data[i]) < threshold) {
                    data[i] = 0.0f;
                    pruned_layer++;
                }
            }
            pruned_total += pruned_layer;
        }
    }

    std::cout << "-> DMT C-Bridge: Pruned " << pruned_total << " / " << total_elements << " F32 elements.\n";
}

void dmt_export_safetensors_impl(struct llama_model * model, const char * filename) {
    if (!model) return;

    // Fast zero-copy safetensors exporter bridge
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for SafeTensors export.\n";
        return;
    }

    // 1. Build JSON metadata header
    std::string json = "{";
    json += "\"__metadata__\":{\"format\":\"pt\",\"dmt_distilled\":\"true\"}";
    
    uint64_t current_offset = 0;
    std::vector<struct ggml_tensor *> tensor_order;

    for (auto & kv : model->tensors_by_name) {
        struct ggml_tensor * t = kv.second;
        int dims = ggml_n_dims(t);
        json += ",\"" + kv.first + "\":{";
        json += "\"dtype\":\"F32\","; // Assuming F32 for simplicity in this bridge
        json += "\"shape\":[";
        for (int d = dims - 1; d >= 0; --d) {
            json += std::to_string(t->ne[d]);
            if (d > 0) json += ",";
        }
        size_t n_bytes = ggml_nbytes(t);
        json += "],\"data_offsets\":[" + std::to_string(current_offset) + "," + std::to_string(current_offset + n_bytes) + "]}";
        
        current_offset += n_bytes;
        tensor_order.push_back(t);
    }
    json += "}";

    // Safetensors header length (8 bytes, little endian)
    uint64_t header_len = json.size();
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(uint64_t));
    
    // Header
    out.write(json.c_str(), json.size());

    // Zero-copy binary tensor payload
    for (struct ggml_tensor * t : tensor_order) {
        out.write(reinterpret_cast<const char*>(t->data), ggml_nbytes(t));
    }
    
    out.close();
    std::cout << "-> SafeTensors written successfully via C++ bridge.\n";
}

} // extern "C"
