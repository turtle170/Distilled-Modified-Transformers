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

void dmt_staple_models(struct llama_model * small_model, struct llama_model * big_model) {
    if (!small_model || !big_model) return;
    
    std::cout << "-> DMT C-Bridge: Engaging Parameter-Level Expansion (PLE) Stapler...\n";
    size_t stapled_tensors = 0;
    size_t pruned_elements = 0;

    // We mathematically map the sparse activation space of the small_model onto the dense space of the big_model.
    for (auto & b_kv : big_model->tensors_by_name) {
        struct ggml_tensor * b_t = b_kv.second;
        
        // Find corresponding tensor in the small model mask
        auto s_kv = std::find_if(small_model->tensors_by_name.begin(), small_model->tensors_by_name.end(), 
            [&](const std::pair<std::string, struct ggml_tensor *>& val) { return val.first == b_kv.first; });
            
        if (s_kv != small_model->tensors_by_name.end()) {
            struct ggml_tensor * s_t = s_kv->second;
            
            if (b_t->type == GGML_TYPE_F32 && s_t->type == GGML_TYPE_F32) {
                float * b_data = (float *)b_t->data;
                float * s_data = (float *)s_t->data;
                
                int64_t b_ne0 = std::max<int64_t>(1, b_t->ne[0]);
                int64_t b_ne1 = std::max<int64_t>(1, b_t->ne[1]);
                int64_t b_ne2 = std::max<int64_t>(1, b_t->ne[2]);
                int64_t b_ne3 = std::max<int64_t>(1, b_t->ne[3]);

                int64_t s_ne0 = std::max<int64_t>(1, s_t->ne[0]);
                int64_t s_ne1 = std::max<int64_t>(1, s_t->ne[1]);
                int64_t s_ne2 = std::max<int64_t>(1, s_t->ne[2]);
                int64_t s_ne3 = std::max<int64_t>(1, s_t->ne[3]);

                for (int64_t i3 = 0; i3 < b_ne3; ++i3) {
                    int64_t s_i3 = (i3 * s_ne3) / b_ne3;
                    for (int64_t i2 = 0; i2 < b_ne2; ++i2) {
                        int64_t s_i2 = (i2 * s_ne2) / b_ne2;
                        for (int64_t i1 = 0; i1 < b_ne1; ++i1) {
                            int64_t s_i1 = (i1 * s_ne1) / b_ne1;
                            for (int64_t i0 = 0; i0 < b_ne0; ++i0) {
                                int64_t s_i0 = (i0 * s_ne0) / b_ne0;
                                
                                size_t b_idx = i3*(b_ne2*b_ne1*b_ne0) + i2*(b_ne1*b_ne0) + i1*b_ne0 + i0;
                                size_t s_idx = s_i3*(s_ne2*s_ne1*s_ne0) + s_i2*(s_ne1*s_ne0) + s_i1*s_ne0 + s_i0;
                                
                                // If the small model parameter is inactive, deactivate the big model parameter.
                                if (std::abs(s_data[s_idx]) < 1e-7f) {
                                    b_data[b_idx] = 0.0f;
                                    pruned_elements++;
                                }
                            }
                        }
                    }
                }
                stapled_tensors++;
            }
        }
    }
    
    std::cout << "-> DMT C-Bridge: Successfully stapled " << stapled_tensors << " tensor groups (Zeroed " << pruned_elements << " dead big-model parameters).\n";
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
