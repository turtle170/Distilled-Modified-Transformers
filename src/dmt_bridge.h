#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;

// Executes the SIMD-accelerated magnitude pruning across all weight tensors
void dmt_prune_model_tensors(struct llama_model * model, float prune_rate);

// Executes a hyper-optimized Universal Sparse Autoencoder conceptually stapling teacher weights into student
void dmt_staple_models(struct llama_model * student, struct llama_model * teacher, bool linear_read);

// Executes a zero-copy export of all tensors into a SafeTensors format
void dmt_export_safetensors_impl(struct llama_model * model, const char * filename);

#ifdef __cplusplus
}
#endif
