#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;

// Executes the SIMD-accelerated magnitude pruning across all weight tensors
void dmt_prune_model_tensors(struct llama_model * model, float prune_rate);

// Executes a zero-copy export of all tensors into a SafeTensors format
void dmt_export_safetensors_impl(struct llama_model * model, const char * filename);

#ifdef __cplusplus
}
#endif
