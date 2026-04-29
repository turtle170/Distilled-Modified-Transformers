# Distilled Modified Transformers (DMT)

DMT is a hyper-optimized, native, cross-platform CLI framework written in Zig and C++ for structurally distilling and pruning Large Language Models via evolutionary algorithms. It interfaces directly with the native C-API of `llama.cpp` to execute training loops and inference with **zero Python overhead**.

Features:
- Native execution on x86_64, AArch64, ARM (with NEON and AVX-512 VNNI support)
- Zero-copy exporter for `.safetensors`
- Model conversion for GGUF, SafeTensors, ONNX, EXL2, AWQ, PyTorch, and TensorFlow
- Auto-Quantization based on parameter scaling (Q4_K_M for >1B, Q8_0 for ~1B)
- Evaluator-Judge distillation Loop 

## Installation
A pre-built `.msi` Windows installer is provided via WiX (see `dmt-cli-installer.msi` under releases or build it yourself using `build_msi.ps1`).

Otherwise, build from source using Zig (0.16.0+):
```bash
# Clone the repo and deps
git clone https://github.com/turtle170/Distilled-Modified-Transformers.git
cd Distilled-Modified-Transformers
git clone https://github.com/ggerganov/llama.cpp deps/llama.cpp

# Compile
zig build -Drelease=true -Dnative=true
```

### Hardware Optimization Build Flags
When building from source, you can explicitly target specific CPU instruction sets for maximum tensor performance. Add these flags to your `zig build` command (e.g., `zig build -Drelease=true -Davx512=true -Davx512_vnni=true`).

* `-Dnative=true`: Automatically detect and compile for your host machine's optimal architecture (includes `-march=native -mtune=native`).
* `-Dsse3=true`: Enable SSE3 instructions.
* `-Davx=true`: Enable AVX instructions.
* `-Davx2=true`: Enable AVX2, FMA, and F16C instructions.
* `-Davx512=true`: Enable base AVX-512 instructions (F, BW, DQ, VL).
* `-Davx512_vnni=true`: Enable AVX-512 VNNI instructions (massively accelerates int8 quantized tensor math).
* `-Davx_vnni=true`: Enable AVX-VNNI (VNNI features without requiring 512-bit registers).
* `-Davx10=true`: Enable AVX10 (AVX10.1-256) instructions.

*Note: ARM architectures (AArch64 / ARM) natively apply `__ARM_NEON` optimizations automatically.*

## Usage

### 1. Evolutionary Training Mode (`train`)
Runs a multi-step loop where the `student` model processes a dataset, and the `judge` evaluates its performance. If performance drops, SIMD magnitude pruning triggers to organically drop the lowest impact weights and restructure the network.

```bash
dmt train --student mistral-7b.gguf \
          --judge gemma-4b.gguf \
          --dataset math_training_set.jsonl \
          --target-params 5.5 \
          --prune-rate 0.01 \
          --save-dir ./checkpoints \
          --out-format safetensors \
          --quant-type iq2_xxs
```

### 2. Pure Distillation Mode (`distill`)
Executes an immediate structural compression and target re-quantization *without* a dataset or judge. Best used to quickly squeeze models down using hardware-accelerated thresholds.

```bash
dmt distill --student llama-3-8b.gguf \
            --prune-rate 0.20 \
            --prune-method magnitude \
            --save-dir ./compressed \
            --out-format gguf \
            --quant-type q4_k_m
```

## License
MIT License - Copyright (c) 2026 turtle170.
