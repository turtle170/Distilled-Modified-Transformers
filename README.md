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

### 3. Model Stapling Mode (`staple`)
Performs Parameter-Level Expansion (PLE) mathematically projecting the intelligence of a massive model directly into the sparse network architecture of a smaller model. Operates directly on the drive (zero RAM allocation for the teacher).

```bash
dmt staple --student llama-3-8b.gguf \
           --teacher llama-3-70b.gguf \
           --save-dir ./stapled \
           --out-format safetensors \
           --read-linear
```

### 4. Staple & Distill (`staple-distill`)
Combines `staple` and `distill` into one step. First interpolates the big model's weights into the small model, then immediately evaluates and surgically distills the stapled topology using the judge.

```bash
dmt staple-distill --student llama-3-8b.gguf \
                   --teacher llama-3-70b.gguf \
                   --judge gemma-4b.gguf \
                   --prune-rate 0.10 \
                   --q 5
```

### Exhaustive Parameter List
* `--student <PATH>`: Path to Student GGUF model.
* `--teacher <PATH>`: Path to massive Teacher GGUF model (for stapling).
* `--judge <PATH>`: Path to Judge GGUF model (for training / distillation evaluation).
* `--dataset <PATH>`: Path to .jsonl dataset (for training).
* `--read-linear`: (Staple Mode) Linearly scan Teacher directly from disk (default: true).
* `--read-random`: (Staple Mode) Randomly access Teacher memory, faster if huge RAM.
* `--threads <N>`: Number of threads for standard generation (default: 8).
* `--threads-batch <N>`: Number of threads for batch/prompt processing (default: 8).
* `--ngl-student <N>`: GPU layers for Student (default: 0).
* `--ngl-judge <N>`: GPU layers for Judge/Teacher (default: 0).
* `-q <N>` / `--quality <N>`: Distillation quality level [1-10] (default: 1). Higher = more refinement cycles.
* `--target-params <F>`: Target parameter size in Billions (default: 5.5). Auto-halts when reached.
* `--prune-rate <F>`: Percentage of weights to prune per drop (default: 0.01).
* `--prune-method <STR>`: Pruning method [magnitude, random] (default: magnitude).
* `--epochs <N>`: Number of training epochs (default: 1).
* `--save-dir <PATH>`: Directory to save checkpoints (default: out).
* `--out-format <FMT>`: Output format [gguf, safetensors] (default: gguf).
* `--quant-type <TYPE>`: Quantization type (e.g., f16, q4_k_m, q8_0, iq2_xxs) (default: q4_k_m).
* `--save-freq <N>`: Save model every N cycles (default: 100).
* `--seed <N>`: Random seed (default: 42).
* `--student-ctx <N>`: Context size for Student (default: 4096).
* `--student-batch-size <N>`: Batch size for Student (default: 512).
* `--student-temp <F>`: Temperature for Student generation (default: 0.7).
* `--student-top-k <N>`: Top-K for Student generation (default: 40).
* `--student-top-p <F>`: Top-P for Student generation (default: 0.95).
* `--judge-ctx <N>`: Context size for Judge (default: 8192).
* `--judge-temp <F>`: Temperature for Judge generation (default: 0.0).
* `--judge-top-k <N>`: Top-K for Judge generation (default: 40).
* `--judge-top-p <F>`: Top-P for Judge generation (default: 0.95).

## License
MIT License - Copyright (c) 2026 turtle170.
