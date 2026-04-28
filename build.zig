const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    const exe = b.addExecutable(.{
        .name = "dmt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
        }),
    });

    // --- Modern libllama & libggml Build Integration ---
    // Assuming llama.cpp source is cloned to 'deps/llama.cpp'
    const llama_dir = "deps/llama.cpp";
    
    const libllama = b.addLibrary(.{
        .linkage = .static,
        .name = "llama",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
        }),
    });

    const use_native = b.option(bool, "native", "Optimize for native CPU architecture (adds -march=native -mtune=native)") orelse false;

    const target_arch = target.result.cpu.arch;
    const target_os = target.result.os.tag;

    // Base flags
    var c_flags_list: std.ArrayList([]const u8) = .empty;
    defer c_flags_list.deinit(b.allocator);
    c_flags_list.appendSlice(b.allocator, &[_][]const u8{
        "-std=c11", "-fPIC", "-O3", "-Wall", "-Wextra",
        "-DGGML_USE_K_QUANTS", "-D_GNU_SOURCE",
        "-DGGML_VERSION=\"unknown\"", "-DGGML_COMMIT=\"unknown\"",
    }) catch @panic("OOM");

    var cpp_flags_list: std.ArrayList([]const u8) = .empty;
    defer cpp_flags_list.deinit(b.allocator);
    cpp_flags_list.appendSlice(b.allocator, &[_][]const u8{
        "-std=c++17", "-fPIC", "-O3", "-Wall", "-Wextra",
        "-DGGML_USE_K_QUANTS", "-D_GNU_SOURCE",
        "-DGGML_VERSION=\"unknown\"", "-DGGML_COMMIT=\"unknown\"",
    }) catch @panic("OOM");

    // Architecture specific optimizations
    if (target_arch == .x86_64 or target_arch == .x86) {
        c_flags_list.appendSlice(b.allocator, &[_][]const u8{
            "-msse3", "-mssse3", "-mcx16",
            "-mavx", "-mavx2", "-mfma", "-mf16c",
        }) catch @panic("OOM");
        cpp_flags_list.appendSlice(b.allocator, &[_][]const u8{
            "-msse3", "-mssse3", "-mcx16",
            "-mavx", "-mavx2", "-mfma", "-mf16c",
        }) catch @panic("OOM");
    } else if (target_arch == .aarch64 or target_arch == .aarch64_be) {
        // ARM NEON is enabled by default on AArch64, but we ensure the preprocessor macros are ready.
        c_flags_list.appendSlice(b.allocator, &[_][]const u8{ "-D__ARM_NEON" }) catch @panic("OOM");
        cpp_flags_list.appendSlice(b.allocator, &[_][]const u8{ "-D__ARM_NEON" }) catch @panic("OOM");
    } else if (target_arch == .arm) {
        c_flags_list.appendSlice(b.allocator, &[_][]const u8{ "-mfpu=neon", "-D__ARM_NEON" }) catch @panic("OOM");
        cpp_flags_list.appendSlice(b.allocator, &[_][]const u8{ "-mfpu=neon", "-D__ARM_NEON" }) catch @panic("OOM");
    }

    if (use_native) {
        c_flags_list.appendSlice(b.allocator, &[_][]const u8{ "-march=native", "-mtune=native" }) catch @panic("OOM");
        cpp_flags_list.appendSlice(b.allocator, &[_][]const u8{ "-march=native", "-mtune=native" }) catch @panic("OOM");
    }

    if (target_os == .linux) {
        libllama.root_module.linkSystemLibrary("pthread", .{});
        libllama.root_module.linkSystemLibrary("m", .{});
        exe.root_module.linkSystemLibrary("pthread", .{});
        exe.root_module.linkSystemLibrary("m", .{});
    }

    const c_flags = c_flags_list.items;
    const cpp_flags = cpp_flags_list.items;

    // 1. Compile GGML core (modern llama.cpp structure)
    libllama.root_module.addCSourceFiles(.{
        .root = b.path(llama_dir),
        .files = &.{
            "ggml/src/ggml.c",
            "ggml/src/ggml-alloc.c",
            "ggml/src/ggml-quants.c",
            "ggml/src/ggml-cpu/ggml-cpu.c",
            "ggml/src/ggml-cpu/quants.c",
        },
        .flags = c_flags,
    });

    var cpp_files_list: std.ArrayList([]const u8) = .empty;
    defer cpp_files_list.deinit(b.allocator);
    
    // Core C++ files
    cpp_files_list.appendSlice(b.allocator, &.{
        "ggml/src/ggml-backend.cpp",
        "ggml/src/ggml-backend-dl.cpp",
        "ggml/src/ggml-backend-meta.cpp",
        "ggml/src/ggml-backend-reg.cpp",
        "ggml/src/ggml-opt.cpp",
        "ggml/src/ggml-threading.cpp",
        "ggml/src/ggml.cpp",
        "ggml/src/gguf.cpp",
        "ggml/src/ggml-cpu/ggml-cpu.cpp",
        "ggml/src/ggml-cpu/ops.cpp",
        "ggml/src/ggml-cpu/repack.cpp",
        "ggml/src/ggml-cpu/binary-ops.cpp",
        "ggml/src/ggml-cpu/unary-ops.cpp",
        "ggml/src/ggml-cpu/vec.cpp",
        "ggml/src/ggml-cpu/hbm.cpp",
        "ggml/src/ggml-cpu/traits.cpp",
        "src/llama-adapter.cpp",
        "src/llama-arch.cpp",
        "src/llama-batch.cpp",
        "src/llama-chat.cpp",
        "src/llama-context.cpp",
        "src/llama-cparams.cpp",
        "src/llama-grammar.cpp",
        "src/llama-graph.cpp",
        "src/llama-hparams.cpp",
        "src/llama-impl.cpp",
        "src/llama-io.cpp",
        "src/llama-kv-cache-iswa.cpp",
        "src/llama-kv-cache.cpp",
        "src/llama-memory-hybrid-iswa.cpp",
        "src/llama-memory-hybrid.cpp",
        "src/llama-memory-recurrent.cpp",
        "src/llama-memory.cpp",
        "src/llama-mmap.cpp",
        "src/llama-model-loader.cpp",
        "src/llama-model-saver.cpp",
        "src/llama-model.cpp",
        "src/llama-quant.cpp",
        "src/llama-sampler.cpp",
        "src/llama-vocab.cpp",
        "src/llama.cpp",
        "src/unicode-data.cpp",
        "src/unicode.cpp",
    }) catch @panic("OOM");

    cpp_files_list.appendSlice(b.allocator, &.{
        "src/models/afmoe.cpp", "src/models/apertus.cpp", "src/models/arcee.cpp",
        "src/models/arctic.cpp", "src/models/arwkv7.cpp", "src/models/baichuan.cpp",
        "src/models/bailingmoe.cpp", "src/models/bailingmoe2.cpp", "src/models/bert.cpp",
        "src/models/bitnet.cpp", "src/models/bloom.cpp", "src/models/chameleon.cpp",
        "src/models/chatglm.cpp", "src/models/codeshell.cpp", "src/models/cogvlm.cpp",
        "src/models/cohere2-iswa.cpp", "src/models/command-r.cpp", "src/models/dbrx.cpp",
        "src/models/deci.cpp", "src/models/deepseek.cpp", "src/models/deepseek2.cpp",
        "src/models/delta-net-base.cpp", "src/models/dots1.cpp", "src/models/dream.cpp",
        "src/models/ernie4-5-moe.cpp", "src/models/ernie4-5.cpp", "src/models/eurobert.cpp",
        "src/models/exaone-moe.cpp", "src/models/exaone.cpp", "src/models/exaone4.cpp",
        "src/models/falcon-h1.cpp", "src/models/falcon.cpp", "src/models/gemma-embedding.cpp",
        "src/models/gemma.cpp", "src/models/gemma2-iswa.cpp", "src/models/gemma3.cpp",
        "src/models/gemma3n-iswa.cpp", "src/models/gemma4-iswa.cpp", "src/models/glm4-moe.cpp",
        "src/models/glm4.cpp", "src/models/gpt2.cpp", "src/models/gptneox.cpp",
        "src/models/granite-hybrid.cpp", "src/models/granite.cpp", "src/models/grok.cpp",
        "src/models/grovemoe.cpp", "src/models/hunyuan-dense.cpp", "src/models/hunyuan-moe.cpp",
        "src/models/internlm2.cpp", "src/models/jais.cpp", "src/models/jais2.cpp",
        "src/models/jamba.cpp", "src/models/kimi-linear.cpp", "src/models/lfm2.cpp",
        "src/models/llada-moe.cpp", "src/models/llada.cpp", "src/models/llama.cpp",
        "src/models/llama4.cpp", "src/models/maincoder.cpp", "src/models/mamba-base.cpp",
        "src/models/mamba.cpp", "src/models/mimo2-iswa.cpp", "src/models/minicpm3.cpp",
        "src/models/minimax-m2.cpp", "src/models/mistral3.cpp", "src/models/modern-bert.cpp",
        "src/models/mpt.cpp", "src/models/nemotron-h.cpp", "src/models/nemotron.cpp",
        "src/models/neo-bert.cpp", "src/models/olmo.cpp", "src/models/olmo2.cpp",
        "src/models/olmoe.cpp", "src/models/openai-moe-iswa.cpp", "src/models/openelm.cpp",
        "src/models/orion.cpp", "src/models/paddleocr.cpp", "src/models/pangu-embedded.cpp",
        "src/models/phi2.cpp", "src/models/phi3.cpp", "src/models/plamo.cpp",
        "src/models/plamo2.cpp", "src/models/plamo3.cpp", "src/models/plm.cpp",
        "src/models/qwen.cpp", "src/models/qwen2.cpp", "src/models/qwen2moe.cpp",
        "src/models/qwen2vl.cpp", "src/models/qwen3.cpp", "src/models/qwen35.cpp",
        "src/models/qwen35moe.cpp", "src/models/qwen3moe.cpp", "src/models/qwen3next.cpp",
        "src/models/qwen3vl-moe.cpp", "src/models/qwen3vl.cpp", "src/models/refact.cpp",
        "src/models/rnd1.cpp", "src/models/rwkv6-base.cpp", "src/models/rwkv6.cpp",
        "src/models/rwkv6qwen2.cpp", "src/models/rwkv7-base.cpp", "src/models/rwkv7.cpp",
        "src/models/seed-oss.cpp", "src/models/smallthinker.cpp", "src/models/smollm3.cpp",
        "src/models/stablelm.cpp", "src/models/starcoder.cpp", "src/models/starcoder2.cpp",
        "src/models/step35-iswa.cpp", "src/models/t5.cpp", "src/models/t5encoder.cpp",
        "src/models/wavtokenizer-dec.cpp", "src/models/xverse.cpp"
    }) catch @panic("OOM");

    libllama.root_module.addCSourceFiles(.{
        .root = b.path(llama_dir),
        .files = cpp_files_list.items,
        .flags = cpp_flags,
    });

    // Include paths
    libllama.root_module.addIncludePath(b.path(std.fmt.comptimePrint("{s}/include", .{llama_dir})));
    libllama.root_module.addIncludePath(b.path(std.fmt.comptimePrint("{s}/ggml/include", .{llama_dir})));
    libllama.root_module.addIncludePath(b.path(std.fmt.comptimePrint("{s}/ggml/src", .{llama_dir})));
    libllama.root_module.addIncludePath(b.path(std.fmt.comptimePrint("{s}/ggml/src/ggml-cpu", .{llama_dir})));
    libllama.root_module.addIncludePath(b.path(std.fmt.comptimePrint("{s}/src", .{llama_dir})));

    // Link library to executable
    exe.root_module.linkLibrary(libllama);
    exe.root_module.addIncludePath(b.path(std.fmt.comptimePrint("{s}/include", .{llama_dir})));
    exe.root_module.addIncludePath(b.path(std.fmt.comptimePrint("{s}/ggml/include", .{llama_dir})));
    
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run DMT CLI");
    run_step.dependOn(&run_cmd.step);
}
