const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    const exe = b.addExecutable(.{
        .name = "dmt",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.want_lto = true;

    // --- Modern libllama & libggml Build Integration ---
    // Assuming llama.cpp source is cloned to 'deps/llama.cpp'
    const llama_dir = "deps/llama.cpp";
    
    const libllama = b.addStaticLibrary(.{
        .name = "llama",
        .target = target,
        .optimize = optimize,
    });
    libllama.want_lto = true;
    libllama.linkLibC();
    libllama.linkLibCpp();

    // Hyper-optimization flags using march=native for portability and performance
    const c_flags = &[_][]const u8{
        "-std=c11", "-fPIC", "-O3", "-Wall", "-Wextra",
        "-DGGML_USE_K_QUANTS", "-D_GNU_SOURCE",
        "-march=native", "-mtune=native",
    };

    const cpp_flags = &[_][]const u8{
        "-std=c++17", "-fPIC", "-O3", "-Wall", "-Wextra",
        "-DGGML_USE_K_QUANTS", "-D_GNU_SOURCE",
        "-march=native", "-mtune=native",
    };

    // 1. Compile GGML core (modern llama.cpp structure)
    libllama.addCSourceFiles(.{
        .root = b.path(llama_dir),
        .files = &.{
            "ggml/src/ggml.c",
            "ggml/src/ggml-alloc.c",
            "ggml/src/ggml-backend.c",
            "ggml/src/ggml-quants.c",
            "ggml/src/ggml-cpu/ggml-cpu.c",
            "ggml/src/ggml-cpu/ggml-cpu-quants.c",
        },
        .flags = c_flags,
    });

    // 2. Compile Llama core
    libllama.addCSourceFiles(.{
        .root = b.path(llama_dir),
        .files = &.{
            "src/llama.cpp",
            "src/llama-vocab.cpp",
            "src/llama-grammar.cpp",
            "src/llama-sampling.cpp",
        },
        .flags = cpp_flags,
    });

    // Include paths
    libllama.addIncludePath(b.path(std.fmt.comptimePrint("{s}/include", .{llama_dir})));
    libllama.addIncludePath(b.path(std.fmt.comptimePrint("{s}/ggml/include", .{llama_dir})));
    libllama.addIncludePath(b.path(std.fmt.comptimePrint("{s}/src", .{llama_dir})));

    // Link library to executable
    exe.linkLibrary(libllama);
    exe.addIncludePath(b.path(std.fmt.comptimePrint("{s}/include", .{llama_dir})));
    
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run DMT CLI");
    run_step.dependOn(&run_cmd.step);
}
