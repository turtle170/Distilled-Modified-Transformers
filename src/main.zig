const std = @import("std");
const pruner = @import("pruner.zig");
const exporter = @import("exporter.zig");

const llama = @cImport({
    @cInclude("llama.h");
});

const Config = struct {
    // Model Paths
    student_path: []const u8 = "",
    judge_path: []const u8 = "",
    dataset_path: []const u8 = "",
    save_dir: []const u8 = "out",
    out_format: []const u8 = "gguf",
    quant_type: []const u8 = "q4_k_m",

    // Hardware & Execution
    threads: u32 = 8,
    threads_batch: u32 = 8,
    seed: u32 = 42,
    ngl_student: i32 = 0, // Number of GPU layers for Student
    ngl_judge: i32 = 0,   // Number of GPU layers for Judge

    // Distillation & Pruning
    target_params_b: f32 = 5.5,
    prune_rate: f32 = 0.01,
    prune_method: []const u8 = "magnitude",
    
    // Student Inference Params
    student_ctx_size: u32 = 4096,
    student_batch_size: u32 = 512,
    student_temp: f32 = 0.7,
    student_top_k: i32 = 40,
    student_top_p: f32 = 0.95,
    
    // Judge Inference Params
    judge_ctx_size: u32 = 8192,
    judge_temp: f32 = 0.0, // Judge should be deterministic
    
    // Training Loop
    epochs: u32 = 1,
    save_freq: u32 = 100, // Save every N cycles
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_it.deinit();

    var args_list: std.ArrayList([]const u8) = .empty;
    defer args_list.deinit(allocator);
    while (args_it.next()) |arg| {
        args_list.append(allocator, arg) catch @panic("OOM");
    }
    const args = args_list.items;

    if (args.len < 2) {
        printUsage();
        return;
    }

    const command = args[1];
    const is_train = std.mem.eql(u8, command, "train");
    const is_distill = std.mem.eql(u8, command, "distill");

    if (!is_train and !is_distill) {
        printUsage();
        return;
    }

    const config = try parseArgs(args[2..]);
    
    if (is_train and (config.student_path.len == 0 or config.judge_path.len == 0 or config.dataset_path.len == 0)) {
        std.debug.print("Error: --student, --judge, and --dataset paths are required for training.\n", .{});
        return;
    }

    if (is_distill and (config.student_path.len == 0 or config.judge_path.len == 0)) {
        std.debug.print("Error: --student and --judge paths are required for distillation.\n", .{});
        return;
    }

    // 1. Initialize llama.cpp backend
    llama.llama_backend_init();
    defer llama.llama_backend_free();

    std.debug.print("DMT Core: Backend Initialized. Threads: {d}\n", .{config.threads});

    // 2. Load Student Model (with Auto-Quantization based on param count)
    const opt_student_path = try prepareModel(init.io, allocator, config.student_path, "student");
    const opt_student_path_z = try allocator.dupeZ(u8, opt_student_path);
    defer allocator.free(opt_student_path_z);

    var student_params = llama.llama_model_default_params();
    student_params.n_gpu_layers = config.ngl_student;
    const student_model = llama.llama_load_model_from_file(opt_student_path_z.ptr, student_params) orelse return error.StudentLoadFailed;
    defer llama.llama_free_model(student_model);

    var engine_pruner = pruner.Pruner.init(allocator);

    // --- Judge Loading (Shared) ---
    const opt_judge_path = try prepareModel(init.io, allocator, config.judge_path, "judge");
    const opt_judge_path_z = try allocator.dupeZ(u8, opt_judge_path);
    defer allocator.free(opt_judge_path_z);

    var judge_params = llama.llama_model_default_params();
    judge_params.n_gpu_layers = config.ngl_judge;
    const judge_model = llama.llama_load_model_from_file(opt_judge_path_z.ptr, judge_params) orelse return error.JudgeLoadFailed;
    defer llama.llama_free_model(judge_model);

    // 3. Setup Contexts
    var student_ctx_params = llama.llama_context_default_params();
    student_ctx_params.n_ctx = config.student_ctx_size;
    student_ctx_params.n_threads = @intCast(config.threads);
    student_ctx_params.n_threads_batch = @intCast(config.threads_batch);
    const student_ctx = llama.llama_new_context_with_model(student_model, student_ctx_params) orelse return error.StudentCtxFailed;
    defer llama.llama_free(student_ctx);

    var judge_ctx_params = llama.llama_context_default_params();
    judge_ctx_params.n_ctx = config.judge_ctx_size;
    judge_ctx_params.n_threads = @intCast(config.threads);
    judge_ctx_params.n_threads_batch = @intCast(config.threads_batch);
    const judge_ctx = llama.llama_new_context_with_model(judge_model, judge_ctx_params) orelse return error.JudgeCtxFailed;
    defer llama.llama_free(judge_ctx);

    if (is_distill) {
        std.debug.print("DMT: Commencing Pure Distillation (No Dataset)...\n", .{});
        
        const test_prompt = "Explain the core concepts of quantum computing in simple terms.";
        
        // Initial Evaluation
        std.debug.print("Running pre-distillation evaluation pass...\n", .{});
        const initial_response = try runInference(student_ctx, test_prompt, 256, config.student_temp);
        const initial_judge_prompt = try std.fmt.allocPrint(allocator, 
            "You are an ultra-pedantic AI judge. Compare the Student's output to the Expected Concept. " ++
            "Evaluate Logic, Factuality, and Efficiency. Output ONLY a single integer score between 0 and 1000000. " ++
            "Context: {s}\nStudent: {s}\nScore:", .{test_prompt, initial_response});
        defer allocator.free(initial_judge_prompt);
        
        const initial_score = parseScore(try runInference(judge_ctx, initial_judge_prompt, 16, config.judge_temp));
        std.debug.print("Pre-Distillation Judge Score: {d}\n", .{initial_score});

        std.debug.print("Performance optimization: Engaging {s} pruning at {d}%...\n", .{config.prune_method, config.prune_rate * 100});
        try pruneModelWeights(student_model, &engine_pruner, config.prune_rate);

        // Post Evaluation
        std.debug.print("Running post-distillation evaluation pass...\n", .{});
        const post_response = try runInference(student_ctx, test_prompt, 256, config.student_temp);
        const post_judge_prompt = try std.fmt.allocPrint(allocator, 
            "You are an ultra-pedantic AI judge. Compare the Student's output to the Expected Concept. " ++
            "Evaluate Logic, Factuality, and Efficiency. Output ONLY a single integer score between 0 and 1000000. " ++
            "Context: {s}\nStudent: {s}\nScore:", .{test_prompt, post_response});
        defer allocator.free(post_judge_prompt);
        
        const post_score = parseScore(try runInference(judge_ctx, post_judge_prompt, 16, config.judge_temp));
        std.debug.print("Post-Distillation Judge Score: {d} | Delta: {d}\n", .{post_score, post_score - initial_score});

        std.debug.print("Saving distilled model to {s}...\n", .{config.save_dir});
        const out_format = exporter.parseFormat(config.out_format);
        const quant_type = exporter.parseQuantType(config.quant_type);
        
        try exporter.exportModel(
            init.io,
            allocator, 
            student_model, 
            config.student_path, 
            config.save_dir, 
            0, 
            out_format, 
            quant_type
        );
        std.debug.print("Pure Distillation Complete.\n", .{});
        return;
    }

    // 4. DMT Training Loop
    var last_score: i64 = 0;
    var cycle: u32 = 0;

    std.debug.print("DMT Loop: Commencing Evolutionary Distillation...\n", .{});

    var epoch: u32 = 0;
    while (epoch < config.epochs) : (epoch += 1) {
        std.debug.print("Epoch {d}/{d}\n", .{epoch + 1, config.epochs});
        
        const dataset_file = std.Io.Dir.openFile(.cwd(), init.io, config.dataset_path, .{}) catch |err| {
            std.debug.print("Failed to open dataset: {}\n", .{err});
            return;
        };
        defer dataset_file.close(init.io);
        
        var file_buf: [32768]u8 = undefined;
        var file_reader = dataset_file.reader(init.io, &file_buf);
        var reader = &file_reader.interface;

        while (try reader.takeDelimiter('\n')) |line_raw| {
            const line = std.mem.trimEnd(u8, line_raw, "\r");
            cycle += 1;
            
            // --- Step 1: Student Inference ---
            // In full implementation, we use llama_decode and samplers here based on config.student_temp
            const response = try runInference(student_ctx, line, 256, config.student_temp);
            
            // --- Step 2: Judge Evaluation ---
            const judge_prompt = try std.fmt.allocPrint(allocator, 
                "You are an ultra-pedantic AI judge. Compare the Student's output to the Dataset Context. " ++
                "Evaluate Logic, Factuality, and Efficiency. Output ONLY a single integer score between 0 and 1000000. " ++
                "Context: {s}\nStudent: {s}\nScore:", .{line, response});
            defer allocator.free(judge_prompt);

            const score_str = try runInference(judge_ctx, judge_prompt, 16, config.judge_temp);
            const current_score = parseScore(score_str);

            std.debug.print("[Cycle {d}] Score: {d} | Delta: {d}\n", .{cycle, current_score, current_score - last_score});

            // --- Step 3: Rewire/Prune ---
            if (current_score < last_score) {
                std.debug.print("Performance drop. Engaging {s} pruning at {d}%...\n", .{config.prune_method, config.prune_rate * 100});
                try pruneModelWeights(student_model, &engine_pruner, config.prune_rate);
            }

            last_score = current_score;

            if (cycle % config.save_freq == 0) {
                std.debug.print("Saving model checkpoint to {s}...\n", .{config.save_dir});
                
                // Construct the export parameters
                const out_format = exporter.parseFormat(config.out_format);
                const quant_type = exporter.parseQuantType(config.quant_type);
                
                // Execute the zero-copy export or trigger conversion bridges
                try exporter.exportModel(
                    init.io,
                    allocator, 
                    student_model, 
                    config.student_path, 
                    config.save_dir, 
                    cycle, 
                    out_format, 
                    quant_type
                );
            }
        }
    }
}

fn parseArgs(args: [][]const u8) !Config {
    var config = Config{};
    var i: usize = 0;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) break;
        const flag = args[i];
        const val = args[i + 1];

        if (std.mem.eql(u8, flag, "--student")) config.student_path = val
        else if (std.mem.eql(u8, flag, "--judge")) config.judge_path = val
        else if (std.mem.eql(u8, flag, "--dataset")) config.dataset_path = val
        else if (std.mem.eql(u8, flag, "--save-dir")) config.save_dir = val
        else if (std.mem.eql(u8, flag, "--out-format")) config.out_format = val
        else if (std.mem.eql(u8, flag, "--quant-type")) config.quant_type = val
        else if (std.mem.eql(u8, flag, "--threads")) config.threads = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--threads-batch")) config.threads_batch = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--ngl-student")) config.ngl_student = try std.fmt.parseInt(i32, val, 10)
        else if (std.mem.eql(u8, flag, "--ngl-judge")) config.ngl_judge = try std.fmt.parseInt(i32, val, 10)
        else if (std.mem.eql(u8, flag, "--target-params")) config.target_params_b = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--prune-rate")) config.prune_rate = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--prune-method")) config.prune_method = val
        else if (std.mem.eql(u8, flag, "--student-ctx")) config.student_ctx_size = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--student-temp")) config.student_temp = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--judge-ctx")) config.judge_ctx_size = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--epochs")) config.epochs = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--save-freq")) config.save_freq = try std.fmt.parseInt(u32, val, 10);
    }
    return config;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: dmt <command> [options]
        \\
        \\Commands:
        \\  train                 Run evolutionary distillation with a Judge and dataset.
        \\  distill               Run pure distillation (pruning/quantization) directly on a Student model.
        \\
        \\Required for Train:
        \\  --student <PATH>      Path to Student GGUF model
        \\  --judge <PATH>        Path to Judge GGUF model
        \\  --dataset <PATH>      Path to .jsonl dataset
        \\
        \\Required for Distill:
        \\  --student <PATH>      Path to Student GGUF model
        \\
        \\Execution Options:
        \\  --threads <N>         Number of threads for generation (default: 8)
        \\  --threads-batch <N>   Number of threads for batch/prompt processing (default: 8)
        \\  --ngl-student <N>     GPU layers for Student (default: 0)
        \\  --ngl-judge <N>       GPU layers for Judge (default: 0)
        \\
        \\Distillation Options:
        \\  --target-params <F>   Target parameter size in Billions (default: 5.5)
        \\  --prune-rate <F>      Percentage of weights to prune per drop (default: 0.01)
        \\  --prune-method <STR>  Pruning method [magnitude, random] (default: magnitude)
        \\
        \\Output Options:
        \\  --save-dir <PATH>     Directory to save checkpoints (default: out)
        \\  --out-format <FMT>    Output format [gguf, safetensors, onnx, exl2, awq, pytorch, tf, tflite] (default: gguf)
        \\  --quant-type <TYPE>   Quantization type (e.g., f16, q4_k_m, q8_0, iq2_xxs) (default: q4_k_m)
        \\
        \\Inference Options:
        \\  --student-ctx <N>     Context size for Student (default: 4096)
        \\  --student-temp <F>    Temperature for Student generation (default: 0.7)
        \\  --judge-ctx <N>       Context size for Judge (default: 8192)
        \\  --epochs <N>          Number of training epochs (default: 1)
        \\  --save-freq <N>       Save model every N cycles (default: 100)
        \\
    , .{});
}

fn runInference(ctx: *llama.llama_context, prompt: []const u8, max_tokens: i32, temp: f32) ![]const u8 {
    _ = ctx; _ = prompt; _ = max_tokens; _ = temp;
    // In production, this sets up the llama_batch, llama_decode, and llama_sampler loops
    return "MOCKED_RESPONSE";
}

fn parseScore(s: []const u8) i64 {
    var val: i64 = 0;
    var found = false;
    for (s) |c| {
        if (std.ascii.isDigit(c)) {
            val = val * 10 + (c - '0');
            found = true;
        }
    }
    return if (found) val else 0;
}

fn pruneModelWeights(model: *llama.llama_model, p: *pruner.Pruner, rate: f32) !void {
    _ = model; _ = p; _ = rate;
    // Internal API implementation: iterates through ggml_tensors and applies Pruner
}

fn prepareModel(io: std.Io, allocator: std.mem.Allocator, orig_path: []const u8, prefix: []const u8) ![]const u8 {
    // 1. Load model with vocab_only to cheaply get metadata (fast, low memory)
    const orig_path_z = try allocator.dupeZ(u8, orig_path);
    defer allocator.free(orig_path_z);
    
    var meta_params = llama.llama_model_default_params();
    meta_params.vocab_only = true;
    const meta_model = llama.llama_load_model_from_file(orig_path_z.ptr, meta_params) orelse return error.MetaLoadFailed;
    
    const n_params = llama.llama_model_n_params(meta_model);
    llama.llama_free_model(meta_model);

    std.debug.print("-> Model '{s}' has {d} parameters.\n", .{prefix, n_params});

    var target_ftype: llama.llama_ftype = llama.LLAMA_FTYPE_MOSTLY_F16; // Default to no quant
    var ftype_name: []const u8 = "F16";
    
    const ONE_B = 1_000_000_000;
    
    // > 1B (e.g. 1.2B+)
    if (n_params > (ONE_B + 200_000_000)) {
        target_ftype = llama.LLAMA_FTYPE_MOSTLY_Q4_K_M;
        ftype_name = "Q4_K_M";
    } 
    // ~ 1B (800M to 1.2B)
    else if (n_params >= (ONE_B - 200_000_000) and n_params <= (ONE_B + 200_000_000)) {
        target_ftype = llama.LLAMA_FTYPE_MOSTLY_Q8_0;
        ftype_name = "Q8_0";
    } 
    // < 1B
    else {
        std.debug.print("   Model '{s}' < 1B params. Loading without forced quantization.\n", .{prefix});
        return orig_path;
    }

    // 2. Generate cached file path
    const cache_dir = ".dmt_cache";
    std.Io.Dir.createDir(.cwd(), io, cache_dir, .default_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    const out_path = try std.fmt.allocPrint(allocator, "{s}/{s}_{s}.gguf", .{cache_dir, prefix, ftype_name});
    
    // 3. If it already exists, return it
    if (std.Io.Dir.access(.cwd(), io, out_path, .{})) |_| {
        std.debug.print("   Cached optimized model found: {s}\n", .{out_path});
        return out_path;
    } else |_| {
        // Need to quantize
        std.debug.print("   Quantizing '{s}' to {s} (size requirements)... this may take a moment.\n", .{prefix, ftype_name});
        
        const out_path_z = try allocator.dupeZ(u8, out_path);
        defer allocator.free(out_path_z);

        var qparams = llama.llama_model_quantize_default_params();
        qparams.ftype = target_ftype;
        
        if (llama.llama_model_quantize(orig_path_z.ptr, out_path_z.ptr, &qparams) != 0) {
            std.debug.print("   Warning: Auto-quantization failed (perhaps already quantized?). Proceeding with original.\n", .{});
            return orig_path;
        }
        return out_path;
    }
}
