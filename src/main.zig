const std = @import("std");
const pruner = @import("pruner.zig");
const exporter = @import("exporter.zig");

const llama = @cImport({
    @cInclude("llama.h");
    @cInclude("ggml-backend.h");
});

const Config = struct {
    // Model Paths
    student_path: []const u8 = "",
    judge_path: []const u8 = "",
    teacher_path: []const u8 = "",
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
    judge_top_k: i32 = 40,
    judge_top_p: f32 = 0.95,
    
    // Training Loop
    epochs: u32 = 1,
    save_freq: u32 = 100, // Save every N cycles
    quality: u8 = 1,      // Distillation quality level 1-10
    // Execution options
    read_linear: bool = true, // Linear is default for stapling
    cpu_only: bool = false,   // Extremely complex CPU optimization flag
};

fn getActiveParams(model: *llama.llama_model) u64 {
    return c_bridge.dmt_get_active_parameters(@ptrCast(model));
}

fn muteLlamaLogs(level: llama.ggml_log_level, text: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    _ = user_data;
    // Only pass through error/warning logs (GGML_LOG_LEVEL_ERROR = 2, GGML_LOG_LEVEL_WARN = 3)
    if (level == 2 or level == 3) {
        std.debug.print("{s}", .{text});
    }
}

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
    const is_staple = std.mem.eql(u8, command, "staple");
    const is_staple_distill = std.mem.eql(u8, command, "staple-distill");

    if (!is_train and !is_distill and !is_staple and !is_staple_distill) {
        printUsage();
        return;
    }

    const config = try parseArgs(args[2..]);
    
    if (is_train and (config.student_path.len == 0 or config.dataset_path.len == 0)) {
        std.debug.print("Error: --student and --dataset paths are required for training.\n", .{});
        return;
    }

    if (is_distill and config.student_path.len == 0) {
        std.debug.print("Error: --student path is required for distillation.\n", .{});
        return;
    }

    if ((is_staple or is_staple_distill) and (config.student_path.len == 0 or config.teacher_path.len == 0)) {
        std.debug.print("Error: --student and --teacher paths are required for stapling.\n", .{});
        return;
    }

    const do_staple = is_staple or is_staple_distill;
    const do_distill = is_distill or is_staple_distill or (is_staple and config.target_params_b > 0.0);

    // 1. Initialize llama.cpp backend
    llama.llama_log_set(muteLlamaLogs, null);
    llama.llama_backend_init();
    defer llama.llama_backend_free();
    llama.ggml_backend_load_all();

    // Core Error Handler
    errdefer |err| {
        std.debug.print("\n[CRITICAL ERROR] DMT Execution Halted: {s}\n", .{@errorName(err)});
        std.debug.print("Please verify paths, memory limits, and Ensure GGUF formats are valid.\n", .{});
    }

    std.debug.print("DMT Core: Backend Initialized. Threads: {d}\n", .{config.threads});

    // 2. Load Student Model (with Auto-Quantization based on param count)
    const opt_student_path = try prepareModel(init.io, allocator, config.student_path, "student", config.cpu_only);
    const opt_student_path_z = try allocator.dupeZ(u8, opt_student_path);
    defer allocator.free(opt_student_path_z);

    var student_params = llama.llama_model_default_params();
    student_params.n_gpu_layers = config.ngl_student;
    student_params.use_mmap = false; // Disable mmap so we can dynamically prune weights in RAM
    const student_model = llama.llama_load_model_from_file(opt_student_path_z.ptr, student_params) orelse return error.StudentLoadFailed;
    defer llama.llama_free_model(student_model);

    var engine_pruner = pruner.Pruner.init(allocator);

    if (do_staple) {
        std.debug.print("DMT: Commencing Parameter-Level Expansion (PLE) Stapling...\n", .{});

        // Apply CPU JIT Compilation to Teacher if requested, otherwise stream directly from disk
        const opt_teacher_path = if (config.cpu_only) 
            try prepareModel(init.io, allocator, config.teacher_path, "teacher", true)
        else 
            config.teacher_path;
            
        const opt_teacher_path_z = try allocator.dupeZ(u8, opt_teacher_path);
        defer allocator.free(opt_teacher_path_z);

        var teacher_params = llama.llama_model_default_params();
        teacher_params.n_gpu_layers = 0; // Force CPU
        teacher_params.use_mmap = true;  // READ DIRECTLY FROM DISK
        const teacher_model = llama.llama_load_model_from_file(opt_teacher_path_z.ptr, teacher_params) orelse return error.TeacherLoadFailed;
        
        // Execute C-Bridge PLE Stapler
        stapleModels(student_model, teacher_model, config.read_linear);
        
        llama.llama_free_model(teacher_model); // Free teacher memory after stapling

        if (!do_distill) {
            std.debug.print("Saving stapled model to {s}...\n", .{config.save_dir});
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
            std.debug.print("Stapling Complete.\n", .{});
            return;
        }
    }

    if (do_distill and !is_train) {
        const judge_model_path = if (config.judge_path.len > 0) config.judge_path else config.teacher_path;
        
        if (judge_model_path.len == 0) {
            // PURE STRUCTURAL DISTILLATION (No Judge provided)
            std.debug.print("DMT: Commencing Pure Structural Distillation (No Judge)...\n", .{});
            
            var current_active = getActiveParams(student_model);
            const target_p = if (config.target_params_b > 0.0) @as(u64, @intFromFloat(config.target_params_b * 1_000_000_000.0)) else 0;
            
            if (target_p > 0 and current_active > target_p) {
                var pass: u32 = 1;
                while (current_active > target_p) : (pass += 1) {
                    std.debug.print("--- Structural Pruning Pass {d} ---\n", .{pass});
                    
                    const diff = current_active - target_p;
                    const excess_ratio = @as(f32, @floatFromInt(diff)) / @as(f32, @floatFromInt(current_active));
                    const step_rate = if (excess_ratio < config.prune_rate) excess_ratio + 0.001 else config.prune_rate;
                    
                    std.debug.print("Guttering parameters (step rate: {d}%)...\n", .{step_rate * 100});
                    try pruneModelWeights(student_model, &engine_pruner, step_rate);
                    current_active = getActiveParams(student_model);
                    std.debug.print("Active Parameters: {d} / Target: {d}\n", .{current_active, target_p});
                }
            } else {
                std.debug.print("Guttering parameters (rate: {d}%)...\n", .{config.prune_rate * 100});
                try pruneModelWeights(student_model, &engine_pruner, config.prune_rate);
            }
            
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
            std.debug.print("Pure Structural Distillation Complete.\n", .{});
            return;
        }

        // JUDGE EVALUATION DISTILLATION (Iterative)
        const opt_judge_path = try prepareModel(init.io, allocator, judge_model_path, "judge", config.cpu_only);
        const opt_judge_path_z = try allocator.dupeZ(u8, opt_judge_path);
        defer allocator.free(opt_judge_path_z);

        var judge_params = llama.llama_model_default_params();
        judge_params.n_gpu_layers = config.ngl_judge;
        judge_params.use_mmap = false;
        const judge_model = llama.llama_load_model_from_file(opt_judge_path_z.ptr, judge_params) orelse return error.JudgeLoadFailed;
        defer llama.llama_free_model(judge_model);

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

        std.debug.print("DMT: Commencing Pure Distillation...\n", .{});
        
        const test_prompt = "Explain the core concepts of quantum computing in simple terms.";
        
        var cycle: u32 = 0;
        var keep_distilling = true;

        while (keep_distilling) {
            cycle += 1;
            std.debug.print("--- Refinement Cycle {d} ---\n", .{cycle});
            
            // Evaluation
            const response = try runInference(allocator, student_ctx, test_prompt, 256, config.student_temp, config.student_top_k, config.student_top_p, config.seed);
            defer allocator.free(response);

            const judge_prompt = try std.fmt.allocPrint(allocator, 
                "You are an ultra-pedantic AI judge. Compare the Student's output to the Expected Concept. " ++
                "Evaluate Logic, Factuality, and Efficiency. Output ONLY a single integer score between 0 and 1000000. " ++
                "Context: {s}\nStudent: {s}\nScore:", .{test_prompt, response});
            defer allocator.free(judge_prompt);
            
            const current_score_str = try runInference(allocator, judge_ctx, judge_prompt, 16, config.judge_temp, config.judge_top_k, config.judge_top_p, config.seed);
            defer allocator.free(current_score_str);
            const current_score = parseScore(current_score_str);
            std.debug.print("Current Judge Score: {d}\n", .{current_score});

            var current_prune_rate = config.prune_rate;

            if (config.target_params_b > 0.0) {
                const active_p = getActiveParams(student_model);
                const target_p = @as(u64, @intFromFloat(config.target_params_b * 1_000_000_000.0));
                
                std.debug.print("Active Parameters: {d} / Target: {d}\n", .{active_p, target_p});
                
                if (active_p <= target_p) {
                    std.debug.print("Target parameter size reached! Finalizing.\n", .{});
                    keep_distilling = false;
                    continue; // Skip pruning
                }
                
                // Adjust prune rate to not overshoot
                const diff = active_p - target_p;
                const excess_ratio = @as(f32, @floatFromInt(diff)) / @as(f32, @floatFromInt(active_p));
                if (excess_ratio < current_prune_rate) {
                    current_prune_rate = excess_ratio + 0.001; // Just enough to hit target
                }
            } else {
                if (cycle >= config.quality) {
                    keep_distilling = false;
                }
                current_prune_rate = config.prune_rate / @as(f32, @floatFromInt(config.quality));
            }

            if (keep_distilling or config.target_params_b > 0.0) {
                std.debug.print("Guttering low-impact parameters (step rate: {d}%)...\n", .{current_prune_rate * 100});
                try pruneModelWeights(student_model, &engine_pruner, current_prune_rate);
            }
        }

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
        std.debug.print("Iterative Distillation Complete.\n", .{});
        return;
    }

    if (is_train) {
        // --- Train Mode Specific Loading ---
        const opt_judge_path = try prepareModel(init.io, allocator, config.judge_path, "judge", config.cpu_only);
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

                if (config.target_params_b > 0.0) {
                    const active_p = getActiveParams(student_model);
                    const target_p = @as(u64, @intFromFloat(config.target_params_b * 1_000_000_000.0));
                    if (active_p <= target_p) {
                        std.debug.print("Target parameters reached ({d}). Stopping distillation.\n", .{active_p});
                        break;
                    }
                }
                
                // --- Step 1: Student Inference ---
                const response = try runInference(allocator, student_ctx, line, 256, config.student_temp, config.student_top_k, config.student_top_p, config.seed);
                defer allocator.free(response);
                
                // --- Step 2: Judge Evaluation ---
                const judge_prompt = try std.fmt.allocPrint(allocator, 
                    "You are an ultra-pedantic AI judge. Compare the Student's output to the Dataset Context. " ++
                    "Evaluate Logic, Factuality, and Efficiency. Output ONLY a single integer score between 0 and 1000000. " ++
                    "Context: {s}\nStudent: {s}\nScore:", .{line, response});
                defer allocator.free(judge_prompt);

                const score_str = try runInference(allocator, judge_ctx, judge_prompt, 16, config.judge_temp, config.judge_top_k, config.judge_top_p, config.seed);
                defer allocator.free(score_str);
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
}

fn parseArgs(args: [][]const u8) !Config {
    var config = Config{};
    var i: usize = 0;
    while (i < args.len) {
        const flag = args[i];

        // Handle parameterless boolean flags
        if (std.mem.eql(u8, flag, "--read-linear")) {
            config.read_linear = true;
            i += 1;
            continue;
        } else if (std.mem.eql(u8, flag, "--read-random")) {
            config.read_linear = false;
            i += 1;
            continue;
        }

        if (i + 1 >= args.len) break;
        const val = args[i + 1];
        i += 2;

        if (std.mem.eql(u8, flag, "--student")) config.student_path = val
        else if (std.mem.eql(u8, flag, "--judge")) config.judge_path = val
        else if (std.mem.eql(u8, flag, "--teacher")) config.teacher_path = val
        else if (std.mem.eql(u8, flag, "--dataset")) config.dataset_path = val
        else if (std.mem.eql(u8, flag, "--save-dir")) config.save_dir = val
        else if (std.mem.eql(u8, flag, "--out-format")) config.out_format = val
        else if (std.mem.eql(u8, flag, "--quant-type")) config.quant_type = val
        else if (std.mem.eql(u8, flag, "--quality") or std.mem.eql(u8, flag, "-q")) config.quality = try std.fmt.parseInt(u8, val, 10)
        else if (std.mem.eql(u8, flag, "--threads")) config.threads = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--threads-batch")) config.threads_batch = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--ngl-student")) config.ngl_student = try std.fmt.parseInt(i32, val, 10)
        else if (std.mem.eql(u8, flag, "--ngl-judge")) config.ngl_judge = try std.fmt.parseInt(i32, val, 10)
        else if (std.mem.eql(u8, flag, "--target-params")) config.target_params_b = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--prune-rate")) config.prune_rate = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--prune-method")) config.prune_method = val
        else if (std.mem.eql(u8, flag, "--student-ctx")) config.student_ctx_size = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--student-batch-size")) config.student_batch_size = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--student-temp")) config.student_temp = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--student-top-k")) config.student_top_k = try std.fmt.parseInt(i32, val, 10)
        else if (std.mem.eql(u8, flag, "--student-top-p")) config.student_top_p = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--judge-ctx")) config.judge_ctx_size = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--judge-temp")) config.judge_temp = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--judge-top-k")) config.judge_top_k = try std.fmt.parseInt(i32, val, 10)
        else if (std.mem.eql(u8, flag, "--judge-top-p")) config.judge_top_p = try std.fmt.parseFloat(f32, val)
        else if (std.mem.eql(u8, flag, "--epochs")) config.epochs = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--save-freq")) config.save_freq = try std.fmt.parseInt(u32, val, 10)
        else if (std.mem.eql(u8, flag, "--seed")) config.seed = try std.fmt.parseInt(u32, val, 10);
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
        \\  staple                Merge/Staple a big Teacher model onto a small Student model topology.
        \\  staple-distill        Staple models and then iteratively distill using the Judge.
        \\
        \\Required for Train:
        \\  --student <PATH>      Path to Student GGUF model
        \\  --judge <PATH>        Path to Judge GGUF model
        \\  --dataset <PATH>      Path to .jsonl dataset
        \\
        \\Required for Distill:
        \\  --student <PATH>      Path to Student GGUF model
        \\  --judge <PATH>        Path to Judge GGUF model (if quality > 1 or targeting parameters)
        \\
        \\Required for Staple / Staple-Distill:
        \\  --student <PATH>      Path to Student GGUF model
        \\  --teacher <PATH>      Path to massive Teacher GGUF model
        \\
        \\Execution Options:
        \\  --threads <N>         Number of threads for generation (default: 8)
        \\  --threads-batch <N>   Number of threads for batch/prompt processing (default: 8)
        \\  --ngl-student <N>     GPU layers for Student (default: 0)
        \\  --ngl-judge <N>       GPU layers for Judge/Teacher (default: 0)
        \\  --read-linear         (Staple Mode) Linearly scan Teacher directly from disk (default: true)
        \\  --read-random         (Staple Mode) Randomly access Teacher memory, faster if huge RAM.
        \\
        \\Distillation Options:
        \\  -q, --quality <N>     Distillation quality level [1-10] (default: 1). Higher = more refinement cycles.
        \\  --target-params <F>   Target parameter size in Billions (default: 5.5). Auto-halts when reached.
        \\  --prune-rate <F>      Percentage of weights to prune per drop (default: 0.01)
        \\  --prune-method <STR>  Pruning method [magnitude, random] (default: magnitude)
        \\  --epochs <N>          Number of training epochs (default: 1)
        \\
        \\Output Options:
        \\  --save-dir <PATH>     Directory to save checkpoints (default: out)
        \\  --out-format <FMT>    Output format [gguf, safetensors] (default: gguf)
        \\  --quant-type <TYPE>   Quantization type (e.g., f16, q4_k_m, q8_0, iq2_xxs) (default: q4_k_m)
        \\  --save-freq <N>       Save model every N cycles (default: 100)
        \\
        \\Inference Options:
        \\  --seed <N>            Random seed (default: 42)
        \\  --student-ctx <N>     Context size for Student (default: 4096)
        \\  --student-batch-size <N> Batch size for Student (default: 512)
        \\  --student-temp <F>    Temperature for Student generation (default: 0.7)
        \\  --student-top-k <N>   Top-K for Student generation (default: 40)
        \\  --student-top-p <F>   Top-P for Student generation (default: 0.95)
        \\  --judge-ctx <N>       Context size for Judge (default: 8192)
        \\  --judge-temp <F>      Temperature for Judge generation (default: 0.0)
        \\  --judge-top-k <N>     Top-K for Judge generation (default: 40)
        \\  --judge-top-p <F>     Top-P for Judge generation (default: 0.95)
        \\
    , .{});
}

fn runInference(allocator: std.mem.Allocator, ctx: *llama.llama_context, prompt: []const u8, max_tokens: i32, temp: f32, top_k: i32, top_p: f32, seed: u32) ![]const u8 {
    const model = llama.llama_get_model(ctx);
    const vocab = llama.llama_model_get_vocab(model);

    // Clear KV cache (sequence 0)
    const memory = llama.llama_get_memory(ctx);
    _ = llama.llama_memory_seq_rm(memory, 0, -1, -1);

    // 1. Tokenize prompt
    const tokens = try allocator.alloc(llama.llama_token, prompt.len + 4);
    defer allocator.free(tokens);
    const n_tokens = llama.llama_tokenize(vocab, prompt.ptr, @intCast(prompt.len), tokens.ptr, @intCast(tokens.len), true, true);
    if (n_tokens < 0) return error.TokenizeFailed;

    // 2. Setup sampler
    const sparams = llama.llama_sampler_chain_default_params();
    const smpl = llama.llama_sampler_chain_init(sparams);
    defer llama.llama_sampler_free(smpl);

    if (temp > 0) {
        llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_top_k(top_k));
        llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_top_p(top_p, 1));
        llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_temp(temp));
        llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_dist(seed));
    } else {
        llama.llama_sampler_chain_add(smpl, llama.llama_sampler_init_greedy());
    }

    // 3. Decode Loop
    var response: std.ArrayList(u8) = .empty;
    errdefer response.deinit(allocator);

    // Use n_tokens as max capacity for prompt batch
    var batch = llama.llama_batch_init(@intCast(n_tokens), 0, 1);
    defer llama.llama_batch_free(batch);

    // Load prompt tokens into batch
    for (0..@intCast(n_tokens)) |i| {
        batch.token[i] = tokens[i];
        batch.pos[i] = @intCast(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    // We only need logits for the last token to sample next
    batch.logits[@intCast(n_tokens - 1)] = 1;

    var n_cur: i32 = n_tokens;
    var n_gen: i32 = 0;
    while (n_gen < max_tokens) {
        if (llama.llama_decode(ctx, batch) != 0) return error.DecodeFailed;
        
        const token = llama.llama_sampler_sample(smpl, ctx, -1);
        llama.llama_sampler_accept(smpl, token);

        if (llama.llama_vocab_is_eog(vocab, token)) break;

        var piece_buf: [256]u8 = undefined;
        const n_piece = llama.llama_token_to_piece(vocab, token, &piece_buf, @intCast(piece_buf.len), 0, true);
        if (n_piece > 0) {
            try response.appendSlice(allocator, piece_buf[0..@intCast(n_piece)]);
        }

        // Prepare next batch with the single sampled token
        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        n_cur += 1;
        n_gen += 1;
    }

    return try response.toOwnedSlice(allocator);
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

const c_bridge = @cImport({
    @cInclude("dmt_bridge.h");
});

fn stapleModels(student: *llama.llama_model, teacher: *llama.llama_model, linear_read: bool) void {
    c_bridge.dmt_staple_models(@ptrCast(student), @ptrCast(teacher), linear_read);
}

fn pruneModelWeights(model: *llama.llama_model, p: *pruner.Pruner, rate: f32) !void {
    _ = p;
    c_bridge.dmt_prune_model_tensors(@ptrCast(model), rate);
}

fn prepareModel(io: std.Io, allocator: std.mem.Allocator, orig_path: []const u8, prefix: []const u8, cpu_only: bool) ![]const u8 {
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
    var ftype_name: []const u8 = if (cpu_only) "CPU_INT8" else "F16";
    
    const ONE_B = 1_000_000_000;
    
    if (cpu_only) {
        // Aggressive INT8/INT4 target compilation logic specifically targeting CPU AVX operations
        target_ftype = llama.LLAMA_FTYPE_MOSTLY_Q8_0;
        std.debug.print("   [CPU MODE] Compiling structural weights aggressively to INT8 bounds...\n", .{});
    } else {
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
        
        if (cpu_only) {
            // Force strict integer topologies into the compiler arrays simulating TPU XLA conversion bounds
            qparams.output_tensor_type = llama.GGML_TYPE_Q8_0;
            qparams.token_embedding_type = llama.GGML_TYPE_Q8_0;
            qparams.pure = true; 
        }
        
        if (llama.llama_model_quantize(orig_path_z.ptr, out_path_z.ptr, &qparams) != 0) {
            std.debug.print("   Warning: Auto-quantization failed (perhaps already quantized?). Proceeding with original.\n", .{});
            return orig_path;
        }
        return out_path;
    }
}
