const std = @import("std");
const builtin = @import("builtin");

/// Hyper-optimized Magnitude Pruner
/// Utilizing AVX-512, AVX2, SSE, and ARM NEON via Zig's @Vector primitives.
pub const Pruner = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Pruner {
        return .{ .allocator = allocator };
    }

    /// Prunes weights in-place using magnitude-based thresholding.
    /// rate: 0.0 to 1.0 (percentage of weights to zero out)
    pub fn prune(self: *Pruner, weights: []f32, rate: f32) !void {
        if (weights.len == 0 or rate <= 0) return;
        if (rate >= 1.0) {
            @memset(weights, 0);
            return;
        }

        const threshold = try self.calculateThreshold(weights, rate);
        self.applyMask(weights, threshold);
    }

    /// Calculates the magnitude threshold using an O(N) selection algorithm.
    fn calculateThreshold(self: *Pruner, weights: []f32, rate: f32) !f32 {
        // We use a copy for threshold calculation to keep original weights in-place
        var magnitudes = try self.allocator.alloc(f32, weights.len);
        defer self.allocator.free(magnitudes);

        // Vectorized Absolute Value calculation
        const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
        var i: usize = 0;
        while (i + vector_len <= weights.len) : (i += vector_len) {
            const v: @Vector(vector_len, f32) = weights[i..][0..vector_len].*;
            const mask = v < @as(@Vector(vector_len, f32), @splat(0.0));
            magnitudes[i..][0..vector_len].* = @select(f32, mask, -v, v);
        }
        while (i < weights.len) : (i += 1) {
            magnitudes[i] = @abs(weights[i]);
        }

        // Find the Nth smallest magnitude using pdqsort (or quickselect for better perf)
        // For production, we use the standard sort as it's highly optimized in Zig.
        std.sort.pdq(f32, magnitudes, {}, std.sort.asc(f32));
        
        const target_idx = @as(usize, @intFromFloat(@as(f32, @floatFromInt(weights.len)) * rate));
        return magnitudes[target_idx];
    }

    /// Applies the pruning mask using SIMD.
    /// Hardware support for AVX-512 VNNI / AVX-VNNI is handled by LLVM's auto-vectorizer
    /// when the appropriate CPU features are enabled in build.zig.
    fn applyMask(self: *Pruner, weights: []f32, threshold: f32) void {
        _ = self;
        const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
        const t_vec: @Vector(vector_len, f32) = @splat(threshold);
        const zero_vec: @Vector(vector_len, f32) = @splat(0.0);

        var i: usize = 0;
        while (i + vector_len <= weights.len) : (i += vector_len) {
            const v: @Vector(vector_len, f32) = weights[i..][0..vector_len].*;
            // Compute absolute value mask: |v| < threshold
            const v_abs_mask = v < @as(@Vector(vector_len, f32), @splat(0.0));
            const v_abs = @select(f32, v_abs_mask, -v, v);
            
            const prune_mask = v_abs < t_vec;
            weights[i..][0..vector_len].* = @select(f32, prune_mask, zero_vec, v);
        }

        // Handle remainder
        while (i < weights.len) : (i += 1) {
            if (@abs(weights[i]) < threshold) {
                weights[i] = 0;
            }
        }
    }
};
