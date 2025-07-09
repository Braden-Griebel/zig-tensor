//! Utility methods for random sampling

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

/// An interface describing a random distribution
fn Distribution(comptime T: type) type {
    return struct {
        /// Pointer to the implementing type
        ptr: *anyopaque,
        /// Function to set the location of a Distribution
        setLocFn: *const fn (ptr: *anyopaque, loc: T) anyerror!void,
        /// Function to set the scale of a Distribution
        setScaleFn: *const fn (ptr: *anyopaque, scaled: T) anyerror!void,
        /// Function to sample from the distribution
        getRvsFn: *const fn (ptr: *anyopaque, allocator: Allocator, count: usize) anyerror!ArrayList(T),

        /// Set the location of the distribution (e.g. mean for a normal distribution)
        fn setLoc(self: Distribution, loc: T) !void {
            return self.setLocFn(self.ptr, loc);
        }

        /// Set the scale of the distribution (e.g. the std deviation for a normal distribution)
        fn setScale(self: Distribution, scale: T) !void {
            return self.setScaleFn(self.ptr, scale);
        }

        /// Get `count` samples from the distribution
        fn getRvs(self: Distribution(T), allocator: Allocator, count: usize) !ArrayList(T) {
            return self.getRvsFn(self.ptr, allocator, count);
        }
    };
}

/// A normal distribution
fn DistNormal(comptime T: type) type {
    return struct {
        /// The mean of the normal distribution
        loc: T,
        /// The standard deviation of the normal distribution
        scale: T,
        /// The random number generator
        generator: std.Random,

        /// Init the distribution with a loc, scale, and generator
        fn init(args: struct { generator: ?std.Random = null, loc: ?T = null, scale: ?T = null }) !DistNormal(T) {
            const dist_generator = args.generator orelse geninit: {
                var prng = std.Random.DefaultPrng.init(blk: {
                    var seed: u64 = undefined;
                    try std.posix.getrandom(std.mem.asBytes(&seed));
                    break :blk seed;
                });
                break :geninit prng.random();
            };
            const dist_loc: T = args.loc orelse @as(T, 0);
            const dist_scale: T = args.scale orelse @as(T, 0);
            return DistNormal(T){
                .loc = dist_loc,
                .scale = dist_scale,
                .generator = dist_generator,
            };
        }

        /// Set the mean of the normal distribution
        fn setLoc(ptr: *anyopaque, loc: T) !void {
            const self: *DistNormal(T) = @ptrCast(@alignCast(ptr));
            self.loc = loc;
        }

        /// Set the standard deviation of the normal distribution
        fn setScale(ptr: *anyopaque, scale: T) !void {
            const self: *DistNormal(T) = @ptrCast(@alignCast(ptr));
            self.scale = scale;
        }

        /// Get `count` samples from a normal distribution
        fn getRvs(ptr: *anyopaque, allocator: Allocator, count: usize) !ArrayList(T) {
            const self: *DistNormal(T) = @ptrCast(@alignCast(ptr));
            var samples = try ArrayList(T).initCapacity(allocator, count);
            while (samples.items.len < count) {
                // Generate two U(0,1) samples from the random number generator
                const unif1: T = self.generator.float(T);
                const unif2: T = self.generator.float(T);
                // Generate a pair of random samples from a normal distribution
                // using the Box-Muller transform
                const r: T = @sqrt(-2 * @log(unif1));
                const theta: T = 2 * std.math.pi * unif2;
                // The first sample will always be appended
                const norm1 = (r * @cos(theta)) * self.scale + self.loc;
                try samples.append(norm1);
                // If the desired number of samples has been generated, stop
                if (samples.items.len == count) {
                    break;
                }
                const norm2 = (r * @sin(theta)) * self.scale + self.loc;
                try samples.append(norm2);
            }
            return samples;
        }

        /// Get a distribution interface
        fn distribution(self: *DistNormal(T)) Distribution(T) {
            return Distribution(T){
                .ptr = self,
                .setLocFn = setLoc,
                .setScaleFn = setScale,
                .getRvsFn = getRvs,
            };
        }
    };
}

fn DistUniform(comptime T: type) type {
    return struct {
        /// The minimum of the uniform distribution
        loc: T,
        /// The maximum of the uniform distribution
        scale: T,
        /// The random number generator
        generator: std.Random,

        /// Initialize the distirbution with a loc(minimum), scale(maximum), and generator
        fn init(args: struct { generator: ?std.Random = null, loc: ?T = null, scale: ?T = null }) !DistUniform(T) {
            const dist_generator = args.generator orelse geninit: {
                var prng = std.Random.DefaultPrng.init(blk: {
                    var seed: u64 = undefined;
                    try std.posix.getrandom(std.mem.asBytes(&seed));
                    break :blk seed;
                });
                break :geninit prng.random();
            };
            const dist_loc: T = args.loc orelse @as(T, 0);
            const dist_scale: T = args.scale orelse @as(T, 0);
            return DistUniform(T){
                .loc = dist_loc,
                .scale = dist_scale,
                .generator = dist_generator,
            };
        }

        /// Set the lower bound of the uniform distribution
        fn setLoc(ptr: *anyopaque, loc: T) !void {
            const self: *DistUniform(T) = @ptrCast(@alignCast(ptr));
            self.loc = loc;
        }

        /// Set the upper bond of the uniform distributionu
        fn setScale(ptr: *anyopaque, scale: T) !void {
            const self: *DistUniform(T) = @ptrCast(@alignCast(ptr));
            self.scale = scale;
        }

        /// Get `count` samples from a Uniform(loc,scale) distribution
        fn getRvs(ptr: *anyopaque, allocator: Allocator, count: usize) anyerror!ArrayList(T) {
            const self: *DistUniform(T) = @ptrCast(@alignCast(ptr));
            var samples = try ArrayList(T).initCapacity(allocator, count);
            while (samples.items.len < count) {
                // Generate a sample of the appropriate numeric type
                try samples.append(switch (@typeInfo(T)) {
                    .int => self.generator.intRangeLessThan(T, self.loc, self.scale),
                    .float => (self.generator.float(T) * (self.scale - self.loc)) + self.loc,
                    else => @compileError("Non-numeric type provided for uniform distribution"),
                });
            }
            return samples;
        }

        /// Get a distribution interface
        fn distribution(self: *DistUniform(T)) Distribution(T) {
            return Distribution(T){
                .ptr = self,
                .setLocFn = setLoc,
                .setScaleFn = setScale,
                .getRvsFn = getRvs,
            };
        }
    };
}

test "Normal Distribution for float64: mean=10,std=3" {
    const num_samples: usize = 1000;
    const desired_mean: f64 = 10.0;
    const desired_std: f64 = 3.0;
    const sample_type: type = f64;
    // Create the test distribution
    var test_prng = std.Random.DefaultPrng.init(314);
    const test_generator = test_prng.random();
    var test_normal_dist = try DistNormal(sample_type).init(.{ .generator = test_generator, .loc = desired_mean, .scale = desired_std });
    const test_dist: Distribution(sample_type) = test_normal_dist.distribution();
    // Get a sample
    const samples = try test_dist.getRvs(testing.allocator, num_samples);
    defer samples.deinit();
    try testing.expectEqual(num_samples, samples.items.len);
    // Calculate the mean (should be about desired mean)
    var sum: sample_type = 0;
    for (samples.items) |sample| {
        sum += sample;
    }
    const mean: sample_type = sum / @as(sample_type, @floatFromInt(num_samples));
    try testing.expectApproxEqAbs(desired_mean, mean, 0.1);
    // Calculate the standard deviation, should be desired std
    var resid_square_sum: sample_type = 0;
    for (samples.items) |sample| {
        resid_square_sum += std.math.pow(sample_type, sample - mean, 2.0);
    }
    const test_std = @sqrt(resid_square_sum / @as(sample_type, @floatFromInt(num_samples)));
    try testing.expectApproxEqAbs(desired_std, test_std, 0.1);
}

test "Uniform Distribution for Int: min=5, max=15" {
    const sample_type: type = i32;
    const num_samples: usize = 1000;
    const desired_min: sample_type = 5;
    const desired_max: sample_type = 15;
    // Create the test distribution
    var test_prng = std.Random.DefaultPrng.init(314);
    const test_generator = test_prng.random();
    var test_uniform_dist = try DistUniform(sample_type).init(.{ .generator = test_generator, .loc = desired_min, .scale = desired_max });
    const test_dist: Distribution(sample_type) = test_uniform_dist.distribution();
    // Get a sample
    const samples = try test_dist.getRvs(testing.allocator, num_samples);
    defer samples.deinit();
    try testing.expectEqual(num_samples, samples.items.len);
}

test "Uniform Distribution for Float: min=5, max=15" {
    const sample_type: type = f32;
    const num_samples: usize = 1000;
    const desired_min: sample_type = 5;
    const desired_max: sample_type = 15;
    // Create the test distribution
    var test_prng = std.Random.DefaultPrng.init(314);
    const test_generator = test_prng.random();
    var test_uniform_dist = try DistUniform(sample_type).init(.{ .generator = test_generator, .loc = desired_min, .scale = desired_max });
    const test_dist: Distribution(sample_type) = test_uniform_dist.distribution();
    // Get a sample
    const samples = try test_dist.getRvs(testing.allocator, num_samples);
    defer samples.deinit();
    try testing.expectEqual(num_samples, samples.items.len);
}
