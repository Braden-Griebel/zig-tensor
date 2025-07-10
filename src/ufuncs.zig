//! Contains "universal" functions which
//! can be used by Tensors elementwise for
//! all numeric child types
const std = @import("std");
const math = std.math;

// This isn't really needed currently,
// but will make it easier to adjust
// implementations to make sense
// across all types

/// Struct with functions which can be applied to
/// all numeric types
pub fn UFunc(comptime T: type) type {
    return struct {
        fn add(e1: T, e2: T) T {
            return switch (@typeInfo(T)) {
                .int => math.add(T, e1, e2) catch math.maxInt(T),
                .float => math.add(T, e1, e2) catch math.inf(T),
            };
        }

        fn sub(e1: T, e2: T) T {
            return switch (@typeInfo(T)) {
                .int => math.sub(T, e1, e2) catch math.minInt(T),
                .float => math.sub(T, e1, e2) catch -math.inf(T),
            };
        }

        fn mult(e1: T, e2: T) T {
            return switch (@typeInfo(T)) {
                .int => math.mul(T, e1, e2) catch blk: {
                    if ((e1 < 0) != (e2 < 0)) {
                        break :blk math.minInt(T);
                    } else {
                        break :blk math.maxInt(T);
                    }
                },
                .float => math.mul(T, e1, e2) catch blk: {
                    if ((e1 < 0) != (e2 < 0)) {
                        break :blk -math.inf(T);
                    } else {
                        break :blk math.inf(T);
                    }
                },
            };
        }

        fn div(e1: T, e2: T) T {
            return switch (@typeInfo(T)) {
                .float => e1 / e2,
                .int => math.divTrunc(T, e1, e2),
            };
        }
    };
}
