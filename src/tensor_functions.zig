//! Contains functions which
//! can be used by Tensors elementwise for
//! all numeric child types

const std = @import("std");
const math = std.math;

/// Struct with unary functions which
/// can be applied to various numeric types
pub fn UFunc(comptime T: type) type {
    return struct {
        pub fn sin(e: T) T {
            return switch (@typeInfo(T)) {
                .float => @sin(e),
                .int => blk: {
                    // This should never by done...because why would you!?
                    const e_float: f64 = @floatFromInt(e);
                    const float_res = @sin(e_float);
                    const int_res: T = @intFromFloat(float_res);
                    break :blk int_res;
                },
                else => @compileError("Tried to take the sine of non-numeric values"),
            };
        }
        pub fn cos(e: T) T {
            return switch (@typeInfo(T)) {
                .float => @cos(e),
                .int => blk: {
                    // This should never by done...because why would you!?
                    const e_float: f64 = @floatFromInt(e);
                    const float_res = @cos(e_float);
                    const int_res: T = @intFromFloat(float_res);
                    break :blk int_res;
                },
                else => @compileError("Tried to take the cosine of non-numeric values"),
            };
        }
    };
}

/// Struct with binary functions which
/// can be applied to various numeric types
pub fn BFunc(comptime T: type) type {
    return struct {
        pub fn add(e1: T, e2: T) T {
            return switch (@typeInfo(T)) {
                .int => math.add(T, e1, e2) catch math.maxInt(T),
                .float => math.add(T, e1, e2) catch math.inf(T),
                else => @compileError("Tried to add non-numeric values"),
            };
        }

        pub fn sub(e1: T, e2: T) T {
            return switch (@typeInfo(T)) {
                .int => math.sub(T, e1, e2) catch math.minInt(T),
                .float => math.sub(T, e1, e2) catch -math.inf(T),
                else => @compileError("Tried to subtract non-numeric values"),
            };
        }

        pub fn mult(e1: T, e2: T) T {
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
                else => @compileError("Tried to multiply non-numeric values"),
            };
        }

        pub fn div(e1: T, e2: T) T {
            return switch (@typeInfo(T)) {
                .float => e1 / e2,
                .int => math.divTrunc(T, e1, e2),
                else => @compileError("Tried to divide non-numeric values"),
            };
        }
    };
}
