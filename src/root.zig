//! Zig-tensor is a small library for creating and working
//! with multidimentional containers commonly called tensors.
//! It's goal is similar to numpy in python, xtensor in c++,
//! or ndarray in rust, but it will remain small and simple, and
//! mostly be useful for learning about how multidimensional arrays
//! are handled.
const std = @import("std");

pub const tensor = @import("tensor.zig");

test "Tensor" {
    std.testing.refAllDecls(@This());
}
