//! Implementation of a Tensor

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const testing = std.testing;

/// A multidimensional container
pub fn Tensor(comptime T: type) type {
    return struct {
        size: usize,
        offset: usize,
        stride: ArrayList(isize),
        dim: usize,
        shape: ArrayList(usize),
        data: TensorData(T),
        allocator: Allocator,

        /// Initialize a Tensor with a given shape
        fn init(allocator: Allocator, shape: []usize) !Tensor(T) {
            // Calculate the size from the shape of the Tensor
            var size: usize = 1;
            for (shape) |dim| size *= dim;
            // Get the number of dimensions
            const dim = shape.len;
            // Find the stride
            const stride = try stride_from_shape(shape, allocator);
            // Create the shape arraylist
            var tensor_shape = ArrayList(usize).init(allocator);
            try tensor_shape.appendSlice(shape);
            // Allocate the data for the Tensor
            var tensor_data = try TensorData(T).initWithSize(allocator, size);
            defer tensor_data.deinit(); // Remove the reference from within this function
            const data = tensor_data.clone(); // Get a reference to the data
            // Creating the tensor
            return .{
                .size = size,
                .offset = 0,
                .stride = stride,
                .dim = dim,
                .shape = tensor_shape,
                .data = data,
                .allocator = allocator,
            };
        }

        /// Fill a Tensor with a particular value
        fn fill(self: *Tensor(T), val: T) void {
            self.data.fill(val);
        }

        /// Get a value from a position in the tensor
        fn at(self: *Tensor(T), index: []usize) TensorError!T {
            // Check that the input index is valid
            if (index.len != self.stride.items.len) {
                return TensorError.InvalidIndex;
            }
            // Calculate the position of the desired entry in items
            var position: isize = @intCast(self.offset);
            for (0..index.len) |i| {
                position += @as(isize, @intCast(index[i])) * self.stride.items[i];
            }
            // Bounds check
            if (position >= self.size) {
                return TensorError.InvalidIndex;
            }
            // Return the desired item
            return self.data.unwrap()[@as(usize, @intCast(position))];
        }

        /// Set a value at a position in the tensor
        fn set(self: *Tensor(T), index: []usize, value: T) TensorError!void {
            // Check that the input index is valid
            if (index.len != self.stride.items.len) {
                return TensorError.InvalidIndex;
            }
            // Calculate the position of the desired entry in items
            var position: isize = @intCast(self.offset);
            for (0..index.len) |i| {
                position += @as(isize, @intCast(index[i])) * self.stride.items[i];
            }
            // Bounds check
            if (position >= self.size or position < 0) {
                return TensorError.InvalidIndex;
            }
            // Set the value
            self.data.unwrap()[@as(usize, @intCast(position))] = value;
        }

        /// Get a slice from the Tensor
        pub fn slice(self: *Tensor(T), indices: []TensorSlice) TensorError!Tensor(T) {
            if (indices.len != self.dim) {
                return TensorError.IncorrectDimensions; // Too many slices
            }
            @panic("todo");
        }

        /// Deinitialize a Tensor
        fn deinit(self: *Tensor(T)) void {
            // Free the underlying items
            self.data.deinit();
            // Deinit the stride
            self.stride.deinit();
            // Deinit the dimsize
            self.shape.deinit();
            // Invalidate the reference
            self.* = undefined;
        }
    };
}

/// Tensor Errors
const TensorError = error{
    /// Attempted to access a position with a
    /// invalid index (wrong index dimension,
    /// or out of bounds)
    InvalidIndex,
    /// Attempted to slice with invalid slices
    InvalidSlice,
    /// Tried to slice with incorrect number of dimensions
    IncorrectDimensions,
};

/// The internal data array of a Tensor, essentially
/// just a reference counted pointer to a slice so that
/// multiple Tensors can hold references to the data
fn TensorData(comptime T: type) type {
    return struct {
        data: []T,
        allocator: Allocator,
        ref_count: *usize,

        /// Initialize a Tensor data value
        fn init(allocator: Allocator) !TensorData(T) {
            const rc = try allocator.create(usize);
            rc.* = 1;
            return .{
                .data = undefined,
                .allocator = allocator,
                .ref_count = rc,
            };
        }

        /// Initialize a Tensor data value with some size
        fn initWithSize(allocator: Allocator, size: usize) !TensorData(T) {
            const rc = try allocator.create(usize);
            rc.* = 1;
            return .{
                .data = try allocator.alloc(T, size),
                .allocator = allocator,
                .ref_count = rc,
            };
        }

        /// Deinitialize the pointer, if no more references remain,
        /// also deinitializes the underlying slice
        fn deinit(self: TensorData(T)) void {
            self.ref_count.* -= 1;
            if (self.ref_count.* == 0) {
                // No outstanding references, free data
                self.allocator.free(self.data);
                // Free the rc counter
                self.allocator.destroy(self.ref_count);
            }
        }

        /// Get another reference to the TensorData
        fn clone(self: TensorData(T)) TensorData(T) {
            self.ref_count.* += 1;
            return self;
        }

        /// Access the underlying slice to the data
        fn unwrap(self: *TensorData(T)) []T {
            return self.data;
        }

        /// Fill the data array with a value
        fn fill(self: *TensorData(T), value: T) void {
            @memset(self.data, value);
        }
    };
}

const TensorSlice = struct {
    start: isize,
    stop: isize,
    step: isize,
};

// Helper Functions
fn stride_from_shape(shape: []usize, allocator: Allocator) !ArrayList(isize) {
    var stride = ArrayList(isize).init(allocator);
    try stride.resize(shape.len);

    if (shape.len == 0) {
        return stride;
    }
    // Set the stride of the last dimension to 1
    stride.items[shape.len - 1] = 1;
    var accumulator: isize = 1;
    var idx = shape.len - 1;
    while (idx > 0) : (idx -= 1) {
        accumulator *= @as(isize, @intCast(shape[idx]));
        stride.items[idx - 1] = accumulator;
    }
    return stride;
}

// TESTS
test "Getting stride from shape (0D)" {
    var test_shape = [_]usize{};
    const test_stride = try stride_from_shape(&test_shape, testing.allocator);
    defer test_stride.deinit();

    var expected_stride_array = [_]isize{};
    var expected_stride = ArrayList(isize).init(testing.allocator);
    defer expected_stride.deinit();
    try expected_stride.appendSlice(&expected_stride_array);

    try testing.expect(std.mem.eql(isize, expected_stride.items, test_stride.items));
}

test "Getting stride from shape (1D)" {
    var test_shape = [_]usize{5};
    const test_stride = try stride_from_shape(&test_shape, testing.allocator);
    defer test_stride.deinit();

    var expected_stride_array = [_]isize{1};
    var expected_stride = ArrayList(isize).init(testing.allocator);
    defer expected_stride.deinit();
    try expected_stride.appendSlice(&expected_stride_array);

    try testing.expect(std.mem.eql(isize, expected_stride.items, test_stride.items));
}
test "Getting stride from shape (3D)" {
    var test_shape = [_]usize{ 2, 3, 5 };
    const test_stride = try stride_from_shape(&test_shape, testing.allocator);
    defer test_stride.deinit();

    var expected_stride_array = [_]isize{ 15, 5, 1 };
    var expected_stride = ArrayList(isize).init(testing.allocator);
    defer expected_stride.deinit();
    try expected_stride.appendSlice(&expected_stride_array);

    try testing.expect(std.mem.eql(isize, expected_stride.items, test_stride.items));
}

test "Creating Tensor Data" {
    var test_tensor_data = try TensorData(f64).initWithSize(testing.allocator, 10);
    defer test_tensor_data.deinit();
}

test "Getting a copy of Tensor Data" {
    var test_tensor_data = try TensorData(f64).initWithSize(testing.allocator, 10);
    defer test_tensor_data.deinit();

    var ref1 = test_tensor_data.clone();
    defer ref1.deinit();
}

test "Filling Tensor Data" {
    var test_tensor_data = try TensorData(f64).initWithSize(testing.allocator, 10);
    defer test_tensor_data.deinit();

    test_tensor_data.fill(0.0);
    for (test_tensor_data.unwrap()) |it| {
        try testing.expectEqual(0.0, it);
    }
}

test "Creating a Tensor (1D)" {
    var test_shape = [_]usize{5};
    var test_tensor = try Tensor(i32).init(testing.allocator, &test_shape);
    defer test_tensor.deinit();
}

test "Filling a Tensor" {
    var test_shape = [_]usize{ 5, 4 };
    var test_tensor = try Tensor(i32).init(testing.allocator, &test_shape);
    defer test_tensor.deinit();
    // Fill the tensor with 0s
    test_tensor.fill(0);
    // For each row/col check that the value is 0
    for (0..5) |row| {
        for (0..4) |col| {
            var index = [_]usize{ row, col };
            try testing.expectEqual(test_tensor.at(&index), 0);
        }
    }
}

test "Setting a Value" {
    var test_shape = [_]usize{ 5, 4 };
    var test_tensor = try Tensor(i32).init(testing.allocator, &test_shape);
    defer test_tensor.deinit();
    // Fill the tensor with 0s
    test_tensor.fill(0);

    // Set 3,1 to 5
    var idx = [_]usize{ 3, 1 };
    try test_tensor.set(&idx, 5);

    // Check that the index was indeed updated
    try testing.expectEqual(5, test_tensor.at(&idx));
}
