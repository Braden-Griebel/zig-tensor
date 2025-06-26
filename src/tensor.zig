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
        stride: ArrayList(usize),
        dim: usize,
        dimsize: ArrayList(usize),
        data: []T,
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
            // Create the dimsize arraylist
            var dimsize = ArrayList(usize).init(allocator);
            try dimsize.appendSlice(shape);
            // Allocate the data for the Tensor
            const data = try allocator.alloc(T, size);
            // // Allocate the data
            // const data = try allocator.alloc(T, 0);
            // // Allocate the stride
            // const stride = ArrayList(usize).init(allocator);
            return .{
                .size = size,
                .offset = 0,
                .stride = stride,
                .dim = dim,
                .dimsize = dimsize,
                .data = data,
                .allocator = allocator,
            };
        }

        /// Fill a Tensor with a particular value
        fn fill(self: *Tensor(T), val: T) void {
            for (self.*.data, 0..) |_, idx| {
                self.*.data[idx] = val;
            }
        }

        /// Get a value from a position in the tensor
        fn at(self: *Tensor(T), index: []usize) TensorError!T {
            // Check that the input index is valid
            if (index.len != self.*.stride.items.len) {
                return TensorError.InvalidIndex;
            }
            // Calculate the position of the desired entry in items
            var position: usize = self.*.offset;
            for (0..index.len) |i| {
                position += index[i] * self.*.stride.items[i];
            }
            // Bounds check
            if (position >= self.*.size) {
                return TensorError.InvalidIndex;
            }
            // Return the desired item
            return self.*.data[position];
        }

        /// Deinitialize a Tensor
        fn deinit(self: *Tensor(T)) void {
            // Free the underlying items
            self.*.allocator.free(self.*.data);
            // Deinit the stride
            self.*.stride.deinit();
            // Deinit the dimsize
            self.*.dimsize.deinit();
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
};

// Helper Functions
fn stride_from_shape(shape: []usize, allocator: Allocator) !ArrayList(usize) {
    var stride = ArrayList(usize).init(allocator);
    if (shape.len == 0) {
        return stride;
    }
    try stride.append(1);
    var accumulator: usize = 1;
    var idx = shape.len - 1;
    while (idx > 0) {
        accumulator *= shape[idx];
        try stride.append(accumulator);
        idx -= 1;
    }
    reverse_array_list(stride);
    return stride;
}

fn reverse_array_list(to_reverse: ArrayList(usize)) void {
    var i: usize = 0;
    var j: usize = to_reverse.items.len - 1;
    var tmp: usize = 0;
    while (i < j) {
        tmp = to_reverse.items[i];
        to_reverse.items[i] = to_reverse.items[j];
        to_reverse.items[j] = tmp;
        i += 1;
        j -= 1;
    }
}

test "Reversing ArrayList" {
    // Create test_array which will be reversed
    var test_arraylist = ArrayList(usize).init(testing.allocator);
    defer test_arraylist.deinit();
    try test_arraylist.appendSlice(&[_]usize{ 5, 4, 3, 2, 1 });

    // Create expected array which is the reverse
    var expected_arraylist = ArrayList(usize).init(testing.allocator);
    defer expected_arraylist.deinit();
    try expected_arraylist.appendSlice(&[_]usize{ 1, 2, 3, 4, 5 });

    // Reverse the test list
    reverse_array_list(test_arraylist);

    // Check that the ArrayList was reversed
    try testing.expect(std.mem.eql(usize, expected_arraylist.items, test_arraylist.items));
}

test "Getting stride from shape (0D)" {
    var test_shape = [_]usize{};
    const test_stride = try stride_from_shape(&test_shape, testing.allocator);
    defer test_stride.deinit();

    var expected_stride_array = [_]usize{};
    var expected_stride = ArrayList(usize).init(testing.allocator);
    defer expected_stride.deinit();
    try expected_stride.appendSlice(&expected_stride_array);

    try testing.expect(std.mem.eql(usize, expected_stride.items, test_stride.items));
}

test "Getting stride from shape (1D)" {
    var test_shape = [_]usize{5};
    const test_stride = try stride_from_shape(&test_shape, testing.allocator);
    defer test_stride.deinit();

    var expected_stride_array = [_]usize{1};
    var expected_stride = ArrayList(usize).init(testing.allocator);
    defer expected_stride.deinit();
    try expected_stride.appendSlice(&expected_stride_array);

    try testing.expect(std.mem.eql(usize, expected_stride.items, test_stride.items));
}
test "Getting stride from shape (3D)" {
    var test_shape = [_]usize{ 2, 3, 5 };
    const test_stride = try stride_from_shape(&test_shape, testing.allocator);
    defer test_stride.deinit();

    var expected_stride_array = [_]usize{ 15, 5, 1 };
    var expected_stride = ArrayList(usize).init(testing.allocator);
    defer expected_stride.deinit();
    try expected_stride.appendSlice(&expected_stride_array);

    try testing.expect(std.mem.eql(usize, expected_stride.items, test_stride.items));
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
