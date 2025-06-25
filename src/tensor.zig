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
        data: []T,
        allocator: Allocator,

        /// Initialize a Tensor with a given shape
        fn init(allocator: Allocator, shape: []usize) !Tensor(T) {
            // Calculate the size from the shape of the Tensor
            var size = 1;
            for (shape) |dim| size *= dim;
            // Find the stride of Tensor
            const stride = stride_from_shape(shape, allocator);
            // Allocate the data
            const data = try allocator.alloc(T, size);
            return .{
                .size = size,
                .offset = 0,
                .stride = stride,
                .data = data,
            };
        }
    };
}

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
