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
        pub fn initWithShape(allocator: Allocator, shape: []usize) !Tensor(T) {
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

        /// Initialize a Tensor with a given shape, and filled with a value
        pub fn initFilled(allocator: Allocator, shape: []usize, fill_val: T) !Tensor(T) {
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
            defer tensor_data.deinit(); // Remove the reference from within this funciton
            // Fill the Tensor data with the specified value
            tensor_data.fill(fill_val);
            const data = tensor_data.clone();
            // Create and return the Tensor
            return Tensor(T){
                .size = size,
                .offset = 0,
                .stride = stride,
                .dim = dim,
                .shape = tensor_shape,
                .data = data,
                .allocator = allocator,
            };
        }

        pub fn initWithSlice(allocator: Allocator, shape: []usize, in_data: []T) !Tensor(T) {
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
            var tensor_data = try TensorData(T).initWithSlice(allocator, in_data);
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
        pub fn fill(self: *Tensor(T), val: T) void {
            self.data.fill(val);
        }

        /// Get a value from a position in the tensor
        pub fn at(self: *Tensor(T), index: []usize) TensorError!T {
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
            if (position >= self.data.unwrap().len) {
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
            if (position >= self.data.unwrap().len or position < 0) {
                return TensorError.InvalidIndex;
            }
            // Set the value
            self.data.unwrap()[@as(usize, @intCast(position))] = value;
        }

        /// Get a slice from the Tensor
        ///
        /// Note that this Tensor will reference the same data as the parent tensor,
        /// and will have the same dimensionality regardless of the slices
        pub fn slice(self: *Tensor(T), indices: []TensorSlice) TensorError!Tensor(T) {
            if (indices.len != self.dim) {
                return TensorError.InvalidDimensions; // Too many slices
            }
            // The new size will be the product of stop-start/step for each dimension
            var new_size: usize = 1;
            // The new offset will be the old offset + stride*start for each slice
            var new_offset: usize = self.offset;
            // The new strides will be old_stride * step for each dimension
            var new_stride: ArrayList(isize) = try ArrayList(isize).initCapacity(
                self.allocator,
                self.dim,
            );
            try new_stride.resize(self.dim);
            // The new dimensionality (same as previous dimensionality)
            const new_dim = self.dim;
            // The new shape will be the stop-start/step for each dimension
            var new_shape = try ArrayList(usize).initCapacity(self.allocator, self.dim);
            try new_shape.resize(self.dim);
            // Get a new data reference
            const new_data = self.data.clone();
            // The allocator will be the same
            const new_allocator = self.allocator;

            // Iterate through the slices to update the values
            for (indices, 0..) |s, idx| {
                const slice_start = s.start orelse 0;
                const slice_step = s.step orelse 1;
                const slice_stop = s.stop orelse @as(isize, @intCast(self.shape.items[idx]));
                const dim_size = @as(usize, @intCast(@divFloor((slice_stop - slice_start), slice_step)));
                // Update the size
                new_size *= dim_size;
                // Update the shape
                new_shape.items[idx] = dim_size;
                // Update the stride
                new_stride.items[idx] = slice_step * self.stride.items[idx];
                // Update the offset
                new_offset += @as(usize, @intCast(self.stride.items[idx] * slice_start));
            }
            // Return the newly created tensor
            return .{
                .size = new_size,
                .offset = new_offset,
                .stride = new_stride,
                .dim = new_dim,
                .shape = new_shape,
                .data = new_data,
                .allocator = new_allocator,
            };
        }

        /// Create a new Tensor view by reshaping the Tensor to the given shape
        ///
        /// Will return an error instead of a new Tensor if the Tensor can't
        /// be reshaped to the provided shape
        ///
        /// Note: Currently this just works on the underlying data of a Tensor
        /// (i.e. just reshaping the linear memory backing a Tensor), which
        /// can yield unexpected results when the Tensor being reshaped
        /// is just a view. It is recommended to clone a Tensor if it is a
        /// view instead
        pub fn reshape(self: *Tensor(T), new_shape: []usize) TensorError!T {
            // Calculate the size from the shape of the Tensor
            var new_size: usize = 1;
            for (new_shape) |dim| new_size *= dim;
            if (new_size != self.size) {
                return TensorError.InvalidShape;
            }
            // Get the number of dimensions
            const dim = new_shape.len;
            // Find the stride
            const stride = try stride_from_shape(new_shape, self.allocator);
            // Create the shape arraylist
            var tensor_shape = ArrayList(usize).init(self.allocator);
            try tensor_shape.appendSlice(new_shape);
            // Copy the data from the previous array
            const data = self.data.clone();
            // Creating the reshaped tensor
            return .{
                .size = new_shape,
                .offset = 0,
                .stride = stride,
                .dim = dim,
                .shape = tensor_shape,
                .data = data,
                .allocator = self.allocator,
            };
        }

        /// Create (and return) a deep copy of the Tensor,
        /// with new memory (which will be laid out in row major
        /// order regardless of the data arrangement of the
        /// Tensor being cloned)
        pub fn clone(self: *Tensor(T)) !Tensor(T) {
            // Start by creating a new TensorData structure with the needed memory
            var new_tensor_data = try TensorData(T).initWithSize(self.allocator, self.size);
            var data_items: []T = new_tensor_data.unwrap();

            // Copy data from self into the new tensor data
            // NOTE: A memcpy won't work, as the arragement may need to change
            // TODO: Determine cases where a simple memcpy will perform as expected
            var old_tensor_iter = try self.getIndexIter();
            defer old_tensor_iter.deinit();

            for (0..self.size) |idx| {
                // Copy over the data
                data_items[idx] = self.data.unwrap()[old_tensor_iter.current_data_index];
                // Increment the old tensor iterator, and if that can no longer iterate,
                // break the loop
                if (!old_tensor_iter.increment()) {
                    std.debug.assert(idx == self.size - 1);
                    break;
                }
            }

            // Get the stride of the new tensor
            const new_tensor_stride = try stride_from_shape(self.shape.items, self.allocator);

            return Tensor(T){
                .size = self.size,
                .offset = 0,
                .stride = new_tensor_stride,
                .dim = self.dim,
                .shape = try self.shape.clone(),
                .data = new_tensor_data,
                .allocator = self.allocator,
            };
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

        /// Get an iterator over the Tensor's indices
        pub fn getIndexIter(self: *Tensor(T)) !TensorIndexIter(T) {
            return try TensorIndexIter(T).init(self);
        }
    };
}

/// A helper struct for iterating through the entries
/// in a Tensor
pub fn TensorIndexIter(comptime T: type) type {
    return struct {
        /// The allocator used for allocating current index
        allocator: Allocator,
        /// The strides of the Tensor being indexed
        tensor_strides: []isize,
        /// The shape of the Tensor being indexed
        tensor_shape: []usize,
        /// The offset of the Tensor being indexed
        tensor_offset: usize,
        /// The current index in the Tensor
        current_tensor_index: []usize,
        /// The current index in the TensorData
        current_data_index: usize,

        /// Initialize the Tensor Index Iter to iterate through the
        /// data in the Tensor
        fn init(tensor: *Tensor(T)) !TensorIndexIter(T) {
            const current_tensor_index = try tensor.allocator.alloc(usize, tensor.dim);
            @memset(current_tensor_index, 0);

            return .{
                .allocator = tensor.allocator,
                .tensor_strides = tensor.stride.items,
                .tensor_shape = tensor.shape.items,
                .tensor_offset = tensor.offset,
                .current_tensor_index = current_tensor_index,
                .current_data_index = 0,
            };
        }

        /// Free the memory associated with the TensorIndexIter (a slice of usize with
        /// length equal to the dimensions of the indexed Tensor)
        fn deinit(self: *TensorIndexIter(T)) void {
            self.allocator.free(self.current_tensor_index);
            self.* = undefined;
        }

        /// Increment the current_data_index to the next
        /// Tensor index
        ///
        /// Returns true if able to increment, and false otherwise
        fn increment(self: *TensorIndexIter(T)) bool {
            // Whether to carry to the next index
            var carry: bool = true;
            // Current index in the tensor index
            var index_index: usize = self.tensor_strides.len;
            // While we need to find something to increment, step through
            while (carry and index_index > 0) {
                index_index -= 1;
                carry = false;
                // Check if the current index can be incremented without carrying
                if (self.current_tensor_index[index_index] < self.tensor_shape[index_index] - 1) {
                    self.current_tensor_index[index_index] += 1;
                    // No need to carry to the next index
                    carry = false;
                } else if (self.current_tensor_index[index_index] == self.tensor_shape[index_index] - 1) {
                    // To increment, need to carry to next index
                    // Set the current index to 0, and mark the need to carry
                    self.current_tensor_index[index_index] = 0;
                    carry = true;
                } else {
                    unreachable;
                }
            }
            // If still needing to carry, the index
            // was unable to be incremented, return false
            if (carry) {
                return false;
            }
            // Update the data index
            var new_data_index = self.tensor_offset;
            for (self.current_tensor_index, self.tensor_strides) |idx, stride| {
                new_data_index += idx * @as(usize, @intCast(stride));
            }
            self.current_data_index = new_data_index;
            return true;
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
    /// Attempted to slice with incorrect number of dimensions
    InvalidDimensions,
    /// Attempted to reshape a Tensor with an invalid shape
    InvalidShape,
    /// If an allocation fails
    OutOfMemory,
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

        /// Create a Tensor data value filled from a passed slice
        fn initWithSlice(allocator: Allocator, data_slice: []T) !TensorData(T) {
            const rc = try allocator.create(usize);
            rc.* = 1;
            return .{
                .data = try allocator.dupe(T, data_slice),
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
    start: ?isize = null,
    stop: ?isize = null,
    step: ?isize = null,
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

test "Filling Tensor Data From Slice" {
    var test_array = [_]i32{ 1, 2, 3, 4, 5, 6 };
    var test_tensor_data = try TensorData(i32).initWithSlice(testing.allocator, &test_array);
    defer test_tensor_data.deinit();

    try testing.expectEqualSlices(i32, &test_array, test_tensor_data.data);
}

test "Creating a Tensor (1D)" {
    var test_shape = [_]usize{5};
    var test_tensor = try Tensor(i32).initWithShape(testing.allocator, &test_shape);
    defer test_tensor.deinit();
}

test "Creating a Tensor from a Slice" {
    var shape_array = [_]usize{ 3, 2 };
    var test_array = [_]i32{
        0,
        1,
        2,
        3,
        4,
        5,
    };
    var test_tensor = try Tensor(i32).initWithSlice(testing.allocator, &shape_array, &test_array);
    defer test_tensor.deinit();

    for (0..6) |idx| {
        var tensor_index = [_]usize{ @divFloor(idx, 2), idx % 2 };
        try testing.expectEqual(@as(i32, @intCast(idx)), test_tensor.at(&tensor_index));
    }
}

test "Filling a Tensor" {
    var test_shape = [_]usize{ 5, 4 };
    var test_tensor = try Tensor(i32).initWithShape(testing.allocator, &test_shape);
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
    var test_tensor = try Tensor(i32).initWithShape(testing.allocator, &test_shape);
    defer test_tensor.deinit();
    // Fill the tensor with 0s
    test_tensor.fill(0);

    // Set 3,1 to 5
    var idx = [_]usize{ 3, 1 };
    try test_tensor.set(&idx, 5);

    // Check that the index was indeed updated
    try testing.expectEqual(5, test_tensor.at(&idx));
}

test "Slicing a Tensor" {
    var test_shape = [_]usize{ 3, 3 };
    var test_tensor_data = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    var test_tensor = try Tensor(i32).initWithSlice(testing.allocator, &test_shape, &test_tensor_data);
    defer test_tensor.deinit();

    var test_slices = [_]TensorSlice{ .{ .start = 1 }, .{ .start = 1 } };
    var test_sliced_tensor = try test_tensor.slice(&test_slices);
    defer test_sliced_tensor.deinit();

    var idx1 = [_]usize{ 0, 0 };
    try testing.expectEqual(4, try test_sliced_tensor.at(&idx1));

    var idx2 = [_]usize{ 0, 1 };
    try testing.expectEqual(5, try test_sliced_tensor.at(&idx2));

    var idx3 = [_]usize{ 1, 0 };
    try testing.expectEqual(7, try test_sliced_tensor.at(&idx3));

    var idx4 = [_]usize{ 1, 1 };
    try testing.expectEqual(8, try test_sliced_tensor.at(&idx4));
}

test "Cloning a Tensor" {
    var test_shape = [_]usize{ 3, 2 };
    var test_tensor_data_array = [_]i32{ 0, 1, 2, 3, 4, 5 };
    var test_tensor = try Tensor(i32).initWithSlice(testing.allocator, &test_shape, &test_tensor_data_array);
    defer test_tensor.deinit();

    // Clone the tensor
    var test_tensor_clone = try test_tensor.clone();
    defer test_tensor_clone.deinit();

    // Check that the data arrays are the same (no slicing so this should work)
    try testing.expectEqualSlices(i32, test_tensor.data.unwrap(), test_tensor_clone.data.unwrap());

    // Now set a value on the original, and make sure it doesn't modify the cloned tensor
    var test_index = [_]usize{ 1, 1 };
    try test_tensor.set(&test_index, 100);

    // Enure that the entry is changed in the original...
    try testing.expectEqual(100, try test_tensor.at(&test_index));
    // but not in the clone
    try testing.expectEqual(3, try test_tensor_clone.at(&test_index));
}
