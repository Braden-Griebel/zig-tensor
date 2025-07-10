//! Implementation of a Tensor

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const testing = std.testing;

const random = @import("random.zig");
const Distribution = random.Distribution;

/// A multidimensional container
pub fn Tensor(comptime T: type) type {
    return struct {
        // Get the type of the Tensor
        const Self = @This();
        /// The size of the Tensor (i.e. the amount of data it holds)
        size: usize,
        /// The offset of the Tensor in the underlying linear array
        offset: usize,
        /// The stride of the Tensor for each dimension
        stride: []isize,
        /// The dimensionality of the Tensor
        dim: usize,
        /// The shape of the Tensor (the length in each dimension)
        shape: []usize,
        /// The underlying linear data for the Tensor
        data: TensorData(T),
        /// The allocator used to allocate memory for the Tensor
        allocator: Allocator,

        /// Initialize a Tensor of a given `shape` with provided TensorData of length `size`,
        /// this is the base initializer which others pass through
        fn initWithTensorData(allocator: Allocator, shape: anytype, size: usize, data: TensorData(T)) !Self {
            // Get the number of dimensions
            const dim = shape.len;
            // Find the stride
            const stride = try stride_from_shape(shape, allocator);
            // Create the shape arraylist
            var tensor_shape = try allocator.alloc(usize, shape.len);
            inline for (shape, 0..) |s, idx| {
                tensor_shape[idx] = s;
            }
            // Creating the Tensor
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

        /// Initialize a Tensor with a given shape
        pub fn initWithShape(allocator: Allocator, shape: anytype) !Self {
            // Calculate the size from the shape of the Tensor
            var size: usize = 1;
            inline for (shape) |dim| size *= dim;
            // Allocate the data for the Tensor
            const tensor_data = try TensorData(T).initWithSize(allocator, size);
            return try Self.initWithTensorData(allocator, shape, size, tensor_data);
        }

        /// Initialize a Tensor with a given shape, and filled with a value
        pub fn initFilled(allocator: Allocator, shape: anytype, fill_val: T) !Self {
            // Calculate the size from the shape of the Tensor
            var size: usize = 1;
            inline for (shape) |dim| size *= dim;
            // Allocate the data for the Tensor
            var tensor_data = try TensorData(T).initWithSize(allocator, size);
            // Fill the Tensor data with the specified value
            tensor_data.fill(fill_val);
            return Self.initWithTensorData(allocator, shape, size, tensor_data);
        }

        /// Initialize a Tensor with a given shape, with data from a slice
        pub fn initWithSlice(allocator: Allocator, shape: anytype, in_data: []T) !Self {
            // Calculate the size from the shape of the Tensor
            var size: usize = 1;
            inline for (shape) |dim| size *= dim;
            // Allocate the data for the Tensor
            const tensor_data = try TensorData(T).initWithSlice(allocator, in_data);
            return Self.initWithTensorData(allocator, shape, size, tensor_data);
        }

        /// Free memory associated with a Tensor
        fn deinit(self: *Self) void {
            // Free the underlying items
            self.data.deinit();
            // Deinit the stride
            self.allocator.free(self.stride);
            // Deinit the dimsize
            self.allocator.free(self.shape);
            // Invalidate the reference
            self.* = undefined;
        }

        /// Fill a Tensor with a particular value
        pub fn fill(self: *Self, val: T) void {
            self.data.fill(val);
        }

        /// Get a value from a position in the tensor
        /// The index must be a tuple with length equal to the
        /// dimension of the Tensor
        pub fn get(self: *Self, index: anytype) TensorError!T {
            const to_get = try self.at(index);
            return to_get.*;
        }

        /// Access an element of the Tensor at the position
        /// specified by index, which must be a tuple with
        /// length equal to the dimensionality of the Tensor
        pub fn at(self: *Self, index: anytype) TensorError!*T {
            switch (@typeInfo(@TypeOf(index))) {
                .pointer => {
                    if (index.len != self.dim) {
                        return TensorError.InvalidIndex;
                    }
                    // Calculate the position of the desired entry
                    var position: isize = @intCast(self.offset);
                    for (index, 0..) |location, stride_index| {
                        position += @as(isize, @intCast(location)) * self.stride[stride_index];
                    }
                    // Get the data slice
                    const data: []T = self.data.unwrap();
                    // Bounds check
                    if (position >= data.len) {
                        return TensorError.InvalidIndex;
                    }
                    // Return a pointer to the desired item
                    return &data[@as(usize, @intCast(position))];
                },
                .@"struct" => {
                    // Check that the input index is valid
                    if (index.len != self.dim) {
                        return TensorError.InvalidIndex;
                    }
                    // Calculate the position of the desired entry
                    var position: isize = @intCast(self.offset);
                    inline for (index, 0..) |location, stride_index| {
                        position += @as(isize, @intCast(location)) * self.stride[stride_index];
                    }
                    // Get the data slice
                    const data: []T = self.data.unwrap();
                    // Bounds check
                    if (position >= data.len) {
                        return TensorError.InvalidIndex;
                    }
                    // Return a pointer to the desired item
                    return &data[@as(usize, @intCast(position))];
                },
                else => {
                    return TensorError.InvalidIndexType;
                },
            }
        }

        /// Set a value at a position in the tensor
        /// the index must be a tuple of the same length as the dimensionality
        /// of the Tensor
        fn set(self: *Self, index: anytype, value: T) TensorError!void {
            const to_set = try self.at(index);
            to_set.* = value;
        }

        /// Get a slice from the Tensor
        ///
        /// Note that this Tensor will reference the same data as the parent tensor,
        /// and will have the same dimensionality regardless of the slices
        pub fn slice(self: *Self, indices: []TensorSlice) TensorError!Self {
            if (indices.len != self.dim) {
                return TensorError.InvalidDimensions; // Too many slices
            }
            // The new size will be the product of stop-start/step for each dimension
            var new_size: usize = 1;
            // The new offset will be the old offset + stride*start for each slice
            var new_offset: usize = self.offset;
            // The new strides will be old_stride * step for each dimension
            var new_stride: []isize = try self.allocator.alloc(isize, self.dim); // The new dimensionality (same as previous dimensionality)
            const new_dim = self.dim;
            // The new shape will be the stop-start/step for each dimension
            var new_shape = try self.allocator.alloc(usize, self.dim);
            // Get a new data reference
            const new_data = self.data.clone();
            // The allocator will be the same
            const new_allocator = self.allocator;

            // Iterate through the slices to update the values
            for (indices, 0..) |s, idx| {
                const slice_start = s.start orelse 0;
                const slice_step = s.step orelse 1;
                const slice_stop = s.stop orelse @as(isize, @intCast(self.shape[idx]));
                const dim_size = @as(usize, @intCast(@divFloor((slice_stop - slice_start), slice_step)));
                // Update the size
                new_size *= dim_size;
                // Update the shape
                new_shape[idx] = dim_size;
                // Update the stride
                new_stride[idx] = slice_step * self.stride[idx];
                // Update the offset
                new_offset += @as(usize, @intCast(self.stride[idx] * slice_start));
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
        /// view prior to using this method.
        pub fn reshape(self: *Self, new_shape: []usize) TensorError!T {
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
        pub fn clone(self: *Self) !Self {
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
            const new_tensor_stride = try stride_from_shape(self.shape, self.allocator);

            return Self{
                .size = self.size,
                .offset = 0,
                .stride = new_tensor_stride,
                .dim = self.dim,
                .shape = try self.allocator.dupe(usize, self.shape),
                .data = new_tensor_data,
                .allocator = self.allocator,
            };
        }

        /// Get an iterator over the Tensor's indices
        pub fn getIndexIter(self: *Self) !TensorIndexIter(T) {
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
                .tensor_strides = tensor.stride,
                .tensor_shape = tensor.shape,
                .tensor_offset = tensor.offset,
                .current_tensor_index = current_tensor_index,
                .current_data_index = tensor.offset,
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
    /// Attempted to access a position with
    /// an index of invalid type
    InvalidIndexType,
    /// Attempted to slice with invalid slices
    InvalidSlice,
    /// Attempted to slice with incorrect number of dimensions
    InvalidDimensions,
    /// Attempted to use a shape that is invalid (for reshaping, or creating a Tensor)
    InvalidShape,
    /// Attempted to use a shape with an invalid type (must be a struct/tuple or slice)
    InvalidShapeType,
    /// If an invalid type is used for initialization (i.e.
    /// trying to create a random normal of ints)
    InvalidType,
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

        /// Create a Tensor data instance with the `data_slice` as the backing memory,
        /// which will be freed using the passed allocator
        fn initWithSliceTake(allocator: Allocator, data_slice: []T) !TensorData(T) {
            const rc = try allocator.create(usize);
            rc.* = 1;
            return .{
                .data = data_slice,
                .allocator = allocator,
                .ref_count = rc,
            };
        }

        /// Create a Tensor data value filled from a passed slice
        fn initWithSlice(allocator: Allocator, data_slice: []T) !TensorData(T) {
            const data_slice_copy = try allocator.dupe(T, data_slice);
            return TensorData(T).initWithSliceTake(allocator, data_slice_copy);
        }

        /// Initialize a Tensor data value with some size from a random distribution
        fn initRandom(allocator: Allocator, size: usize, dist: Distribution(T)) !TensorData(T) {
            const random_sample: ArrayList(T) = try dist.getRvs(allocator, size);
            defer random_sample.deinit();
            const random_data: []T = try random_sample.toOwnedSlice();
            return try TensorData(T).initWithSliceTake(allocator, random_data);
        }

        /// Initialize a Tensor data value with some size from a random normal distribution
        fn initRandomNormal(allocator: Allocator, size: usize, loc: T, scale: T) !TensorData(T) {
            switch (@typeInfo(T)) {
                .float => {
                    var prng = std.Random.DefaultPrng.init(blk: {
                        var seed: u64 = undefined;
                        try std.posix.getrandom(std.mem.asBytes(&seed));
                        break :blk seed;
                    });
                    const generator = prng.random();
                    var sampling_distribution = try random.DistNormal(T).init(.{ .generator = generator, .loc = loc, .scale = scale });
                    const sampler: Distribution(T) = sampling_distribution.distribution();
                    return try TensorData(T).initRandom(allocator, size, sampler);
                },
                else => {
                    return TensorError.InvalidType;
                },
            }
        }

        /// Initialize a Tensor data value with some size from a random uniform distribution
        fn initRandomUniform(allocator: Allocator, size: usize, loc: T, scale: T) !TensorData(T) {
            var prng = std.Random.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                try std.posix.getrandom(std.mem.asBytes(&seed));
                break :blk seed;
            });
            const generator = prng.random();
            var sampling_distribution = try random.DistUniform(T).init(.{ .generator = generator, .loc = loc, .scale = scale });
            const sampler = sampling_distribution.distribution();
            return try TensorData(T).initRandom(allocator, size, sampler);
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

/// A slice of a Tensor in a single dimension
const TensorSlice = struct {
    /// The start of the slice
    start: ?isize = null,
    /// The stop position of the slice,
    /// slice will include everything up
    /// to (but not including) this position
    stop: ?isize = null,
    /// Step size of the slice
    step: ?isize = null,
};

// Helper Functions
/// Determine the stride of the Tensor from a stride tuple
fn stride_from_shape(shape: anytype, allocator: Allocator) ![]isize {
    switch (@typeInfo(@TypeOf(shape))) {
        .pointer => { // Pointers are treated as slices
            const stride: []isize = try allocator.alloc(isize, shape.len);

            if (shape.len == 0) {
                return stride;
            }
            // Move the values from the shape into the stride
            for (shape, 0..) |s, stride_index| {
                stride[stride_index] = @as(isize, @intCast(s));
            }
            // Iterate through the stride slice in reverse,
            // updating the values to create the correct stride
            var accumulator: isize = 1;
            var stride_index = shape.len;
            var tmp: isize = undefined;
            while (stride_index > 0) {
                stride_index -= 1;
                tmp = stride[stride_index];
                stride[stride_index] = accumulator;
                accumulator *= tmp;
            }
            return stride;
        },
        .@"struct" => { // Anonymous struct, requires an inline for loop
            const stride: []isize = try allocator.alloc(isize, shape.len);

            if (shape.len == 0) {
                return stride;
            }
            // Move the values from the shape into the stride
            inline for (shape, 0..) |s, stride_index| {
                stride[stride_index] = @as(isize, @intCast(s));
            }
            // Iterate through the stride slice in reverse,
            // updating the values to create the correct stride
            var accumulator: isize = 1;
            var stride_index = shape.len;
            var tmp: isize = undefined;
            while (stride_index > 0) {
                stride_index -= 1;
                tmp = stride[stride_index];
                stride[stride_index] = accumulator;
                accumulator *= tmp;
            }
            return stride;
        },
        else => {
            return TensorError.InvalidShapeType;
        },
    }
}

// TESTS
test "Getting stride from shape (0D)" {
    const test_stride = try stride_from_shape(.{}, testing.allocator);
    defer testing.allocator.free(test_stride);

    var expected_stride_array = [_]isize{};
    var expected_stride = ArrayList(isize).init(testing.allocator);
    defer expected_stride.deinit();
    try expected_stride.appendSlice(&expected_stride_array);

    try testing.expect(std.mem.eql(isize, expected_stride.items, test_stride));
}

test "Getting stride from shape (1D)" {
    const test_stride = try stride_from_shape(.{5}, testing.allocator);
    defer testing.allocator.free(test_stride);

    const expected_stride = [_]isize{1};

    try testing.expect(std.mem.eql(isize, &expected_stride, test_stride));
}

test "Getting stride from shape (3D)" {
    const test_stride = try stride_from_shape(.{ 2, 3, 5 }, testing.allocator);
    defer testing.allocator.free(test_stride);

    var expected_stride_array = [_]isize{ 15, 5, 1 };
    var expected_stride = ArrayList(isize).init(testing.allocator);
    defer expected_stride.deinit();
    try expected_stride.appendSlice(&expected_stride_array);

    try testing.expect(std.mem.eql(isize, expected_stride.items, test_stride));
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
        try testing.expectEqual(@as(i32, @intCast(idx)), test_tensor.get(.{ @divFloor(idx, 2), idx % 2 }));
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
            try testing.expectEqual(test_tensor.get(.{ row, col }), 0);
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
    try test_tensor.set(.{ 3, 1 }, 5);

    // Check that the index was indeed updated
    try testing.expectEqual(5, test_tensor.get(.{ 3, 1 }));
}

test "Slicing a Tensor" {
    var test_shape = [_]usize{ 3, 3 };
    var test_tensor_data = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    var test_tensor = try Tensor(i32).initWithSlice(testing.allocator, &test_shape, &test_tensor_data);
    defer test_tensor.deinit();

    var test_slices = [_]TensorSlice{ .{ .start = 1 }, .{ .start = 1 } };
    var test_sliced_tensor = try test_tensor.slice(&test_slices);
    defer test_sliced_tensor.deinit();

    try testing.expectEqual(4, try test_sliced_tensor.get(.{ 0, 0 }));

    try testing.expectEqual(5, try test_sliced_tensor.get(.{ 0, 1 }));

    try testing.expectEqual(7, try test_sliced_tensor.get(.{ 1, 0 }));

    try testing.expectEqual(8, try test_sliced_tensor.get(.{ 1, 1 }));
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
    try test_tensor.set(.{ 1, 1 }, 100);

    // Enure that the entry is changed in the original...
    try testing.expectEqual(100, try test_tensor.get(.{ 1, 1 }));
    // but not in the clone
    try testing.expectEqual(3, try test_tensor_clone.get(.{ 1, 1 }));
}

test "Cloning a Sliced Tensor" {
    var test_shape = [_]usize{ 3, 2 };
    var test_tensor_data_array = [_]i32{ 0, 1, 2, 3, 4, 5 };
    var test_tensor = try Tensor(i32).initWithSlice(testing.allocator, &test_shape, &test_tensor_data_array);
    defer test_tensor.deinit();

    // Slice the tensor
    var test_slices = [_]TensorSlice{ .{ .start = 1 }, .{} };
    var test_tensor_sliced = try test_tensor.slice(&test_slices);
    defer test_tensor_sliced.deinit();

    // Clone the sliced tensor
    var test_tensor_cloned = try test_tensor_sliced.clone();
    defer test_tensor_cloned.deinit();

    // Check the expected values
    try testing.expectEqual(2, test_tensor_cloned.get(.{ 0, 0 }));

    try testing.expectEqual(3, test_tensor_cloned.get(.{ 0, 1 }));

    try testing.expectEqual(4, test_tensor_cloned.get(.{ 1, 0 }));

    try testing.expectEqual(5, test_tensor_cloned.get(.{ 1, 1 }));
}
