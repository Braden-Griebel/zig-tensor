const std = @import("std");

pub fn build(b: *std.Build) void {
    // Get the target
    const target = b.standardTargetOptions(.{});

    // Add a module (in this case, since its a library this is all that's needed)
    const mod = b.addModule("zig_tensor", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    // Add Tests for the module
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the executable
    const run_mod_tests = b.addRunArtifact(mod_tests);

    // A run step that will run the test executable
    const test_step = b.step("test", "Run tests");
    // Mark that the 'zig run test' requires the test executable
    test_step.dependOn(&run_mod_tests.step);
}
