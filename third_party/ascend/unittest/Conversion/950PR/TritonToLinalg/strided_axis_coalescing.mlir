// RUN: triton-opt %s --triton-to-unstructure='compile-on-910-95=true force-simt-template=true' \
// RUN:                --triton-to-linalg='compile-on-910-95=true' --split-input-file \
// RUN: | FileCheck %s --implicit-check-not="linalg.reduce"

// -----
// FLA backward cumsum variant:
//   total = reduce(v)
//   fwd = cumsum(v)
//   out = total - fwd
// is exclusive reverse cumsum. StridedAxisCoalescing should rewrite it to
// reverse_cumsum(v) - v, removing tt.reduce before the 2D strided-axis lift.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 1 : i32, hacc.coalesce_factor = 8 : i32
// CHECK-LABEL: func.func @strided_axis_exclusive_reverse_cumsum
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [128, 8]
// CHECK-SAME: strides: [8, 1]
// CHECK: call @triton_cumsum_0
// CHECK-SAME: %[[TRUE]]
// CHECK: arith.subf
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_exclusive_reverse_cumsum(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %n: i32) attributes {noinline = false} {
    %zero = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %c8_i64 = arith.constant 8 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %pid_t = tt.get_program_id x : i32
    %pid_h = tt.get_program_id y : i32
    %chunk = arith.divsi %pid_h, %c8_i32 : i32
    %lane = arith.remsi %pid_h, %c8_i32 : i32
    %chunk_n = arith.muli %chunk, %n : i32
    %base_off = arith.muli %chunk_n, %c8_i32 : i32
    %src_base = tt.addptr %arg0, %base_off : !tt.ptr<f16>, i32
    %src = tt.addptr %src_base, %lane : !tt.ptr<f16>, i32
    %t_off = arith.muli %pid_t, %c128_i32 : i32
    %n_i64 = arith.extsi %n : i32 to i64
    %src_block = tt.make_tensor_ptr %src, [%n_i64], [%c8_i64], [%t_off]
        {order = array<i32: 0>} : <tensor<128xf16>>
    %dst_base = tt.addptr %arg1, %base_off : !tt.ptr<f16>, i32
    %dst = tt.addptr %dst_base, %lane : !tt.ptr<f16>, i32
    %dst_block = tt.make_tensor_ptr %dst, [%n_i64], [%c8_i64], [%t_off]
        {order = array<i32: 0>} : <tensor<128xf16>>
    %v_f16 = tt.load %src_block {boundaryCheck = array<i32: 0>, padding = 1 : i32}
        : !tt.ptr<tensor<128xf16>>
    %v = arith.extf %v_f16 : tensor<128xf16> to tensor<128xf32>
    %fwd = "tt.scan"(%v) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<128xf32>) -> tensor<128xf32>
    %total = "tt.reduce"(%v) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<128xf32>) -> f32
    %neg_fwd = arith.subf %zero, %fwd : tensor<128xf32>
    %total_vec = tt.splat %total : f32 -> tensor<128xf32>
    %exclusive_rev = arith.addf %neg_fwd, %total_vec : tensor<128xf32>
    %out = arith.truncf %exclusive_rev : tensor<128xf32> to tensor<128xf16>
    tt.store %dst_block, %out {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<128xf16>>
    tt.return
  }
}

// -----
// FLA forward gate variant:
//   x = block_load(stride=S)
//   bias = load(bias + pid_h % S)
//   gate = 1 - exp(load(gate + pid_h % S))
//   out = cumsum(gate * softplus(x + bias)) * scale
// The lane-dependent scalar loads must be lifted to tensor<S> and broadcast
// across the T axis; otherwise the 1D -> 2D strided-axis coalesce must bail.
// CHECK-LABEL: module attributes {hacc.coalesce_axis = 1 : i32, hacc.coalesce_factor = 8 : i32
// CHECK: func.func private @triton_cumsum_0(tensor<64x8xf32>
// CHECK-LABEL: func.func @strided_axis_lane_scalar_loads
// CHECK: memref.reinterpret_cast
// CHECK-SAME: sizes: [64, 8]
// CHECK-SAME: strides: [8, 1]
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [8], strides: [1]
// CHECK: linalg.broadcast
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [8], strides: [1]
// CHECK: math.exp
// CHECK: linalg.broadcast
// CHECK: call @triton_cumsum_0
// CHECK-SAME: tensor<64x8xf32>
// CHECK: bufferization.materialize_in_destination
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @strided_axis_lane_scalar_loads(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %scale: f32,
      %n: i32) attributes {noinline = false} {
    %c20 = arith.constant dense<2.000000e+01> : tensor<64xf32>
    %c1_vec = arith.constant dense<1.000000e+00> : tensor<64xf32>
    %c1 = arith.constant 1.000000e+00 : f32
    %c8_i64 = arith.constant 8 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %pid_t = tt.get_program_id x : i32
    %pid_h = tt.get_program_id y : i32
    %chunk = arith.divsi %pid_h, %c8_i32 : i32
    %lane = arith.remsi %pid_h, %c8_i32 : i32
    %chunk_n = arith.muli %chunk, %n : i32
    %base_off = arith.muli %chunk_n, %c8_i32 : i32
    %src_base = tt.addptr %arg0, %base_off : !tt.ptr<f32>, i32
    %src = tt.addptr %src_base, %lane : !tt.ptr<f32>, i32
    %t_off = arith.muli %pid_t, %c64_i32 : i32
    %n_i64 = arith.extsi %n : i32 to i64
    %src_block = tt.make_tensor_ptr %src, [%n_i64], [%c8_i64], [%t_off]
        {order = array<i32: 0>} : <tensor<64xf32>>
    %dst_base = tt.addptr %arg3, %base_off : !tt.ptr<f32>, i32
    %dst = tt.addptr %dst_base, %lane : !tt.ptr<f32>, i32
    %dst_block = tt.make_tensor_ptr %dst, [%n_i64], [%c8_i64], [%t_off]
        {order = array<i32: 0>} : <tensor<64xf32>>
    %x = tt.load %src_block {boundaryCheck = array<i32: 0>, padding = 1 : i32}
        : !tt.ptr<tensor<64xf32>>
    %bias_ptr = tt.addptr %arg2, %lane : !tt.ptr<f32>, i32
    %bias = tt.load %bias_ptr : !tt.ptr<f32>
    %bias_vec = tt.splat %bias : f32 -> tensor<64xf32>
    %biased = arith.addf %x, %bias_vec : tensor<64xf32>
    %gate_ptr = tt.addptr %arg1, %lane : !tt.ptr<f32>, i32
    %gate_raw = tt.load %gate_ptr : !tt.ptr<f32>
    %gate_exp = math.exp %gate_raw : f32
    %gate = arith.subf %c1, %gate_exp : f32
    %pred = arith.cmpf olt, %biased, %c20 : tensor<64xf32>
    %exp_biased = math.exp %biased : tensor<64xf32>
    %plus_one = arith.addf %exp_biased, %c1_vec : tensor<64xf32>
    %softplus = math.log %plus_one : tensor<64xf32>
    %selected = arith.select %pred, %softplus, %biased : tensor<64xi1>, tensor<64xf32>
    %gate_vec = tt.splat %gate : f32 -> tensor<64xf32>
    %scan_in = arith.mulf %gate_vec, %selected : tensor<64xf32>
    %scan = "tt.scan"(%scan_in) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.scan.return %sum : f32
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %scale_vec = tt.splat %scale : f32 -> tensor<64xf32>
    %out = arith.mulf %scan, %scale_vec : tensor<64xf32>
    tt.store %dst_block, %out {boundaryCheck = array<i32: 0>} : !tt.ptr<tensor<64xf32>>
    tt.return
  }
}
