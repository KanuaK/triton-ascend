// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @map_elementwise_add_mul
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = {{.*}} iter_args(%[[OUT:.*]] = %[[EMPTY]]) -> (tensor<8xf32>) {
// CHECK: %[[X:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<8xf32>
// CHECK: %[[Y:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<8xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[X]], %[[Y]] : f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[Y]] : f32
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[MUL]] into %[[OUT]][%[[IV]]] : tensor<8xf32>
// CHECK: scf.yield %[[INSERTED]] : tensor<8xf32>
// CHECK: return %[[LOOP]] : tensor<8xf32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @map_elementwise_add_mul(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    %0 = "tt.map_elementwise"(%arg0, %arg1) ({
    ^bb0(%x: f32, %y: f32):
      %add = arith.addf %x, %y : f32
      %mul = arith.mulf %add, %y : f32
      tt.map_elementwise.return %mul : f32
    }) {pack = 1 : i32} : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @map_elementwise_scalar_if
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xf32>
// CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = {{.*}} iter_args(%[[OUT:.*]] = %[[EMPTY]]) -> (tensor<8xf32>) {
// CHECK: %[[X:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<8xf32>
// CHECK: %[[PRED:.*]] = arith.cmpf ogt, %[[X]], %[[ZERO]] : f32
// CHECK: %[[IF:.*]] = scf.if %[[PRED]] -> (f32) {
// CHECK: scf.yield %[[X]] : f32
// CHECK: } else {
// CHECK: %[[NEG:.*]] = arith.subf %[[ZERO]], %[[X]] : f32
// CHECK: scf.yield %[[NEG]] : f32
// CHECK: }
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[IF]] into %[[OUT]][%[[IV]]] : tensor<8xf32>
// CHECK: scf.yield %[[INSERTED]] : tensor<8xf32>
// CHECK: return %[[LOOP]] : tensor<8xf32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @map_elementwise_scalar_if(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %0 = "tt.map_elementwise"(%arg0) ({
    ^bb0(%x: f32):
      %zero = arith.constant 0.000000e+00 : f32
      %pred = arith.cmpf ogt, %x, %zero : f32
      %abs = scf.if %pred -> (f32) {
        scf.yield %x : f32
      } else {
        %neg = arith.subf %zero, %x : f32
        scf.yield %neg : f32
      }
      tt.map_elementwise.return %abs : f32
    }) {pack = 1 : i32} : (tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// -----

// CHECK-LABEL: func.func @map_elementwise_cf_sign
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xi32>
// CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = {{.*}} iter_args(%[[OUT:.*]] = %[[EMPTY]]) -> (tensor<8xi32>) {
// CHECK: %[[X:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<8xi16>
// CHECK: %[[X_I32:.*]] = arith.extsi %[[X]] : i16 to i32
// CHECK: %[[IS_NEG:.*]] = arith.cmpi slt, %[[X_I32]], %{{.*}} : i32
// CHECK: %[[SIGN:.*]] = scf.if %[[IS_NEG]] -> (i32) {
// CHECK: scf.yield %{{.*}} : i32
// CHECK: } else {
// CHECK: %[[IS_POS:.*]] = arith.cmpi sgt, %[[X_I32]], %{{.*}} : i32
// CHECK: %[[SIGN_POS:.*]] = arith.extui %[[IS_POS]] : i1 to i32
// CHECK: scf.yield %[[SIGN_POS]] : i32
// CHECK: }
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[SIGN]] into %[[OUT]][%[[IV]]] : tensor<8xi32>
// CHECK: scf.yield %[[INSERTED]] : tensor<8xi32>
// CHECK: return %[[LOOP]] : tensor<8xi32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @map_elementwise_cf_sign(%arg0: tensor<8xi16>) -> tensor<8xi32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c-1 = arith.constant -1 : i32
    %0 = "tt.map_elementwise"(%arg0) ({
    ^bb0(%x: i16):
      %x_i32 = arith.extsi %x : i16 to i32
      %is_neg = arith.cmpi slt, %x_i32, %c0 : i32
      cf.cond_br %is_neg, ^bb2(%c-1 : i32), ^bb1
    ^bb1:
      %is_pos = arith.cmpi sgt, %x_i32, %c0 : i32
      cf.cond_br %is_pos, ^bb2(%c1 : i32), ^bb2(%c0 : i32)
    ^bb2(%ret: i32):
      tt.map_elementwise.return %ret : i32
    }) {pack = 1 : i32} : (tensor<8xi16>) -> tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}

// -----

// CHECK-LABEL: func.func @map_elementwise_add_2d
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK: %[[OUTER:.*]] = scf.for %[[I:.*]] = {{.*}} iter_args(%[[OUTER_ARG:.*]] = %[[EMPTY]]) -> (tensor<2x4xf32>) {
// CHECK: %[[INNER:.*]] = scf.for %[[J:.*]] = {{.*}} iter_args(%[[INNER_ARG:.*]] = %[[OUTER_ARG]]) -> (tensor<2x4xf32>) {
// CHECK: %[[X:.*]] = tensor.extract %{{.*}}[%[[I]], %[[J]]] : tensor<2x4xf32>
// CHECK: %[[Y:.*]] = tensor.extract %{{.*}}[%[[I]], %[[J]]] : tensor<2x4xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[X]], %[[Y]] : f32
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[ADD]] into %[[INNER_ARG]][%[[I]], %[[J]]] : tensor<2x4xf32>
// CHECK: scf.yield %[[INSERTED]] : tensor<2x4xf32>
// CHECK: scf.yield %[[INNER]] : tensor<2x4xf32>
// CHECK: return %[[OUTER]] : tensor<2x4xf32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @map_elementwise_add_2d(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "tt.map_elementwise"(%arg0, %arg1) ({
    ^bb0(%x: f32, %y: f32):
      %add = arith.addf %x, %y : f32
      tt.map_elementwise.return %add : f32
    }) {pack = 1 : i32} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

// -----

// CHECK-LABEL: func.func @map_elementwise_divmod
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xi32>
// CHECK: %[[LOOP:.*]]:2 = scf.for %[[IV:.*]] = {{.*}} iter_args(%[[OUT0:.*]] = %[[EMPTY]], %[[OUT1:.*]] = %[[EMPTY]]) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK: %[[A:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<8xi32>
// CHECK: %[[B:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<8xi32>
// CHECK: %[[Q:.*]] = arith.divsi %[[A]], %[[B]] : i32
// CHECK: %[[R:.*]] = arith.remsi %[[A]], %[[B]] : i32
// CHECK: %[[INSERTED_Q:.*]] = tensor.insert %[[Q]] into %[[OUT0]][%[[IV]]] : tensor<8xi32>
// CHECK: %[[INSERTED_R:.*]] = tensor.insert %[[R]] into %[[OUT1]][%[[IV]]] : tensor<8xi32>
// CHECK: scf.yield %[[INSERTED_Q]], %[[INSERTED_R]] : tensor<8xi32>, tensor<8xi32>
// CHECK: return %[[LOOP]]#0, %[[LOOP]]#1 : tensor<8xi32>, tensor<8xi32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @map_elementwise_divmod(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %q, %r = "tt.map_elementwise"(%arg0, %arg1) ({
    ^bb0(%a: i32, %b: i32):
      %q0 = arith.divsi %a, %b : i32
      %r0 = arith.remsi %a, %b : i32
      tt.map_elementwise.return %q0, %r0 : i32, i32
    }) {pack = 1 : i32} : (tensor<8xi32>, tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>)
    return %q, %r : tensor<8xi32>, tensor<8xi32>
  }
}

// -----

// CHECK-LABEL: func.func @map_elementwise_pack2_divmod
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<8xi32>
// CHECK: %[[LOOP:.*]]:2 = scf.for %[[IV:.*]] = {{.*}} step {{.*}} iter_args(%[[OUT0:.*]] = %[[EMPTY]], %[[OUT1:.*]] = %[[EMPTY]]) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK: %[[IV1:.*]] = arith.addi %[[IV]], {{.*}} : index
// CHECK: %[[A0:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<8xi32>
// CHECK: %[[A1:.*]] = tensor.extract %{{.*}}[%[[IV1]]] : tensor<8xi32>
// CHECK: %[[B0:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<8xi32>
// CHECK: %[[B1:.*]] = tensor.extract %{{.*}}[%[[IV1]]] : tensor<8xi32>
// CHECK: %[[Q0:.*]] = arith.divsi %[[A0]], %[[B0]] : i32
// CHECK: %[[Q1:.*]] = arith.divsi %[[A1]], %[[B1]] : i32
// CHECK: %[[R0:.*]] = arith.remsi %[[A0]], %[[B0]] : i32
// CHECK: %[[R1:.*]] = arith.remsi %[[A1]], %[[B1]] : i32
// CHECK: %[[OUT0_0:.*]] = tensor.insert %[[Q0]] into %[[OUT0]][%[[IV]]] : tensor<8xi32>
// CHECK: %[[OUT0_1:.*]] = tensor.insert %[[Q1]] into %[[OUT0_0]][%[[IV1]]] : tensor<8xi32>
// CHECK: %[[OUT1_0:.*]] = tensor.insert %[[R0]] into %[[OUT1]][%[[IV]]] : tensor<8xi32>
// CHECK: %[[OUT1_1:.*]] = tensor.insert %[[R1]] into %[[OUT1_0]][%[[IV1]]] : tensor<8xi32>
// CHECK: scf.yield %[[OUT0_1]], %[[OUT1_1]] : tensor<8xi32>, tensor<8xi32>
// CHECK: return %[[LOOP]]#0, %[[LOOP]]#1 : tensor<8xi32>, tensor<8xi32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @map_elementwise_pack2_divmod(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %q, %r = "tt.map_elementwise"(%arg0, %arg1) ({
    ^bb0(%a0: i32, %a1: i32, %b0: i32, %b1: i32):
      %q0 = arith.divsi %a0, %b0 : i32
      %q1 = arith.divsi %a1, %b1 : i32
      %r0 = arith.remsi %a0, %b0 : i32
      %r1 = arith.remsi %a1, %b1 : i32
      tt.map_elementwise.return %q0, %q1, %r0, %r1 : i32, i32, i32, i32
    }) {pack = 2 : i32} : (tensor<8xi32>, tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>)
    return %q, %r : tensor<8xi32>, tensor<8xi32>
  }
}

// -----

// CHECK-LABEL: func.func @map_elementwise_pack2_add_2d
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x4xf32>
// CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = {{.*}} step {{.*}} iter_args(%[[OUT:.*]] = %[[EMPTY]]) -> (tensor<2x4xf32>) {
// CHECK: %[[J0:.*]] = arith.remui %[[IV]], {{.*}} : index
// CHECK: %[[I0:.*]] = arith.divui %[[IV]], {{.*}} : index
// CHECK: %[[IV1:.*]] = arith.addi %[[IV]], {{.*}} : index
// CHECK: %[[J1:.*]] = arith.remui %[[IV1]], {{.*}} : index
// CHECK: %[[I1:.*]] = arith.divui %[[IV1]], {{.*}} : index
// CHECK: %[[X0:.*]] = tensor.extract %{{.*}}[%[[I0]], %[[J0]]] : tensor<2x4xf32>
// CHECK: %[[X1:.*]] = tensor.extract %{{.*}}[%[[I1]], %[[J1]]] : tensor<2x4xf32>
// CHECK: %[[Y0:.*]] = tensor.extract %{{.*}}[%[[I0]], %[[J0]]] : tensor<2x4xf32>
// CHECK: %[[Y1:.*]] = tensor.extract %{{.*}}[%[[I1]], %[[J1]]] : tensor<2x4xf32>
// CHECK: %[[ADD0:.*]] = arith.addf %[[X0]], %[[Y0]] : f32
// CHECK: %[[ADD1:.*]] = arith.addf %[[X1]], %[[Y1]] : f32
// CHECK: %[[OUT0:.*]] = tensor.insert %[[ADD0]] into %[[OUT]][%[[I0]], %[[J0]]] : tensor<2x4xf32>
// CHECK: %[[OUT1:.*]] = tensor.insert %[[ADD1]] into %[[OUT0]][%[[I1]], %[[J1]]] : tensor<2x4xf32>
// CHECK: scf.yield %[[OUT1]] : tensor<2x4xf32>
// CHECK: return %[[LOOP]] : tensor<2x4xf32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @map_elementwise_pack2_add_2d(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = "tt.map_elementwise"(%arg0, %arg1) ({
    ^bb0(%x0: f32, %x1: f32, %y0: f32, %y1: f32):
      %add0 = arith.addf %x0, %y0 : f32
      %add1 = arith.addf %x1, %y1 : f32
      tt.map_elementwise.return %add0, %add1 : f32, f32
    }) {pack = 2 : i32} : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}


// -----

// CHECK-LABEL: func.func @map_elementwise_dynamic_add
// CHECK: %[[DIM:.*]] = tensor.dim %{{.*}}, %c0 : tensor<?xi32>
// CHECK: %[[EMPTY:.*]] = tensor.empty(%[[DIM]]) : tensor<?xi32>
// CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = {{.*}} to %[[DIM]] step {{.*}} iter_args(%[[OUT:.*]] = %[[EMPTY]]) -> (tensor<?xi32>) {
// CHECK: %[[X:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<?xi32>
// CHECK: %[[Y:.*]] = tensor.extract %{{.*}}[%[[IV]]] : tensor<?xi32>
// CHECK: %[[ADD:.*]] = arith.addi %[[X]], %[[Y]] : i32
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[ADD]] into %[[OUT]][%[[IV]]] : tensor<?xi32>
// CHECK: scf.yield %[[INSERTED]] : tensor<?xi32>
// CHECK: return %[[LOOP]] : tensor<?xi32>
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  func.func @map_elementwise_dynamic_add(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tt.map_elementwise"(%arg0, %arg1) ({
    ^bb0(%x: i32, %y: i32):
      %add = arith.addi %x, %y : i32
      tt.map_elementwise.return %add : i32
    }) {pack = 1 : i32} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}
