/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonMemoryAccess/BlockPointerAnalysis.h"
#include "PointerAnalysisTestUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::pointer;

namespace {

std::optional<int64_t> integerOf(OpFoldResult value) {
  Attribute attr = dyn_cast<Attribute>(value);
  if (!attr) {
    auto constant = cast<Value>(value).getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return std::nullopt;
    attr = constant.getValue();
  }
  if (auto integer = dyn_cast<IntegerAttr>(attr))
    return integer.getValue().getSExtValue();
  return std::nullopt;
}

TEST(BlockPointerAnalysis, EffectiveCoordinatesUseI64AcrossI32Boundary) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @wide_coordinate(%base: !tt.ptr<i32>) {
        %shape = arith.constant 16 : i64
        %stride = arith.constant 1 : i64
        %max = arith.constant 2147483647 : i32
        %one = arith.constant 1 : i32
        %ptr = tt.make_tensor_ptr %base, [%shape], [%stride], [%max] {order = array<i32: 0>} : !tt.ptr<tensor<4xi32>>
        %next = tt.advance %ptr, [%one] : !tt.ptr<tensor<4xi32>>
        %again = tt.advance %next, [%one] : !tt.ptr<tensor<4xi32>>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  SmallVector<triton::AdvanceOp> advances;
  func.walk([&](triton::AdvanceOp op) { advances.push_back(op); });
  ASSERT_EQ(advances.size(), 2u);
  OpBuilder builder(&context);
  BlockPointerAnalysis analysis(builder);
  auto first = analysis.analyzeBlockPointer(advances[0]);
  auto second = analysis.analyzeBlockPointer(advances[1]);
  ASSERT_TRUE(succeeded(first));
  ASSERT_TRUE(succeeded(second));
  ASSERT_EQ(first->coordinateOffsets.size(), 1u);
  ASSERT_EQ(second->coordinateOffsets.size(), 1u);
  EXPECT_EQ(integerOf(first->coordinateOffsets[0]), 2147483648LL);
  EXPECT_EQ(integerOf(second->coordinateOffsets[0]), 2147483649LL);
  EXPECT_TRUE(
      first->coordinateRanges[0].contains(llvm::APInt(64, 2147483648ULL)));
  EXPECT_TRUE(
      second->coordinateRanges[0].contains(llvm::APInt(64, 2147483649ULL)));
  EXPECT_EQ(second->base, func.getArgument(0));
  EXPECT_EQ(second->blockShape, SmallVector<int64_t>({4}));
  EXPECT_EQ(second->order, SmallVector<int32_t>({0}));
  EXPECT_TRUE(analysis.findCachedBlockAdvanceDelta(advances[0]));
  EXPECT_TRUE(analysis.findCachedBlockPointer(advances[1]));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(BlockPointerAnalysis, AdvanceDeltaDoesNotNeedResolvedPointer) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @delta_only(%unresolved: !tt.ptr<tensor<4xi32>>, %delta: i32) {
        %next = tt.advance %unresolved, [%delta] : !tt.ptr<tensor<4xi32>>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto advance = *func.getBody().front().getOps<triton::AdvanceOp>().begin();
  OpBuilder builder(&context);
  BlockPointerAnalysis analysis(builder);
  auto delta = analysis.analyzeBlockAdvanceDelta(advance);
  ASSERT_TRUE(succeeded(delta));
  ASSERT_EQ(delta->coordinateDeltas.size(), 1u);
  EXPECT_TRUE(isa<Value>(delta->coordinateDeltas[0]));
  EXPECT_TRUE(cast<Value>(delta->coordinateDeltas[0]).getType().isInteger(64));
  EXPECT_EQ(delta->ranges[0].getBitWidth(), 64u);
  EXPECT_TRUE(failed(analysis.analyzeBlockPointer(advance)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(BlockPointerAnalysis, AddressViewUsesCoordinatesAndElementStrides) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @address_view(%base: !tt.ptr<i32>) {
        %m = arith.constant 8 : i64
        %n = arith.constant 12 : i64
        %row_stride = arith.constant 12 : i64
        %one = arith.constant 1 : i64
        %row = arith.constant 2 : i32
        %col = arith.constant 3 : i32
        %ptr = tt.make_tensor_ptr %base, [%m, %n], [%row_stride, %one], [%row, %col] {order = array<i32: 1, 0>} : !tt.ptr<tensor<2x4xi32>>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto make = *func.getBody().front().getOps<triton::MakeTensorPtrOp>().begin();
  auto ret = *func.getBody().front().getOps<triton::ReturnOp>().begin();
  OpBuilder builder(ret);
  BlockPointerAnalysis analysis(builder);
  auto block = analysis.analyzeBlockPointer(make);
  ASSERT_TRUE(succeeded(block));
  EXPECT_EQ(block->parentShape.size(), 2u);
  EXPECT_EQ(block->order, SmallVector<int32_t>({1, 0}));
  auto view = buildAddressView(*block, builder, make.getLoc());
  ASSERT_TRUE(succeeded(view));
  EXPECT_EQ(view->shape, SmallVector<int64_t>({2, 4}));
  EXPECT_EQ(view->domain, ArithmeticDomain::ElementAddress);
  EXPECT_EQ(integerOf(view->uniformOffset), 27);
  EXPECT_EQ(integerOf(view->strides[0]), 12);
  EXPECT_EQ(integerOf(view->strides[1]), 1);
  EXPECT_TRUE(isa<Value>(view->completeOffset));
  EXPECT_EQ(cast<Value>(view->completeOffset).getType(),
            RankedTensorType::get({2, 4}, builder.getI64Type()));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(BlockPointerAnalysis, FullAddressOracleAfterDynamicAndNegativeAdvances) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @block(%base: !tt.ptr<i32>, %shape: i64, %stride: i64, %coord: i32, %delta: i32) {
      %n = arith.constant 12 : i64
      %one = arith.constant 1 : i64
      %col = arith.constant -3 : i32
      %negative = arith.constant -2 : i32
      %positive = arith.constant 4 : i32
      %ptr = tt.make_tensor_ptr %base, [%shape, %n], [%stride, %one], [%coord, %col] {order = array<i32: 1, 0>} : !tt.ptr<tensor<2x4xi32>>
      %a = tt.advance %ptr, [%delta, %negative] : !tt.ptr<tensor<2x4xi32>>
      %b = tt.advance %a, [%positive, %positive] : !tt.ptr<tensor<2x4xi32>>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  SmallVector<Value> pointers;
  f.walk([&](Operation *op) {
    if (isa<triton::MakeTensorPtrOp, triton::AdvanceOp>(op))
      pointers.push_back(op->getResult(0));
  });
  OpBuilder b(f.getBody().front().getTerminator());
  BlockPointerAnalysis analysis(b);
  for (unsigned stage = 0; stage < pointers.size(); ++stage) {
    auto block = analysis.analyzeBlockPointer(pointers[stage]);
    ASSERT_TRUE(succeeded(block));
    auto view = buildAddressView(*block, b, pointers[stage].getLoc());
    ASSERT_TRUE(succeeded(view));
    for (int64_t coordinate : {-2147483648LL, 7LL, 2147483647LL})
      for (int64_t delta : {-3LL, 2LL})
        for (int64_t stride : {-12LL, 12LL}) {
          test::Environment env{
              {f.getArgument(1), {llvm::APInt(64, 100)}},
              {f.getArgument(2), {llvm::APInt(64, stride, true)}},
              {f.getArgument(3), {llvm::APInt(32, coordinate, true)}},
              {f.getArgument(4), {llvm::APInt(32, delta, true)}}};
          llvm::APInt row(64, coordinate, true), col(64, -3, true);
          if (stage >= 1) {
            row += llvm::APInt(64, delta, true);
            col -= 2;
          }
          if (stage >= 2) {
            row += 4;
            col += 4;
          }
          auto lanes = test::evaluate(view->completeOffset, env);
          ASSERT_TRUE(lanes);
          ASSERT_EQ(lanes->size(), 8u);
          auto c = test::evaluate(view->uniformOffset, env);
          ASSERT_TRUE(c);
          EXPECT_EQ(c->front(), row * llvm::APInt(64, stride, true) + col);
          EXPECT_TRUE(block->coordinateRanges[0].contains(row));
          EXPECT_TRUE(block->coordinateRanges[1].contains(col));
          for (unsigned lane = 0; lane < 8; ++lane) {
            auto expected = (row + llvm::APInt(64, lane / 4)) *
                                llvm::APInt(64, stride, true) +
                            col + llvm::APInt(64, lane % 4);
            EXPECT_EQ((*lanes)[lane], expected);
            auto s0 = test::evaluate(view->strides[0], env),
                 s1 = test::evaluate(view->strides[1], env);
            ASSERT_TRUE(s0);
            ASSERT_TRUE(s1);
            EXPECT_EQ(c->front() + llvm::APInt(64, lane / 4) * s0->front() +
                          llvm::APInt(64, lane % 4) * s1->front(),
                      expected);
          }
        }
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

TEST(BlockPointerAnalysis, RebindingInvalidatesDerivedCoordinatesAndRanges) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @bind(%base: !tt.ptr<i32>, %boundary: !tt.ptr<tensor<4xi32>>) {
      %shape = arith.constant 16 : i64
      %stride = arith.constant 1 : i64
      %coord = arith.constant 3 : i32
      %ptr = tt.make_tensor_ptr %base, [%shape], [%stride], [%coord] {order = array<i32: 0>} : !tt.ptr<tensor<4xi32>>
      %next = tt.advance %boundary, [%coord] : !tt.ptr<tensor<4xi32>>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto make = *f.getBody().front().getOps<triton::MakeTensorPtrOp>().begin();
  auto next = *f.getBody().front().getOps<triton::AdvanceOp>().begin();
  OpBuilder b(&context);
  BlockPointerAnalysis analysis(b);
  EXPECT_TRUE(failed(analysis.analyzeBlockPointer(f.getArgument(1))));
  auto binding = analysis.analyzeBlockPointer(make);
  ASSERT_TRUE(succeeded(binding));
  for (int64_t coord : {10, -20}) {
    binding->coordinateOffsets[0] = b.getI64IntegerAttr(coord);
    binding->coordinateRanges[0] =
        llvm::ConstantRange(llvm::APInt(64, coord, true));
    ASSERT_TRUE(
        succeeded(analysis.bindBlockPointer(f.getArgument(1), *binding)));
    auto result = analysis.analyzeBlockPointer(next);
    ASSERT_TRUE(succeeded(result));
    EXPECT_EQ(integerOf(result->coordinateOffsets[0]), coord + 3);
    EXPECT_EQ(result->coordinateRanges[0],
              llvm::ConstantRange(llvm::APInt(64, coord + 3, true)));
    auto bad = *binding;
    bad.coordinateRanges.clear();
    EXPECT_TRUE(failed(analysis.bindBlockPointer(f.getArgument(1), bad)));
    auto retained = analysis.analyzeBlockPointer(next);
    ASSERT_TRUE(succeeded(retained));
    EXPECT_EQ(retained->coordinateRanges, result->coordinateRanges);
  }
  analysis.clear();
  EXPECT_FALSE(analysis.findCachedBlockPointer(next));
  EXPECT_TRUE(failed(analysis.analyzeBlockPointer(next)));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(BlockPointerAnalysis,
     AddressViewRejectsUnsupportedFullShapeBeforeMutation) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, triton::TritonDialect>();
  for (auto shape : {SmallVector<int64_t>{0}, SmallVector<int64_t>{3},
                     SmallVector<int64_t>{1024, 2048},
                     SmallVector<int64_t>{ShapedType::kDynamic},
                     SmallVector<int64_t>{int64_t(1) << 40, int64_t(1) << 40},
                     SmallVector<int64_t>{1024, 1024}}) {
    bool supported = shape == SmallVector<int64_t>({1024, 1024});
    std::string dims, fields, coords, order;
    for (unsigned i = 0; i < shape.size(); ++i) {
      dims +=
          (shape[i] == ShapedType::kDynamic ? "?" : std::to_string(shape[i])) +
          "x";
      if (i) {
        fields += ", ";
        coords += ", ";
        order += ", ";
      }
      fields += "%n";
      coords += "%c";
      order += std::to_string(i);
    }
    auto module = parseSourceString<ModuleOp>(
        "module { tt.func @shape(%base: !tt.ptr<i32>) { "
        "%n = arith.constant 16 : i64 %c = arith.constant 0 : i32 "
        "%p = tt.make_tensor_ptr %base, [" +
            fields + "], [" + fields + "], [" + coords +
            "] {order = array<i32: " + order + ">} : !tt.ptr<tensor<" + dims +
            "i32>> tt.return } }",
        &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    auto f = *module->getOps<triton::FuncOp>().begin();
    auto make = *f.getBody().front().getOps<triton::MakeTensorPtrOp>().begin();
    OpBuilder builder(f.getBody().front().getTerminator());
    BlockPointerAnalysis analysis(builder);
    auto descriptor = analysis.analyzeBlockPointer(make);
    ASSERT_TRUE(succeeded(descriptor));
    ASSERT_TRUE(succeeded(analysis.bindBlockPointer(make, *descriptor)));
    size_t before = f.getBody().front().getOperations().size();
    auto view = buildAddressView(*descriptor, builder, make.getLoc());
    EXPECT_EQ(succeeded(view), supported);
    if (!supported)
      EXPECT_EQ(f.getBody().front().getOperations().size(), before);
    EXPECT_TRUE(succeeded(verify(*module)));
  }
}

} // namespace
