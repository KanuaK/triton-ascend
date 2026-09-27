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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_BLOCK_POINTER_ANALYSIS_H
#define TRITON_ASCEND_MEMORY_ACCESS_BLOCK_POINTER_ANALYSIS_H

#include "TritonMemoryAccess/PointerAnalysis.h"
#include "TritonMemoryAccess/PointerAnalysisTransfer.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/IR/ConstantRange.h"

namespace mlir::triton::pointer {

/// Source descriptor with *effective* i64 coordinates. The original
/// make_tensor_ptr / advance operands remain i32 in their source operations.
struct BlockPointerComponents {
  Value base;
  SmallVector<OpFoldResult> parentShape;
  SmallVector<int64_t> blockShape;
  SmallVector<OpFoldResult> coordinateOffsets;
  SmallVector<llvm::ConstantRange> coordinateRanges;
  SmallVector<OpFoldResult> strides;
  SmallVector<int32_t> order;
  RankedTensorType blockTensorType;
  Type elementType;
  unsigned addressSpace = 0;
};

struct BlockAdvanceDelta {
  SmallVector<OpFoldResult> coordinateDeltas;
  SmallVector<llvm::ConstantRange> ranges;
};

struct BlockCoordinates {
  SmallVector<OpFoldResult> values;
  SmallVector<llvm::ConstantRange> ranges;
};

/// This function only combines supplied fields. It never revisits an advance
/// or pointer producer, so a control-flow adapter can use newly built fields.
FailureOr<BlockCoordinates> advanceBlockOffsets(
    ArrayRef<OpFoldResult> current, ArrayRef<llvm::ConstantRange> currentRanges,
    const BlockAdvanceDelta &delta, OpBuilder &builder, Location loc);

/// Materialize O(lane) = sum((C[d] + lane[d]) * stride[d]), in i64 elements.
/// Requires a static, positive, unencoded shape with a power-of-two element
/// count within Triton's tensor-size limit. Descriptors may be more general.
FailureOr<OffsetComponents>
buildAddressView(const BlockPointerComponents &block, OpBuilder &builder,
                 Location loc);

/// A combined session: basic requests and block requests share one driver.
class BlockPointerAnalysis {
public:
  BlockPointerAnalysis(OpBuilder &builder, AnalysisOptions options = {});
  ~BlockPointerAnalysis();
  BlockPointerAnalysis(BlockPointerAnalysis &&) noexcept;
  BlockPointerAnalysis &operator=(BlockPointerAnalysis &&) noexcept;

  FailureOr<OffsetComponents> analyzeOffset(Value value);
  FailureOr<OffsetComponents> analyzeAddressDelta(Value value);
  FailureOr<PointerComponents> analyzePointer(Value value);
  FailureOr<BlockPointerComponents> analyzeBlockPointer(Value value);
  FailureOr<BlockAdvanceDelta> analyzeBlockAdvanceDelta(triton::AdvanceOp op);

  const OffsetComponents *findCachedOffset(Value value) const;
  const OffsetComponents *findCachedAddressDelta(Value value) const;
  const BlockPointerComponents *findCachedBlockPointer(Value value) const;
  const BlockAdvanceDelta *
  findCachedBlockAdvanceDelta(triton::AdvanceOp op) const;

  LogicalResult bindOffset(Value boundary, const OffsetComponents &components);
  LogicalResult bindPointer(Value boundary,
                            const PointerComponents &components);
  LogicalResult bindBlockPointer(Value boundary,
                                 const BlockPointerComponents &components);
  LogicalResult remapAndForgetScope(Operation *oldScope,
                                    const IRMapping &resultMapping);
  void clear();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

namespace detail {

struct BlockResult {
  BasicResult basic;
  std::optional<BlockPointerComponents> block;
  std::optional<BlockAdvanceDelta> advanceDelta;
};

class BlockPointerRules {
public:
  using Result = BlockResult;
  explicit BlockPointerRules(AnalysisOptions options) : basic(options) {}

  std::optional<Result> resolveBoundary(AnalysisRequest request) const;
  FailureOr<SmallVector<AnalysisRequest>>
  collectInputs(AnalysisRequest request) const;
  FailureOr<Result> transfer(AnalysisRequest request, ArrayRef<Result> inputs,
                             OpBuilder &builder) const;
  LogicalResult
  mapResultValues(Result &result,
                  llvm::function_ref<FailureOr<Value>(Value)> mapper) const;
  AnalysisOptions getOptions() const { return basic.getOptions(); }

private:
  BasicPointerRules basic;
};

} // namespace detail

} // namespace mlir::triton::pointer

#endif // TRITON_ASCEND_MEMORY_ACCESS_BLOCK_POINTER_ANALYSIS_H
