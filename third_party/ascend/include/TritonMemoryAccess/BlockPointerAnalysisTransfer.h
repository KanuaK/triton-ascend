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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_BLOCK_POINTER_ANALYSIS_TRANSFER_H
#define TRITON_ASCEND_MEMORY_ACCESS_BLOCK_POINTER_ANALYSIS_TRANSFER_H

#include "TritonMemoryAccess/BlockPointerAnalysis.h"
#include "TritonMemoryAccess/PointerAnalysisTransfer.h"

// Include this extension interface only when composing block and basic rules.
namespace mlir::triton::pointer {

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

#endif // TRITON_ASCEND_MEMORY_ACCESS_BLOCK_POINTER_ANALYSIS_TRANSFER_H
