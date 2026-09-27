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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_TRANSFER_H
#define TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_TRANSFER_H

#include "TritonMemoryAccess/PointerAnalysis.h"
#include "TritonMemoryAccess/PointerAnalysisDriver.h"
#include "llvm/IR/ConstantRange.h"

namespace mlir::triton::pointer::detail {

/// Only source-integer requests have a range. Unknown producers get the full
/// source-width set, never a guessed interval or a zero-value proof.
struct BasicResult {
  std::optional<OffsetComponents> offset;
  std::optional<PointerComponents> pointer;
  std::optional<llvm::ConstantRange> integerRange;
};

/// Nonrecursive rules. A caller can combine this with extra result fields and
/// operand requests while keeping AnalysisDriver as the only traversal.
class BasicPointerRules {
public:
  using Result = BasicResult;

  explicit BasicPointerRules(AnalysisOptions options) : options(options) {}

  std::optional<Result> resolveBoundary(AnalysisRequest request) const;
  FailureOr<SmallVector<AnalysisRequest>>
  collectInputs(AnalysisRequest request) const;
  FailureOr<Result> transfer(AnalysisRequest request, ArrayRef<Result> inputs,
                             OpBuilder &builder) const;
  LogicalResult
  mapResultValues(Result &result,
                  llvm::function_ref<FailureOr<Value>(Value)> mapper) const;

  AnalysisOptions getOptions() const { return options; }

private:
  AnalysisOptions options;
};

const OffsetComponents *getBasicOffset(const BasicResult &result);
const PointerComponents *getBasicPointer(const BasicResult &result);
bool validOffsetBinding(Value boundary, const OffsetComponents &components,
                        ArithmeticDomain expectedDomain,
                        AnalysisOptions options);
bool validPointerBinding(Value boundary, const PointerComponents &components,
                         AnalysisOptions options);

} // namespace mlir::triton::pointer::detail

#endif // TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_TRANSFER_H
