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

#ifndef TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_H
#define TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

namespace mlir {
class RewriterBase;
}

namespace mlir::triton::pointer {

enum class AxisKind { Invariant, Structured, Unknown };
enum class ArithmeticDomain { SourceInteger, ElementAddress };

/// The complete offset is always exact. c/strides reconstruct it only when
/// every axis is known; otherwise c and unknown strides are typed zero slots.
struct OffsetComponents {
  Type valueType;
  ArithmeticDomain domain = ArithmeticDomain::SourceInteger;
  SmallVector<int64_t> shape;
  OpFoldResult completeOffset;
  OpFoldResult uniformOffset;
  SmallVector<OpFoldResult> strides;
  SmallVector<AxisKind> axes;
};

struct PointerComponents {
  Value base;
  OffsetComponents offsets;
  Type elementType;
  unsigned addressSpace = 0;
};

struct AnalysisOptions {
  // PR1 address queries and bindings require 64; SourceInteger is independent.
  unsigned addressBitWidth = 64;
  std::optional<unsigned> indexBitWidth;
};

/// A constructive analysis session for one stable source IR and boundary set.
/// It may insert only pure numeric/shape operations. Callers must load Arith,
/// Triton, and Tensor (rank-zero extraction) dialects. Call clear() before an
/// unrelated rewrite or CSE, or remapAndForgetScope() at an equivalent clone.
class PointerAnalysis {
public:
  PointerAnalysis(OpBuilder &builder, AnalysisOptions options = {});
  ~PointerAnalysis();
  PointerAnalysis(PointerAnalysis &&) noexcept;
  PointerAnalysis &operator=(PointerAnalysis &&) noexcept;

  FailureOr<OffsetComponents> analyzeOffset(Value value);
  FailureOr<OffsetComponents> analyzeAddressDelta(Value value);
  FailureOr<PointerComponents> analyzePointer(Value pointer);
  const OffsetComponents *findCachedOffset(Value value) const;
  const OffsetComponents *findCachedAddressDelta(Value value) const;
  LogicalResult bindOffset(Value boundary, const OffsetComponents &components);
  LogicalResult bindPointer(Value boundary,
                            const PointerComponents &components);
  LogicalResult remapAndForgetScope(Operation *oldScope,
                                    const IRMapping &resultMapping);
  void clear();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

/// Erase only trivially-dead numeric/shape producers, never source memory or
/// control-flow operations. Run after the session has stopped using its cache.
void eraseDeadPointerArithmetic(Operation *scope, RewriterBase &rewriter);

} // namespace mlir::triton::pointer

#endif // TRITON_ASCEND_MEMORY_ACCESS_POINTER_ANALYSIS_H
