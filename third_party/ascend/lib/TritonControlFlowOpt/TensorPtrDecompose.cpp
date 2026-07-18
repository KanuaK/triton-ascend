/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "TritonControlFlowOpt/TensorPtrDecompose.h"

#include "TritonControlFlowOpt/ControlFlowRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::controlflow;

namespace {

static bool isTensorPointerType(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

/// Tensor-pointer state used only by this policy:
///
///   control-flow components = [complete_offsets]
///   rewrite-only invariants = [common_base]
///   attributes = [base_is_scalar]
///
/// Only `components` expand an SCF signature. The common base is deliberately
/// kept out of iter-args and results; it must be identical at every incoming
/// edge and is used to rebuild the tensor-of-pointers inside and after the
/// rewritten control-flow operation.
static bool hasValidLayout(const DecomposedValue &value) {
  return isTensorPointerType(value.originalType) &&
         value.components.size() == 1 && value.invariants.size() == 1 &&
         value.attributes.size() == 1 &&
         isa<BoolAttr>(value.attributes.front());
}

static Value createZeroLike(OpBuilder &builder, Location loc, Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    auto elementType = dyn_cast<IntegerType>(tensorType.getElementType());
    if (!elementType)
      return nullptr;
    auto attr = DenseElementsAttr::get(tensorType,
                                       builder.getIntegerAttr(elementType, 0));
    return builder.create<arith::ConstantOp>(loc, attr);
  }
  if (type.isIndex())
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  if (auto integerType = dyn_cast<IntegerType>(type))
    return builder.create<arith::ConstantIntOp>(loc, 0,
                                                integerType.getWidth());
  return nullptr;
}

static Value createZeroOffsets(OpBuilder &builder, Location loc,
                               Type pointerType) {
  auto tensorType = cast<RankedTensorType>(pointerType);
  Type offsetsType =
      RankedTensorType::get(tensorType.getShape(), builder.getI32Type());
  return createZeroLike(builder, loc, offsetsType);
}

static Value castIntegerLike(OpBuilder &builder, Location loc, Value value,
                             Type targetType) {
  if (!value || value.getType() == targetType)
    return value;

  Type sourceType = value.getType();
  if ((sourceType.isIndex() && isa<IntegerType>(targetType)) ||
      (isa<IntegerType>(sourceType) && targetType.isIndex()))
    return builder.create<arith::IndexCastOp>(loc, targetType, value);

  auto sourceInt = dyn_cast<IntegerType>(sourceType);
  auto targetInt = dyn_cast<IntegerType>(targetType);
  if (sourceInt && targetInt) {
    if (sourceInt.getWidth() < targetInt.getWidth())
      return builder.create<arith::ExtSIOp>(loc, targetType, value);
    if (sourceInt.getWidth() > targetInt.getWidth())
      return builder.create<arith::TruncIOp>(loc, targetType, value);
    return value;
  }

  auto sourceTensor = dyn_cast<RankedTensorType>(sourceType);
  auto targetTensor = dyn_cast<RankedTensorType>(targetType);
  if (!sourceTensor || !targetTensor ||
      sourceTensor.getShape() != targetTensor.getShape())
    return nullptr;
  auto sourceElement = dyn_cast<IntegerType>(sourceTensor.getElementType());
  auto targetElement = dyn_cast<IntegerType>(targetTensor.getElementType());
  if (!sourceElement || !targetElement)
    return nullptr;
  if (sourceElement.getWidth() < targetElement.getWidth())
    return builder.create<arith::ExtSIOp>(loc, targetType, value);
  if (sourceElement.getWidth() > targetElement.getWidth())
    return builder.create<arith::TruncIOp>(loc, targetType, value);
  return value;
}

static FailureOr<Type> getWiderIntegerLikeType(Type lhs, Type rhs) {
  if (lhs == rhs)
    return lhs;
  if (lhs.isIndex() && isa<IntegerType>(rhs))
    return lhs;
  if (isa<IntegerType>(lhs) && rhs.isIndex())
    return rhs;

  auto lhsInt = dyn_cast<IntegerType>(lhs);
  auto rhsInt = dyn_cast<IntegerType>(rhs);
  if (lhsInt && rhsInt)
    return lhsInt.getWidth() >= rhsInt.getWidth() ? lhs : rhs;

  auto lhsTensor = dyn_cast<RankedTensorType>(lhs);
  auto rhsTensor = dyn_cast<RankedTensorType>(rhs);
  if (!lhsTensor || !rhsTensor || lhsTensor.getShape() != rhsTensor.getShape())
    return failure();
  Type lhsElement = lhsTensor.getElementType();
  Type rhsElement = rhsTensor.getElementType();
  if (lhsElement == rhsElement)
    return lhs;
  if (lhsElement.isIndex() && isa<IntegerType>(rhsElement))
    return lhs;
  if (isa<IntegerType>(lhsElement) && rhsElement.isIndex())
    return rhs;
  auto lhsElementInt = dyn_cast<IntegerType>(lhsElement);
  auto rhsElementInt = dyn_cast<IntegerType>(rhsElement);
  if (!lhsElementInt || !rhsElementInt)
    return failure();
  return lhsElementInt.getWidth() >= rhsElementInt.getWidth() ? lhs : rhs;
}

static Value createAddWithWiderType(OpBuilder &builder, Location loc, Value lhs,
                                    Value rhs) {
  if (!lhs || !rhs)
    return nullptr;
  FailureOr<Type> type = getWiderIntegerLikeType(lhs.getType(), rhs.getType());
  if (failed(type))
    return nullptr;
  lhs = castIntegerLike(builder, loc, lhs, *type);
  rhs = castIntegerLike(builder, loc, rhs, *type);
  if (!lhs || !rhs)
    return nullptr;
  return builder.create<arith::AddIOp>(loc, lhs, rhs);
}

class TensorPtrDecomposePolicy final : public ControlFlowRewritePolicy {
public:
  bool matches(Type type) const override {
    // A tensor pointer here means tensor<...x!tt.ptr<...>>. Scalar block
    // pointers have already been handled by BlockPtrDecompose.
    return isTensorPointerType(type);
  }

  FailureOr<DecomposedValue>
  decompose(Value value, const ControlFlowRewriteContext &context,
            OpBuilder &builder, Location loc) const override {
    if (const DecomposedValue *known = context.lookup(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    value = context.remap(value);
    if (auto addPtr = value.getDefiningOp<triton::AddPtrOp>()) {
      FailureOr<DecomposedValue> result =
          decompose(addPtr.getPtr(), context, builder, loc);
      if (failed(result) || !hasValidLayout(*result))
        return failure();

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(addPtr);
      Value offsets = createAddWithWiderType(
          builder, addPtr.getLoc(), result->components.front(),
          context.remap(addPtr.getOffset()));
      if (!offsets)
        return failure();
      result->originalType = value.getType();
      result->components.front() = offsets;
      return *result;
    }

    if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
      if (!isa<triton::PointerType>(splat.getSrc().getType()))
        return failure();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(splat);
      Value offsets = createZeroOffsets(builder, splat.getLoc(), value.getType());
      if (!offsets)
        return failure();
      return DecomposedValue{value.getType(), {offsets}, {splat.getSrc()},
                             {builder.getBoolAttr(true)}};
    }

    if (!matches(value.getType()))
      return failure();

    OpBuilder::InsertionGuard guard(builder);
    if (Operation *definingOp = value.getDefiningOp())
      builder.setInsertionPointAfter(definingOp);
    else if (auto blockArg = dyn_cast<BlockArgument>(value))
      builder.setInsertionPointToStart(blockArg.getOwner());
    Value offsets = createZeroOffsets(builder, loc, value.getType());
    if (!offsets)
      return failure();
    return DecomposedValue{value.getType(), {offsets}, {value},
                           {builder.getBoolAttr(false)}};
  }

  Value recompose(const DecomposedValue &value, OpBuilder &builder,
                  Location loc) const override {
    if (!hasValidLayout(value))
      return nullptr;
    Value base = value.invariants.front();
    if (cast<BoolAttr>(value.attributes.front()).getValue())
      base = builder.create<triton::SplatOp>(loc, value.originalType, base);
    return builder.create<triton::AddPtrOp>(loc, value.originalType, base,
                                            value.components.front());
  }

  FailureOr<SmallVector<unsigned>>
  getLoopComponentIndices(const DecomposedValue &value) const override {
    if (!hasValidLayout(value))
      return failure();
    return SmallVector<unsigned>{0};
  }

  bool areLoopStatesCompatible(const DecomposedValue &initial,
                               const DecomposedValue &next) const override {
    return hasValidLayout(initial) && hasValidLayout(next) &&
           initial.originalType == next.originalType &&
           initial.invariants == next.invariants &&
           initial.attributes == next.attributes &&
           initial.components.front().getType() ==
               next.components.front().getType();
  }

  FailureOr<SmallVector<unsigned>>
  getIfComponentIndices(const DecomposedValue &thenValue,
                        const DecomposedValue &elseValue) const override {
    if (!areLoopStatesCompatible(thenValue, elseValue))
      return failure();
    return SmallVector<unsigned>{0};
  }

  bool shouldDecomposeOperation(Operation *op) const override {
    return isa<triton::AddPtrOp>(op);
  }

  FailureOr<SmallVector<Value>>
  matchForInductionDeltas(scf::ForOp, const DecomposedValue &, unsigned,
                          Value) const override {
    return failure();
  }

  FailureOr<DecomposedValue>
  applyForInductionDeltas(const DecomposedValue &, ArrayRef<Value>, Value,
                          const ControlFlowRewriteContext &, OpBuilder &,
                          Location) const override {
    return failure();
  }
};

} // namespace

namespace mlir::triton::controlflow {

LogicalResult runTensorPtrDecompose(ModuleOp module) {
  // Carry only complete per-lane offsets through SCF. The common scalar base
  // remains a rewrite invariant and is used to rebuild tensor-of-pointers at
  // each region boundary. This decomposition is independent of
  // BlockPtrDecompose.
  // TODO: Replace this local extraction with TritonToUnstructure's common-base
  // analysis. Different or lane-wise bases must become explicit diagnostics
  // instead of pattern misses.
  TensorPtrDecomposePolicy policy;
  return rewriteControlFlow(module, policy);
}

} // namespace mlir::triton::controlflow
