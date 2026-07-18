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

#include "TritonControlFlowOpt/BlockPtrDecompose.h"

#include "TritonControlFlowOpt/ControlFlowRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::controlflow;

namespace {

/// Block-pointer component layout used only by this policy:
///
///   components = [shape..., strides..., offsets...]
///   invariants = [base]
///   attributes = [order]
///
/// Loops currently carry only `offsets`; shape and strides must remain
/// invariant across a backedge. An scf.if may select any component whose SSA
/// value differs between its branches.
static FailureOr<unsigned> getRank(const DecomposedValue &value) {
  auto pointerType = dyn_cast<triton::PointerType>(value.originalType);
  if (!pointerType)
    return failure();
  auto tensorType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!tensorType)
    return failure();
  return tensorType.getRank();
}

static bool hasValidLayout(const DecomposedValue &value) {
  FailureOr<unsigned> rank = getRank(value);
  return succeeded(rank) && value.components.size() == 3 * *rank &&
         value.invariants.size() == 1 && value.attributes.size() == 1 &&
         isa<DenseI32ArrayAttr>(value.attributes.front());
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
  if (!sourceInt || !targetInt)
    return nullptr;
  if (sourceInt.getWidth() < targetInt.getWidth())
    return builder.create<arith::ExtSIOp>(loc, targetType, value);
  if (sourceInt.getWidth() > targetInt.getWidth())
    return builder.create<arith::TruncIOp>(loc, targetType, value);
  return value;
}

static Value createAdd(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  if (!lhs || !rhs)
    return nullptr;
  rhs = castIntegerLike(builder, loc, rhs, lhs.getType());
  if (!rhs)
    return nullptr;
  return builder.create<arith::AddIOp>(loc, lhs, rhs);
}

static Value createMul(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  if (!lhs || !rhs)
    return nullptr;
  rhs = castIntegerLike(builder, loc, rhs, lhs.getType());
  if (!rhs)
    return nullptr;
  return builder.create<arith::MulIOp>(loc, lhs, rhs);
}

static bool haveSameTypes(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [](auto values) {
           return std::get<0>(values).getType() ==
                  std::get<1>(values).getType();
         });
}

static bool isConstantIndex(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  return constant && constant.value() == expected;
}

static bool isDefinedOutside(Operation *scope, Value value) {
  if (Operation *defOp = value.getDefiningOp())
    return !scope->isAncestor(defOp);

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return false;
  Operation *owner = blockArg.getOwner()->getParentOp();
  return owner != scope && (!owner || !scope->isAncestor(owner));
}

class BlockPtrPolicy final : public ControlFlowRewritePolicy {
public:
  bool matches(Type type) const override {
    // A Triton block pointer is a scalar !tt.ptr whose pointee is a ranked
    // tensor. Tensor-of-pointers are handled by their own decomposition.
    auto pointerType = dyn_cast<triton::PointerType>(type);
    return pointerType && isa<RankedTensorType>(pointerType.getPointeeType());
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
    if (auto makePtr = value.getDefiningOp<triton::MakeTensorPtrOp>()) {
      DecomposedValue result;
      result.originalType = value.getType();
      result.components.append(makePtr.getShape().begin(),
                               makePtr.getShape().end());
      result.components.append(makePtr.getStrides().begin(),
                               makePtr.getStrides().end());
      result.components.append(makePtr.getOffsets().begin(),
                               makePtr.getOffsets().end());
      result.invariants.push_back(makePtr.getBase());
      result.attributes.push_back(makePtr.getOrderAttr());
      if (!hasValidLayout(result))
        return failure();
      return result;
    }

    auto advance = value.getDefiningOp<triton::AdvanceOp>();
    if (!advance)
      return failure();

    FailureOr<DecomposedValue> result =
        decompose(advance.getPtr(), context, builder, loc);
    if (failed(result) || !hasValidLayout(*result))
      return failure();
    FailureOr<unsigned> rank = getRank(*result);
    if (advance.getOffsets().size() != *rank)
      return failure();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(advance);
    for (auto [dim, delta] : llvm::enumerate(advance.getOffsets())) {
      unsigned component = 2 * *rank + dim;
      Value offset = createAdd(builder, advance.getLoc(),
                               result->components[component],
                               context.remap(delta));
      if (!offset)
        return failure();
      result->components[component] = offset;
    }
    result->originalType = value.getType();
    return *result;
  }

  Value recompose(const DecomposedValue &value, OpBuilder &builder,
                  Location loc) const override {
    if (!hasValidLayout(value))
      return nullptr;
    unsigned rank = *getRank(value);
    auto order = cast<DenseI32ArrayAttr>(value.attributes.front());
    return builder.create<triton::MakeTensorPtrOp>(
        loc, value.originalType, value.invariants.front(),
        ValueRange(value.components).take_front(rank),
        ValueRange(value.components).slice(rank, rank),
        ValueRange(value.components).take_back(rank), order);
  }

  FailureOr<SmallVector<unsigned>>
  getLoopComponentIndices(const DecomposedValue &value) const override {
    if (!hasValidLayout(value))
      return failure();
    unsigned rank = *getRank(value);
    SmallVector<unsigned> indices;
    for (unsigned dim = 0; dim < rank; ++dim)
      indices.push_back(2 * rank + dim);
    return indices;
  }

  bool areLoopStatesCompatible(const DecomposedValue &initial,
                               const DecomposedValue &next) const override {
    if (!hasValidLayout(initial) || !hasValidLayout(next) ||
        initial.originalType != next.originalType ||
        initial.invariants != next.invariants ||
        initial.attributes != next.attributes)
      return false;

    unsigned rank = *getRank(initial);
    ArrayRef<Value> initialStatic(initial.components.data(), 2 * rank);
    ArrayRef<Value> nextStatic(next.components.data(), 2 * rank);
    ArrayRef<Value> initialOffsets(initial.components.data() + 2 * rank, rank);
    ArrayRef<Value> nextOffsets(next.components.data() + 2 * rank, rank);
    return initialStatic == nextStatic &&
           haveSameTypes(initialOffsets, nextOffsets);
  }

  FailureOr<SmallVector<unsigned>>
  getIfComponentIndices(const DecomposedValue &thenValue,
                        const DecomposedValue &elseValue) const override {
    if (!hasValidLayout(thenValue) || !hasValidLayout(elseValue) ||
        thenValue.originalType != elseValue.originalType ||
        thenValue.invariants != elseValue.invariants ||
        thenValue.attributes != elseValue.attributes ||
        thenValue.components.size() != elseValue.components.size())
      return failure();

    SmallVector<unsigned> indices;
    for (auto [index, values] :
         llvm::enumerate(llvm::zip(thenValue.components,
                                  elseValue.components))) {
      Value thenComponent = std::get<0>(values);
      Value elseComponent = std::get<1>(values);
      if (thenComponent == elseComponent)
        continue;
      if (thenComponent.getType() != elseComponent.getType())
        return failure();
      indices.push_back(index);
    }
    return indices;
  }

  bool shouldDecomposeOperation(Operation *op) const override {
    return isa<triton::AdvanceOp>(op);
  }

  FailureOr<SmallVector<Value>>
  matchForInductionDeltas(scf::ForOp forOp,
                          const DecomposedValue &initial,
                          unsigned iterArgIndex,
                          Value yieldOperand) const override {
    if (!hasValidLayout(initial) ||
        !isConstantIndex(forOp.getLowerBound(), 0) ||
        !isConstantIndex(forOp.getStep(), 1))
      return failure();

    auto advance = yieldOperand.getDefiningOp<triton::AdvanceOp>();
    if (!advance ||
        advance.getPtr() != forOp.getRegionIterArgs()[iterArgIndex])
      return failure();

    unsigned rank = *getRank(initial);
    if (advance.getOffsets().size() != rank)
      return failure();

    SmallVector<Value> deltas;
    for (auto [dim, delta] : llvm::enumerate(advance.getOffsets())) {
      Value initialOffset = initial.components[2 * rank + dim];
      if ((!initialOffset.getType().isIndex() &&
           !isa<IntegerType>(initialOffset.getType())) ||
          (!delta.getType().isIndex() && !isa<IntegerType>(delta.getType())) ||
          !isDefinedOutside(forOp, delta))
        return failure();
      deltas.push_back(delta);
    }
    return deltas;
  }

  FailureOr<DecomposedValue>
  applyForInductionDeltas(const DecomposedValue &initial,
                          ArrayRef<Value> deltas, Value inductionVar,
                          const ControlFlowRewriteContext &context,
                          OpBuilder &builder, Location loc) const override {
    FailureOr<SmallVector<unsigned>> indices =
        getLoopComponentIndices(initial);
    if (failed(indices) || indices->size() != deltas.size())
      return failure();

    DecomposedValue result = initial;
    for (auto [component, delta] : llvm::zip(*indices, deltas)) {
      Type type = result.components[component].getType();
      Value iv = castIntegerLike(builder, loc, inductionVar, type);
      Value typedDelta =
          castIntegerLike(builder, loc, context.remap(delta), type);
      Value scaled = createMul(builder, loc, iv, typedDelta);
      Value offset =
          createAdd(builder, loc, result.components[component], scaled);
      if (!iv || !typedDelta || !scaled || !offset)
        return failure();
      result.components[component] = offset;
    }
    return result;
  }
};

} // namespace

namespace mlir::triton::controlflow {

LogicalResult runBlockPtrDecompose(ModuleOp module) {
  // Make the explicit descriptor carried by a block pointer cross each
  // supported SCF boundary as ordinary SSA components.
  BlockPtrPolicy policy;
  return rewriteControlFlow(module, policy);
}

} // namespace mlir::triton::controlflow
