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
#include "PointerAnalysisInternal.h"
#include "TritonMemoryAccess/PointerAnalysisDriver.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <algorithm>

namespace mlir::triton::pointer {
namespace {

bool isI64(OpFoldResult value) {
  if (auto ssa = dyn_cast<Value>(value))
    return ssa.getType().isInteger(64);
  if (auto attr = dyn_cast<Attribute>(value))
    return isa<TypedAttr>(attr) &&
           cast<TypedAttr>(attr).getType().isInteger(64);
  return false;
}

bool validBlockBinding(Value value, const BlockPointerComponents &components) {
  if (!value || !components.base ||
      !isa<triton::PointerType>(components.base.getType()))
    return false;
  auto pointerType = dyn_cast<triton::PointerType>(value.getType());
  if (!pointerType ||
      components.blockTensorType != pointerType.getPointeeType() ||
      components.elementType != components.blockTensorType.getElementType() ||
      components.blockShape !=
          llvm::to_vector(components.blockTensorType.getShape()) ||
      components.addressSpace != pointerType.getAddressSpace())
    return false;
  auto baseType = cast<triton::PointerType>(components.base.getType());
  if (baseType.getPointeeType() != components.elementType ||
      baseType.getAddressSpace() != components.addressSpace)
    return false;
  unsigned rank = components.blockShape.size();
  if (components.parentShape.size() != rank ||
      components.coordinateOffsets.size() != rank ||
      components.coordinateRanges.size() != rank ||
      components.strides.size() != rank || components.order.size() != rank)
    return false;
  SmallVector<bool> seen(rank, false);
  for (int32_t dim : components.order) {
    if (dim < 0 || static_cast<unsigned>(dim) >= rank || seen[dim])
      return false;
    seen[dim] = true;
  }
  for (unsigned i = 0; i < rank; ++i)
    if (!isI64(components.parentShape[i]) || !isI64(components.strides[i]) ||
        !isI64(components.coordinateOffsets[i]) ||
        components.coordinateRanges[i].getBitWidth() != 64)
      return false;
  return true;
}

} // namespace

FailureOr<BlockCoordinates> advanceBlockOffsets(
    ArrayRef<OpFoldResult> current, ArrayRef<llvm::ConstantRange> currentRanges,
    const BlockAdvanceDelta &delta, OpBuilder &builder, Location loc) {
  if (current.size() != currentRanges.size() ||
      current.size() != delta.coordinateDeltas.size() ||
      current.size() != delta.ranges.size())
    return failure();
  BlockCoordinates result;
  result.values.reserve(current.size());
  result.ranges.reserve(current.size());
  for (unsigned dim = 0; dim < current.size(); ++dim) {
    if (!isI64(current[dim]) || !isI64(delta.coordinateDeltas[dim]) ||
        currentRanges[dim].getBitWidth() != 64 ||
        delta.ranges[dim].getBitWidth() != 64)
      return failure();
    FailureOr<OpFoldResult> sum = detail::foldOrCreateBinary(
        detail::BinaryKind::Add, current[dim], delta.coordinateDeltas[dim],
        builder.getI64Type(), AnalysisOptions{}, builder, loc);
    if (failed(sum))
      return failure();
    result.values.push_back(*sum);
    result.ranges.push_back(currentRanges[dim].add(delta.ranges[dim]));
  }
  return result;
}

FailureOr<OffsetComponents>
buildAddressView(const BlockPointerComponents &block, OpBuilder &builder,
                 Location loc) {
  if (!detail::canMaterializeAddressTensor(block.blockTensorType) ||
      block.blockShape != llvm::to_vector(block.blockTensorType.getShape()) ||
      block.coordinateOffsets.size() != block.blockShape.size() ||
      block.strides.size() != block.blockShape.size())
    return failure();
  auto tensorType =
      RankedTensorType::get(block.blockShape, builder.getI64Type(),
                            block.blockTensorType.getEncoding());
  OpFoldResult uniform = builder.getI64IntegerAttr(0);
  Value complete;
  OffsetComponents result;
  result.valueType = tensorType;
  result.domain = ArithmeticDomain::ElementAddress;
  result.shape = block.blockShape;
  for (unsigned dim = 0; dim < block.blockShape.size(); ++dim) {
    if (!isI64(block.coordinateOffsets[dim]) || !isI64(block.strides[dim]))
      return failure();
    FailureOr<OpFoldResult> contribution = detail::foldOrCreateBinary(
        detail::BinaryKind::Multiply, block.coordinateOffsets[dim],
        block.strides[dim], builder.getI64Type(), AnalysisOptions{}, builder,
        loc);
    if (failed(contribution))
      return failure();
    FailureOr<OpFoldResult> nextUniform = detail::foldOrCreateBinary(
        detail::BinaryKind::Add, uniform, *contribution, builder.getI64Type(),
        AnalysisOptions{}, builder, loc);
    if (failed(nextUniform))
      return failure();
    uniform = *nextUniform;
    bool invariant = false;
    if (FailureOr<llvm::APInt> constant =
            detail::getIntegerConstant(block.strides[dim], 64);
        succeeded(constant))
      invariant = constant->isZero();
    result.axes.push_back(invariant ? AxisKind::Invariant
                                    : AxisKind::Structured);
    result.strides.push_back(invariant
                                 ? OpFoldResult(builder.getI64IntegerAttr(0))
                                 : block.strides[dim]);

    FailureOr<Value> lane =
        detail::materializeAxisRange(tensorType, dim, builder, loc);
    FailureOr<Value> coordinate = detail::materialize(
        block.coordinateOffsets[dim], builder.getI64Type(), builder, loc);
    FailureOr<Value> stride = detail::materialize(
        block.strides[dim], builder.getI64Type(), builder, loc);
    if (failed(lane) || failed(coordinate) || failed(stride))
      return failure();
    FailureOr<Value> coordinateTensor =
        detail::splatTo(*coordinate, tensorType, builder, loc);
    FailureOr<Value> strideTensor =
        detail::splatTo(*stride, tensorType, builder, loc);
    if (failed(coordinateTensor) || failed(strideTensor))
      return failure();
    Value atLane = builder.create<arith::AddIOp>(loc, *coordinateTensor, *lane);
    Value term = builder.create<arith::MulIOp>(loc, atLane, *strideTensor);
    complete =
        complete ? builder.create<arith::AddIOp>(loc, complete, term) : term;
  }
  if (!complete) {
    Value zero =
        builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(0));
    FailureOr<Value> splat = detail::splatTo(zero, tensorType, builder, loc);
    if (failed(splat))
      return failure();
    complete = *splat;
  }
  result.completeOffset = complete;
  result.uniformOffset = uniform;
  return result;
}

namespace detail {

std::optional<BlockPointerRules::Result>
BlockPointerRules::resolveBoundary(AnalysisRequest request) const {
  if (request.kind <= OrdinaryPointer) {
    auto resolved = basic.resolveBoundary(request);
    if (resolved)
      return Result{*resolved, std::nullopt, std::nullopt};
  }
  return std::nullopt;
}

FailureOr<SmallVector<AnalysisRequest>>
BlockPointerRules::collectInputs(AnalysisRequest request) const {
  if (request.kind != SourceInteger &&
      !supportsAddressAnalysis(basic.getOptions()))
    return failure();
  if (request.kind <= OrdinaryPointer)
    return basic.collectInputs(request);
  SmallVector<AnalysisRequest> inputs;
  if (request.kind == AdvanceDeltaRequest) {
    auto advance = request.value.getDefiningOp<triton::AdvanceOp>();
    if (!advance)
      return failure();
    for (Value delta : advance.getOffsets())
      inputs.push_back({delta, AddressDelta});
    return inputs;
  }
  if (request.kind != BlockPointer)
    return failure();
  if (auto make = request.value.getDefiningOp<triton::MakeTensorPtrOp>()) {
    for (Value coordinate : make.getOffsets())
      inputs.push_back({coordinate, AddressDelta});
    return inputs;
  }
  if (auto advance = request.value.getDefiningOp<triton::AdvanceOp>()) {
    inputs.push_back({advance.getPtr(), BlockPointer});
    inputs.push_back({request.value, AdvanceDeltaRequest});
    return inputs;
  }
  return failure();
}

FailureOr<BlockPointerRules::Result>
BlockPointerRules::transfer(AnalysisRequest request, ArrayRef<Result> inputs,
                            OpBuilder &builder) const {
  if (request.kind <= OrdinaryPointer) {
    SmallVector<BasicResult, 1> basicInputs;
    basicInputs.reserve(inputs.size());
    for (const Result &input : inputs)
      basicInputs.push_back(input.basic);
    FailureOr<BasicResult> basicResult =
        basic.transfer(request, basicInputs, builder);
    if (failed(basicResult))
      return failure();
    Result result;
    result.basic = std::move(*basicResult);
    return result;
  }
  if (request.kind == AdvanceDeltaRequest) {
    BlockAdvanceDelta delta;
    for (const Result &input : inputs) {
      if (!input.basic.offset || !input.basic.integerRange ||
          input.basic.offset->domain != ArithmeticDomain::ElementAddress ||
          !input.basic.offset->shape.empty() ||
          input.basic.integerRange->getBitWidth() != 64)
        return failure();
      delta.coordinateDeltas.push_back(input.basic.offset->completeOffset);
      delta.ranges.push_back(*input.basic.integerRange);
    }
    Result result;
    result.advanceDelta = std::move(delta);
    return result;
  }
  if (request.kind != BlockPointer)
    return failure();
  if (auto make = request.value.getDefiningOp<triton::MakeTensorPtrOp>()) {
    auto pointerType = dyn_cast<triton::PointerType>(make.getType());
    if (!pointerType || !isa<RankedTensorType>(pointerType.getPointeeType()) ||
        inputs.size() != make.getOffsets().size())
      return failure();
    BlockPointerComponents block;
    block.base = make.getBase();
    block.blockTensorType =
        cast<RankedTensorType>(pointerType.getPointeeType());
    block.blockShape = llvm::to_vector(block.blockTensorType.getShape());
    block.elementType = block.blockTensorType.getElementType();
    block.addressSpace = pointerType.getAddressSpace();
    block.order = llvm::to_vector(make.getOrderAttr().asArrayRef());
    for (Value shape : make.getShape())
      block.parentShape.push_back(shape);
    for (Value stride : make.getStrides())
      block.strides.push_back(stride);
    for (const Result &input : inputs) {
      if (!input.basic.offset || !input.basic.integerRange ||
          input.basic.offset->domain != ArithmeticDomain::ElementAddress ||
          !input.basic.offset->shape.empty() ||
          input.basic.integerRange->getBitWidth() != 64)
        return failure();
      block.coordinateOffsets.push_back(input.basic.offset->completeOffset);
      block.coordinateRanges.push_back(*input.basic.integerRange);
    }
    if (!validBlockBinding(request.value, block))
      return failure();
    Result result;
    result.block = std::move(block);
    return result;
  }
  if (auto advance = request.value.getDefiningOp<triton::AdvanceOp>()) {
    if (inputs.size() != 2 || !inputs[0].block || !inputs[1].advanceDelta)
      return failure();
    BlockPointerComponents block = *inputs[0].block;
    FailureOr<BlockCoordinates> updated =
        advanceBlockOffsets(block.coordinateOffsets, block.coordinateRanges,
                            *inputs[1].advanceDelta, builder, advance.getLoc());
    if (failed(updated))
      return failure();
    block.coordinateOffsets = std::move(updated->values);
    block.coordinateRanges = std::move(updated->ranges);
    Result result;
    result.block = std::move(block);
    return result;
  }
  return failure();
}

LogicalResult BlockPointerRules::mapResultValues(
    Result &result, llvm::function_ref<FailureOr<Value>(Value)> mapper) const {
  if (failed(basic.mapResultValues(result.basic, mapper)))
    return failure();
  if (result.block) {
    FailureOr<Value> base = mapper(result.block->base);
    if (failed(base))
      return failure();
    result.block->base = *base;
    for (OpFoldResult &field : result.block->parentShape)
      if (failed(mapOfr(field, mapper)))
        return failure();
    for (OpFoldResult &field : result.block->coordinateOffsets)
      if (failed(mapOfr(field, mapper)))
        return failure();
    for (OpFoldResult &field : result.block->strides)
      if (failed(mapOfr(field, mapper)))
        return failure();
  }
  if (result.advanceDelta)
    for (OpFoldResult &field : result.advanceDelta->coordinateDeltas)
      if (failed(mapOfr(field, mapper)))
        return failure();
  return success();
}

} // namespace detail

struct BlockPointerAnalysis::Impl {
  explicit Impl(OpBuilder &builder, AnalysisOptions options)
      : options(options), driver(builder, detail::BlockPointerRules(options)) {}
  AnalysisOptions options;
  detail::AnalysisDriver<detail::BlockPointerRules> driver;
};

BlockPointerAnalysis::BlockPointerAnalysis(OpBuilder &builder,
                                           AnalysisOptions options)
    : impl(std::make_unique<Impl>(builder, options)) {}
BlockPointerAnalysis::~BlockPointerAnalysis() = default;
BlockPointerAnalysis::BlockPointerAnalysis(BlockPointerAnalysis &&) noexcept =
    default;
BlockPointerAnalysis &
BlockPointerAnalysis::operator=(BlockPointerAnalysis &&) noexcept = default;

FailureOr<OffsetComponents> BlockPointerAnalysis::analyzeOffset(Value value) {
  auto result = impl->driver.analyze({value, detail::SourceInteger});
  if (failed(result) || !result->basic.offset)
    return failure();
  return *result->basic.offset;
}

FailureOr<OffsetComponents>
BlockPointerAnalysis::analyzeAddressDelta(Value value) {
  if (!detail::supportsAddressAnalysis(impl->options))
    return failure();
  auto result = impl->driver.analyze({value, detail::AddressDelta});
  if (failed(result) || !result->basic.offset)
    return failure();
  return *result->basic.offset;
}

FailureOr<PointerComponents> BlockPointerAnalysis::analyzePointer(Value value) {
  if (!detail::supportsAddressAnalysis(impl->options))
    return failure();
  auto result = impl->driver.analyze({value, detail::OrdinaryPointer});
  if (failed(result) || !result->basic.pointer)
    return failure();
  return *result->basic.pointer;
}

FailureOr<BlockPointerComponents>
BlockPointerAnalysis::analyzeBlockPointer(Value value) {
  if (!detail::supportsAddressAnalysis(impl->options))
    return failure();
  auto result = impl->driver.analyze({value, detail::BlockPointer});
  if (failed(result) || !result->block)
    return failure();
  return *result->block;
}

FailureOr<BlockAdvanceDelta>
BlockPointerAnalysis::analyzeBlockAdvanceDelta(triton::AdvanceOp op) {
  if (!detail::supportsAddressAnalysis(impl->options))
    return failure();
  if (!op)
    return failure();
  auto result =
      impl->driver.analyze({op.getResult(), detail::AdvanceDeltaRequest});
  if (failed(result) || !result->advanceDelta)
    return failure();
  return *result->advanceDelta;
}

const OffsetComponents *
BlockPointerAnalysis::findCachedOffset(Value value) const {
  auto *result = impl->driver.lookupCompleted({value, detail::SourceInteger});
  return result ? detail::getBasicOffset(result->basic) : nullptr;
}

const OffsetComponents *
BlockPointerAnalysis::findCachedAddressDelta(Value value) const {
  if (!detail::supportsAddressAnalysis(impl->options))
    return nullptr;
  auto *result = impl->driver.lookupCompleted({value, detail::AddressDelta});
  return result ? detail::getBasicOffset(result->basic) : nullptr;
}

const BlockPointerComponents *
BlockPointerAnalysis::findCachedBlockPointer(Value value) const {
  if (!detail::supportsAddressAnalysis(impl->options))
    return nullptr;
  auto *result = impl->driver.lookupCompleted({value, detail::BlockPointer});
  return result && result->block ? &*result->block : nullptr;
}

const BlockAdvanceDelta *
BlockPointerAnalysis::findCachedBlockAdvanceDelta(triton::AdvanceOp op) const {
  if (!detail::supportsAddressAnalysis(impl->options))
    return nullptr;
  if (!op)
    return nullptr;
  auto *result = impl->driver.lookupCompleted(
      {op.getResult(), detail::AdvanceDeltaRequest});
  return result && result->advanceDelta ? &*result->advanceDelta : nullptr;
}

LogicalResult
BlockPointerAnalysis::bindOffset(Value boundary,
                                 const OffsetComponents &components) {
  if (!detail::validOffsetBinding(
          boundary, components, ArithmeticDomain::SourceInteger, impl->options))
    return failure();
  auto width = detail::getIntegerWidth(boundary.getType(), impl->options);
  detail::BlockResult result;
  result.basic.offset = components;
  result.basic.integerRange = llvm::ConstantRange::getFull(*width);
  impl->driver.bindBoundary({boundary, detail::SourceInteger},
                            std::move(result));
  return success();
}

LogicalResult
BlockPointerAnalysis::bindPointer(Value boundary,
                                  const PointerComponents &components) {
  if (!detail::validPointerBinding(boundary, components, impl->options))
    return failure();
  detail::BlockResult result;
  result.basic.pointer = components;
  impl->driver.bindBoundary({boundary, detail::OrdinaryPointer},
                            std::move(result));
  return success();
}

LogicalResult BlockPointerAnalysis::bindBlockPointer(
    Value boundary, const BlockPointerComponents &components) {
  if (!detail::supportsAddressAnalysis(impl->options))
    return failure();
  if (!validBlockBinding(boundary, components))
    return failure();
  detail::BlockResult result;
  result.block = components;
  impl->driver.bindBoundary({boundary, detail::BlockPointer},
                            std::move(result));
  return success();
}

LogicalResult
BlockPointerAnalysis::remapAndForgetScope(Operation *oldScope,
                                          const IRMapping &resultMapping) {
  return impl->driver.remapAndForgetScope(oldScope, resultMapping);
}

void BlockPointerAnalysis::clear() { impl->driver.clear(); }

} // namespace mlir::triton::pointer
