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

#include "TritonControlFlowOpt/ControlFlowRewrite.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using mlir::triton::controlflow::ControlFlowRewriteContext;
using mlir::triton::controlflow::ControlFlowRewritePolicy;
using mlir::triton::controlflow::DecomposedValue;

namespace mlir::triton::controlflow {

Value ControlFlowRewriteContext::remap(Value value) const {
  if (Value mapped = valueMapping.lookupOrNull(value))
    return mapped;
  return value;
}

const DecomposedValue *ControlFlowRewriteContext::lookup(Value value) const {
  auto it = decomposedValues.find(value);
  return it == decomposedValues.end() ? nullptr : &it->second;
}

} // namespace mlir::triton::controlflow

namespace {

// Keep the mechanical if/for/while rewrite in one translation unit. These
// handlers are mutually recursive through rewriteBodyOps(), share one
// short-lived RewriteEnv, and must agree on signature expansion, nested-op
// ordering and failure cleanup. Splitting them by op kind would expose those
// private invariants through additional internal headers without creating an
// independently reusable component.
//
//===----------------------------------------------------------------------===//
// Per-rewrite state and generic component manipulation
//===----------------------------------------------------------------------===//

struct RewriteEnv {
  explicit RewriteEnv(const ControlFlowRewritePolicy &policy)
      : policy(policy) {}

  IRMapping valueMapping;
  DenseMap<Value, DecomposedValue> decomposedValues;
  const ControlFlowRewritePolicy &policy;
};

// RewriteEnv is copied when entering a newly built region. The copy inherits
// mappings visible at the region boundary and records additional mappings only
// for that recursive rewrite. Nothing is stored on the IR or shared between
// decomposition policies.

struct LoopPointerInfo {
  unsigned oldIndex = 0;
  DecomposedValue initInfo;
  SmallVector<unsigned> componentIndices;
  SmallVector<unsigned> newIndices;
  SmallVector<Value> ivDeltas;
};

struct IfPointerInfo {
  unsigned oldIndex = 0;
  DecomposedValue thenInfo;
  DecomposedValue elseInfo;
  SmallVector<unsigned> componentIndices;
};

static Value remapValue(Value value, const RewriteEnv &env) {
  return ControlFlowRewriteContext(env.valueMapping, env.decomposedValues)
      .remap(value);
}

static FailureOr<DecomposedValue>
decompose(Value value, const RewriteEnv &env, OpBuilder &builder,
          Location loc) {
  return env.policy.decompose(
      value, ControlFlowRewriteContext(env.valueMapping, env.decomposedValues),
      builder, loc);
}

static void recordDecomposition(Value oldValue, const DecomposedValue &info,
                                Value rebuiltValue,
                                RewriteEnv &env) {
  env.decomposedValues[oldValue] = info;
  env.valueMapping.map(oldValue, rebuiltValue);
}

static SmallVector<Value>
getComponentValues(const DecomposedValue &info, ArrayRef<unsigned> indices) {
  SmallVector<Value> values;
  values.reserve(indices.size());
  for (unsigned index : indices)
    values.push_back(info.components[index]);
  return values;
}

static FailureOr<DecomposedValue>
withComponentValues(DecomposedValue info, ArrayRef<unsigned> indices,
                    ArrayRef<Value> values) {
  if (indices.size() != values.size())
    return failure();
  for (auto [index, value] : llvm::zip(indices, values)) {
    if (index >= info.components.size() ||
        info.components[index].getType() != value.getType())
      return failure();
    info.components[index] = value;
  }
  return info;
}

//===----------------------------------------------------------------------===//
// Shared recursive body rewrite
//===----------------------------------------------------------------------===//

static LoopPointerInfo *findLoopInfo(SmallVectorImpl<LoopPointerInfo> &infos,
                                     unsigned oldIndex) {
  for (LoopPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static const LoopPointerInfo *findLoopInfo(ArrayRef<LoopPointerInfo> infos,
                                           unsigned oldIndex) {
  for (const LoopPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static SmallVector<Value> collectForComponents(const LoopPointerInfo &info,
                                               scf::ForOp forOp,
                                               bool useResults) {
  SmallVector<Value> values;
  for (unsigned newIndex : info.newIndices)
    values.push_back(useResults ? forOp.getResult(newIndex)
                                : forOp.getRegionIterArgs()[newIndex]);
  return values;
}

static SmallVector<Value> collectWhileComponents(const LoopPointerInfo &info,
                                                 scf::WhileOp whileOp,
                                                 bool useResults,
                                                 bool useAfterArgs) {
  SmallVector<Value> values;
  for (unsigned newIndex : info.newIndices) {
    if (useResults)
      values.push_back(whileOp.getResult(newIndex));
    else if (useAfterArgs)
      values.push_back(whileOp.getAfterArguments()[newIndex]);
    else
      values.push_back(whileOp.getBeforeArguments()[newIndex]);
  }
  return values;
}

static LogicalResult rewriteControlFlowOp(Operation *op, OpBuilder &builder,
                                          RewriteEnv &env);

static LogicalResult materializePointerResult(Operation &bodyOp,
                                              Operation *clonedOp,
                                              OpBuilder &builder,
                                              RewriteEnv &env) {
  // Each policy decides which pointer-producing operations need their exact
  // components recorded immediately after cloning.
  if (!env.policy.shouldDecomposeOperation(&bodyOp))
    return success();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(clonedOp);

  for (auto [oldResult, clonedResult] :
       llvm::zip(bodyOp.getResults(), clonedOp->getResults())) {
    if (!env.policy.matches(oldResult.getType()))
      continue;

    FailureOr<DecomposedValue> info =
        decompose(clonedResult, env, builder, oldResult.getLoc());
    if (failed(info))
      continue;

    Value rebuilt = env.policy.recompose(*info, builder, oldResult.getLoc());
    if (!rebuilt)
      return failure();
    recordDecomposition(oldResult, *info, rebuilt, env);
  }

  return success();
}

static LogicalResult rewriteBodyOps(Block *oldBlock, OpBuilder &builder,
                                    RewriteEnv &env) {
  // Process operations in program order. Nested control flow is rewritten
  // recursively with the same policy; ordinary operations are cloned through
  // the current SSA mapping.
  for (Operation &bodyOp : oldBlock->without_terminator()) {
    // TODO: Distinguish "no slot owned by this policy" from "owned slot failed
    // to rewrite". The latter must fail the enclosing rewrite instead of
    // cloning an opaque nested control-flow operation.
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(bodyOp) &&
        succeeded(rewriteControlFlowOp(&bodyOp, builder, env)))
      continue;
    Operation *clonedOp = builder.clone(bodyOp, env.valueMapping);
    if (failed(materializePointerResult(bodyOp, clonedOp, builder, env)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// scf.for rewrite
//===----------------------------------------------------------------------===//

static LogicalResult rewriteForOp(scf::ForOp forOp, OpBuilder &builder,
                                  RewriteEnv &env) {
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<LoopPointerInfo, 4> pointerInfos;

  // First determine which old iter-argument slots belong to this policy and
  // recover the exact component schema from their init values.
  OpBuilder analysisBuilder(forOp.getContext());
  analysisBuilder.setInsertionPoint(forOp);

  for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (!env.policy.matches(iterArg.getType()))
      continue;
    if (idx >= forOp.getInitArgs().size() || idx >= yieldOp.getNumOperands())
      return failure();

    FailureOr<DecomposedValue> initInfo =
        decompose(forOp.getInitArgs()[idx], env, analysisBuilder,
                  forOp.getLoc());
    if (failed(initInfo))
      continue;
    FailureOr<SmallVector<unsigned>> componentIndices =
        env.policy.getLoopComponentIndices(*initInfo);
    if (failed(componentIndices))
      continue;
    pointerInfos.push_back(LoopPointerInfo{static_cast<unsigned>(idx),
                                           *initInfo, *componentIndices, {}});
  }

  if (pointerInfos.empty())
    return failure();

  for (LoopPointerInfo &info : pointerInfos) {
    FailureOr<SmallVector<Value>> deltas =
        env.policy.matchForInductionDeltas(
            forOp, info.initInfo, info.oldIndex,
            yieldOp.getOperand(info.oldIndex));
    if (succeeded(deltas))
      info.ivDeltas = *deltas;
  }

  SmallVector<Value> newInitArgs;
  SmallVector<unsigned> oldToNewStart(forOp.getInitArgs().size(), 0);
  // Expand each owned pointer init into its runtime components. Non-pointer and
  // other-policy slots retain one position in the new signature.
  for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    oldToNewStart[idx] = newInitArgs.size();
    if (LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      for (Value component :
           getComponentValues(info->initInfo, info->componentIndices)) {
        info->newIndices.push_back(newInitArgs.size());
        newInitArgs.push_back(component);
      }
      continue;
    }
    newInitArgs.push_back(remapValue(initArg, env));
  }

  bool bodyOk = true;
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), remapValue(forOp.getLowerBound(), env),
      remapValue(forOp.getUpperBound(), env), remapValue(forOp.getStep(), env),
      newInitArgs,
      [&](OpBuilder &bodyBuilder, Location loc, Value iv, ValueRange args) {
        RewriteEnv bodyEnv = env;
        bodyEnv.valueMapping.map(forOp.getInductionVar(), iv);

        // Reconstruct the original iter-argument type at region entry before
        // cloning users. This keeps the body semantics independent of the
        // signature expansion.
        for (auto [idx, oldArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            FailureOr<DecomposedValue> argInfo = withComponentValues(
                info->initInfo, info->componentIndices, values);
            FailureOr<DecomposedValue> closedFormInfo =
                env.policy.applyForInductionDeltas(
                    info->initInfo, info->ivDeltas, iv,
                    ControlFlowRewriteContext(bodyEnv.valueMapping,
                                              bodyEnv.decomposedValues),
                    bodyBuilder, loc);
            if (succeeded(closedFormInfo))
              argInfo = *closedFormInfo;
            if (failed(argInfo)) {
              bodyOk = false;
              continue;
            }
            Value rebuilt = env.policy.recompose(*argInfo, bodyBuilder, loc);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordDecomposition(oldArg, *argInfo, rebuilt, bodyEnv);
            continue;
          }
          bodyEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(rewriteBodyOps(forOp.getBody(), bodyBuilder, bodyEnv)))
          bodyOk = false;

        SmallVector<Value> newYieldOperands;
        // Decompose yielded pointers back into the expanded component order.
        // Compatibility guards the loop-carried schema across the backedge.
        for (auto [idx, oldOperand] : llvm::enumerate(yieldOp.getOperands())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<DecomposedValue> nextInfo = decompose(
                oldOperand, bodyEnv, bodyBuilder, yieldOp.getLoc());
            if (failed(nextInfo) ||
                !env.policy.areLoopStatesCompatible(info->initInfo,
                                                    *nextInfo)) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newYieldOperands.push_back(args[newIndex]);
              continue;
            }
            for (Value component :
                 getComponentValues(*nextInfo, info->componentIndices))
              newYieldOperands.push_back(component);
            continue;
          }
          newYieldOperands.push_back(remapValue(oldOperand, bodyEnv));
        }

        bodyBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newForOp->setAttrs(forOp->getAttrs());

  if (!bodyOk) {
    newForOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newForOp);
  // Rebuild pointer results after the new loop and map every untouched result
  // to the corresponding expanded-result index.
  for (auto [idx, oldResult] : llvm::enumerate(forOp.getResults())) {
    if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      FailureOr<DecomposedValue> resultInfo = withComponentValues(
          info->initInfo,
          info->componentIndices,
          collectForComponents(*info, newForOp, /*useResults=*/true));
      if (failed(resultInfo)) {
        newForOp.erase();
        return failure();
      }
      Value rebuilt =
          env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
      if (!rebuilt) {
        newForOp.erase();
        return failure();
      }
      recordDecomposition(oldResult, *resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newForOp.getResult(oldToNewStart[idx]));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// scf.while rewrite
//===----------------------------------------------------------------------===//

static LogicalResult rewriteWhileOp(scf::WhileOp whileOp, OpBuilder &builder,
                                    RewriteEnv &env) {
  scf::ConditionOp conditionOp = whileOp.getConditionOp();
  scf::YieldOp yieldOp = whileOp.getYieldOp();
  SmallVector<LoopPointerInfo, 4> pointerInfos;

  // The before arguments, condition forwarded values, after arguments and
  // yield operands all share one positional schema. Analyze that schema from
  // the initial values before creating either replacement region.
  OpBuilder analysisBuilder(whileOp.getContext());
  analysisBuilder.setInsertionPoint(whileOp);

  for (auto [idx, beforeArg] : llvm::enumerate(whileOp.getBeforeArguments())) {
    if (!env.policy.matches(beforeArg.getType()))
      continue;
    if (idx >= whileOp.getInits().size() ||
        idx >= conditionOp.getArgs().size() || idx >= yieldOp.getNumOperands())
      return failure();

    FailureOr<DecomposedValue> initInfo = decompose(
        whileOp.getInits()[idx], env, analysisBuilder, whileOp.getLoc());
    if (failed(initInfo))
      continue;
    FailureOr<SmallVector<unsigned>> componentIndices =
        env.policy.getLoopComponentIndices(*initInfo);
    if (failed(componentIndices))
      continue;
    pointerInfos.push_back(LoopPointerInfo{static_cast<unsigned>(idx),
                                           *initInfo, *componentIndices, {}});
  }

  if (pointerInfos.empty())
    return failure();

  SmallVector<Value> newInits;
  SmallVector<Type> newResultTypes;
  SmallVector<unsigned> oldToNewStart(whileOp.getInits().size(), 0);
  for (auto [idx, initArg] : llvm::enumerate(whileOp.getInits())) {
    oldToNewStart[idx] = newInits.size();
    if (LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      for (Value component :
           getComponentValues(info->initInfo, info->componentIndices)) {
        info->newIndices.push_back(newInits.size());
        newInits.push_back(component);
        newResultTypes.push_back(component.getType());
      }
      continue;
    }
    newInits.push_back(remapValue(initArg, env));
    newResultTypes.push_back(whileOp.getResult(idx).getType());
  }

  bool bodyOk = true;
  auto newWhileOp = builder.create<scf::WhileOp>(
      whileOp.getLoc(), newResultTypes, newInits,
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange args) {
        RewriteEnv beforeEnv = env;
        // Rebuild pointer values at the before-region boundary, rewrite the
        // body, then decompose the values forwarded by scf.condition.
        for (auto [idx, oldArg] :
             llvm::enumerate(whileOp.getBeforeArguments())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            FailureOr<DecomposedValue> argInfo = withComponentValues(
                info->initInfo, info->componentIndices, values);
            if (failed(argInfo)) {
              bodyOk = false;
              continue;
            }
            Value rebuilt = env.policy.recompose(*argInfo, bodyBuilder, loc);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordDecomposition(oldArg, *argInfo, rebuilt, beforeEnv);
            continue;
          }
          beforeEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(rewriteBodyOps(whileOp.getBeforeBody(), bodyBuilder,
                                  beforeEnv)))
          bodyOk = false;

        SmallVector<Value> newConditionArgs;
        for (auto [idx, oldArg] : llvm::enumerate(conditionOp.getArgs())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<DecomposedValue> conditionInfo = decompose(
                oldArg, beforeEnv, bodyBuilder, conditionOp.getLoc());
            if (failed(conditionInfo) ||
                !env.policy.areLoopStatesCompatible(info->initInfo,
                                                    *conditionInfo)) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newConditionArgs.push_back(args[newIndex]);
              continue;
            }
            for (Value component :
                 getComponentValues(*conditionInfo, info->componentIndices))
              newConditionArgs.push_back(component);
            continue;
          }
          newConditionArgs.push_back(remapValue(oldArg, beforeEnv));
        }

        bodyBuilder.create<scf::ConditionOp>(
            conditionOp.getLoc(),
            remapValue(conditionOp.getCondition(), beforeEnv),
            newConditionArgs);
      },
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange args) {
        RewriteEnv afterEnv = env;
        // Apply the same reconstruction/decomposition contract to the after
        // region and its backedge yield.
        for (auto [idx, oldArg] :
             llvm::enumerate(whileOp.getAfterArguments())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            SmallVector<Value> values;
            for (unsigned newIndex : info->newIndices)
              values.push_back(args[newIndex]);
            FailureOr<DecomposedValue> argInfo = withComponentValues(
                info->initInfo, info->componentIndices, values);
            if (failed(argInfo)) {
              bodyOk = false;
              continue;
            }
            Value rebuilt = env.policy.recompose(*argInfo, bodyBuilder, loc);
            if (!rebuilt) {
              bodyOk = false;
              continue;
            }
            recordDecomposition(oldArg, *argInfo, rebuilt, afterEnv);
            continue;
          }
          afterEnv.valueMapping.map(oldArg, args[oldToNewStart[idx]]);
        }

        if (failed(
                rewriteBodyOps(whileOp.getAfterBody(), bodyBuilder, afterEnv)))
          bodyOk = false;

        SmallVector<Value> newYieldOperands;
        for (auto [idx, oldOperand] : llvm::enumerate(yieldOp.getOperands())) {
          if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
            FailureOr<DecomposedValue> nextInfo = decompose(
                oldOperand, afterEnv, bodyBuilder, yieldOp.getLoc());
            if (failed(nextInfo) ||
                !env.policy.areLoopStatesCompatible(info->initInfo,
                                                    *nextInfo)) {
              bodyOk = false;
              for (unsigned newIndex : info->newIndices)
                newYieldOperands.push_back(args[newIndex]);
              continue;
            }
            for (Value component :
                 getComponentValues(*nextInfo, info->componentIndices))
              newYieldOperands.push_back(component);
            continue;
          }
          newYieldOperands.push_back(remapValue(oldOperand, afterEnv));
        }

        bodyBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newWhileOp->setAttrs(whileOp->getAttrs());

  if (!bodyOk) {
    newWhileOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newWhileOp);
  for (auto [idx, oldResult] : llvm::enumerate(whileOp.getResults())) {
    if (const LoopPointerInfo *info = findLoopInfo(pointerInfos, idx)) {
      FailureOr<DecomposedValue> resultInfo = withComponentValues(
          info->initInfo,
          info->componentIndices,
          collectWhileComponents(*info, newWhileOp, /*useResults=*/true,
                                 /*useAfterArgs=*/false));
      if (failed(resultInfo)) {
        newWhileOp.erase();
        return failure();
      }
      Value rebuilt =
          env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
      if (!rebuilt) {
        newWhileOp.erase();
        return failure();
      }
      recordDecomposition(oldResult, *resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newWhileOp.getResult(oldToNewStart[idx]));
  }

  return success();
}

static FailureOr<DecomposedValue>
analyzeValueForIfPlanning(Value value, const RewriteEnv &env,
                          OpBuilder &builder, Location loc);

static FailureOr<DecomposedValue>
analyzeNestedIfResultForPlanning(scf::IfOp ifOp, unsigned resultIndex,
                                 const RewriteEnv &env, OpBuilder &builder,
                                 Location loc) {
  // Planning an outer operation may reach a result of an inner scf.if before
  // that inner operation has been cloned. Inspect the two yields directly so
  // outer-first rewriting can still determine the pointer schema.
  if (!ifOp.elseBlock() || resultIndex >= ifOp.getNumResults())
    return failure();

  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = ifOp.elseYield();
  if (resultIndex >= thenYield.getNumOperands() ||
      resultIndex >= elseYield.getNumOperands())
    return failure();

  FailureOr<DecomposedValue> thenInfo = analyzeValueForIfPlanning(
      thenYield.getOperand(resultIndex), env, builder, loc);
  FailureOr<DecomposedValue> elseInfo = analyzeValueForIfPlanning(
      elseYield.getOperand(resultIndex), env, builder, loc);
  if (failed(thenInfo) || failed(elseInfo) ||
      failed(env.policy.getIfComponentIndices(*thenInfo, *elseInfo)))
    return failure();
  return *thenInfo;
}

static FailureOr<DecomposedValue>
analyzeValueForIfPlanning(Value value, const RewriteEnv &env,
                          OpBuilder &builder, Location loc) {
  if (auto it = env.decomposedValues.find(value);
      it != env.decomposedValues.end())
    return it->second;

  Value mapped = remapValue(value, env);
  if (auto result = dyn_cast<OpResult>(mapped)) {
    if (auto nestedIf = dyn_cast<scf::IfOp>(result.getOwner())) {
      FailureOr<DecomposedValue> nestedInfo =
          analyzeNestedIfResultForPlanning(
              nestedIf, result.getResultNumber(), env, builder, loc);
      if (succeeded(nestedInfo))
        return nestedInfo;
    }
  }

  return decompose(value, env, builder, loc);
}

//===----------------------------------------------------------------------===//
// scf.if component planning and rewrite
//===----------------------------------------------------------------------===//

static const IfPointerInfo *findIfInfo(ArrayRef<IfPointerInfo> infos,
                                       unsigned oldIndex) {
  for (const IfPointerInfo &info : infos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

static LogicalResult rewriteIfOp(scf::IfOp ifOp, OpBuilder &builder,
                                 RewriteEnv &env) {
  if (!ifOp.elseBlock() || ifOp->getNumResults() == 0)
    return failure();

  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = ifOp.elseYield();
  SmallVector<IfPointerInfo, 4> pointerInfos;

  OpBuilder analysisBuilder(ifOp.getContext());
  analysisBuilder.setInsertionPoint(ifOp);

  // Decide the complete result signature before creating the replacement op.
  // Components shared by both branches remain invariants outside the if;
  // only components whose SSA values differ become new scf.if results.
  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (!env.policy.matches(result.getType()))
      continue;
    if (thenYield.getOperand(idx) == elseYield.getOperand(idx))
      continue;

    FailureOr<DecomposedValue> thenInfo = analyzeValueForIfPlanning(
        thenYield.getOperand(idx), env, analysisBuilder, thenYield.getLoc());
    FailureOr<DecomposedValue> elseInfo = analyzeValueForIfPlanning(
        elseYield.getOperand(idx), env, analysisBuilder, elseYield.getLoc());
    if (failed(thenInfo) || failed(elseInfo))
      continue;

    IfPointerInfo info;
    info.oldIndex = idx;
    info.thenInfo = *thenInfo;
    info.elseInfo = *elseInfo;
    FailureOr<SmallVector<unsigned>> componentIndices =
        env.policy.getIfComponentIndices(info.thenInfo, info.elseInfo);
    if (failed(componentIndices))
      continue;
    info.componentIndices = *componentIndices;
    pointerInfos.push_back(info);
  }

  if (pointerInfos.empty())
    return failure();

  SmallVector<Type> newResultTypes;
  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
      for (Value component :
           getComponentValues(info->thenInfo, info->componentIndices))
        newResultTypes.push_back(component.getType());
      continue;
    }
    newResultTypes.push_back(result.getType());
  }

  bool bodyOk = true;
  auto buildBranch = [&](OpBuilder &branchBuilder, Location loc,
                         bool isThen) -> LogicalResult {
    // Each branch gets an isolated environment because values defined in one
    // branch must never be visible while cloning the other branch.
    RewriteEnv branchEnv = env;
    Block *oldBlock = isThen ? ifOp.thenBlock() : ifOp.elseBlock();
    scf::YieldOp oldYield = isThen ? thenYield : elseYield;
    if (failed(rewriteBodyOps(oldBlock, branchBuilder, branchEnv)))
      return failure();

    SmallVector<Value> newYieldOperands;
    for (auto [idx, oldOperand] : llvm::enumerate(oldYield.getOperands())) {
      if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
        FailureOr<DecomposedValue> branchInfo = decompose(
            oldOperand, branchEnv, branchBuilder, oldYield.getLoc());
        if (failed(branchInfo))
          return failure();
        SmallVector<Value> values =
            getComponentValues(*branchInfo, info->componentIndices);
        SmallVector<Value> plannedValues =
            getComponentValues(info->thenInfo, info->componentIndices);
        if (values.size() != plannedValues.size() ||
            !llvm::all_of(llvm::zip(values, plannedValues), [](auto pair) {
              return std::get<0>(pair).getType() ==
                     std::get<1>(pair).getType();
            }))
          return failure();
        newYieldOperands.append(values.begin(), values.end());
        continue;
      }
      newYieldOperands.push_back(remapValue(oldOperand, branchEnv));
    }
    branchBuilder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);
    return success();
  };

  auto newIfOp =
      builder.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                remapValue(ifOp.getCondition(), env), true);
  newIfOp->setAttrs(ifOp->getAttrs());

  {
    OpBuilder::InsertionGuard guard(builder);
    if (newResultTypes.empty()) {
      newIfOp.thenBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIfOp.thenBlock());
    } else {
      builder.setInsertionPointToStart(newIfOp.thenBlock());
    }
    if (failed(buildBranch(builder, ifOp.getLoc(), /*isThen=*/true)))
      bodyOk = false;
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    if (newResultTypes.empty()) {
      newIfOp.elseBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIfOp.elseBlock());
    } else {
      builder.setInsertionPointToStart(newIfOp.elseBlock());
    }
    if (failed(buildBranch(builder, ifOp.getLoc(), /*isThen=*/false)))
      bodyOk = false;
  }

  if (!bodyOk) {
    newIfOp.erase();
    return failure();
  }

  // Reassemble the pointer immediately after the replacement if. Downstream
  // operations therefore keep their original operand types; decomposition is
  // limited to the control-flow boundary itself.
  builder.setInsertionPointAfter(newIfOp);
  unsigned newResultIndex = 0;
  for (auto [idx, oldResult] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info = findIfInfo(pointerInfos, idx)) {
      SmallVector<Value> componentValues;
      for (unsigned i = 0; i < info->componentIndices.size(); ++i)
        componentValues.push_back(newIfOp.getResult(newResultIndex++));
      FailureOr<DecomposedValue> resultInfo = withComponentValues(
          info->thenInfo, info->componentIndices, componentValues);
      if (failed(resultInfo)) {
        newIfOp.erase();
        return failure();
      }
      Value rebuilt =
          env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
      if (!rebuilt) {
        newIfOp.erase();
        return failure();
      }
      recordDecomposition(oldResult, *resultInfo, rebuilt, env);
      continue;
    }
    env.valueMapping.map(oldResult, newIfOp.getResult(newResultIndex++));
  }

  return success();
}

static LogicalResult rewriteControlFlowOp(Operation *op, OpBuilder &builder,
                                          RewriteEnv &env) {
  // Keep the operation dispatch next to the shared recursive implementation:
  // all supported region operations must obey the same mapping and cleanup
  // rules. Pointer-specific semantics enter only through env.policy.
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewriteForOp(forOp, builder, env);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewriteWhileOp(whileOp, builder, env);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp, builder, env);
  // TODO: Add the frontend-produced scope.scope operation here. Scope support
  // belongs in this shared plumbing rather than in both decompositions.
  return failure();
}

static SmallVector<Value> collectReplacements(Operation *op,
                                              const RewriteEnv &env) {
  SmallVector<Value> replacements;
  replacements.reserve(op->getNumResults());
  for (Value result : op->getResults())
    replacements.push_back(remapValue(result, env));
  return replacements;
}

static LogicalResult
tryDecoupleControlFlowOp(Operation *op, IRRewriter &rewriter,
                         const ControlFlowRewritePolicy &policy) {
  // Build a replacement beside the original operation. The original remains
  // untouched until every result has a valid mapped value, after which the
  // standard rewriter performs the single externally visible replacement.
  RewriteEnv env(policy);
  rewriter.setInsertionPoint(op);
  if (failed(rewriteControlFlowOp(op, rewriter, env)))
    return failure();

  SmallVector<Value> replacements = collectReplacements(op, env);
  if (replacements.size() != op->getNumResults() ||
      llvm::any_of(replacements, [](Value value) { return !value; }))
    return failure();
  rewriter.replaceOp(op, replacements);
  return success();
}

} // namespace

namespace mlir::triton::controlflow {

static void rewriteRegion(Region &region, IRRewriter &rewriter,
                          const ControlFlowRewritePolicy &policy) {
  // Visit outer operations first. A successful rewrite clones and recursively
  // rewrites its nested regions, so descending again would process them twice.
  // If the outer operation has no slot owned by this policy, descend normally
  // to give nested control flow an independent chance to match.
  for (Block &block : region) {
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op) &&
          succeeded(tryDecoupleControlFlowOp(&op, rewriter, policy)))
        continue;

      for (Region &nested : op.getRegions())
        rewriteRegion(nested, rewriter, policy);
    }
  }
}

LogicalResult rewriteControlFlow(ModuleOp module,
                                 const ControlFlowRewritePolicy &policy) {
  // The environment is deliberately recreated per top-level candidate. No
  // address-readiness marker or analysis state escapes into the IR or the next
  // decomposition invocation.
  IRRewriter rewriter(module.getContext());
  for (Region &region : module->getRegions())
    rewriteRegion(region, rewriter, policy);
  // TODO: Once policy analysis reports typed failures, propagate unsupported
  // target slots and materialization-cleanup failures through this result.
  return success();
}

} // namespace mlir::triton::controlflow
