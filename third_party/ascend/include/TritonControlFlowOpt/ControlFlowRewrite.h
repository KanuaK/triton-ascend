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

#ifndef TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H
#define TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::controlflow {

/// Policy-owned description of one value crossing a control-flow boundary.
///
/// `components` are runtime values that a policy may place in an expanded SCF
/// signature. `invariants` and `attributes` are retained by the policy to
/// rebuild the original value. Their layout is private to that policy; the
/// shared rewrite never interprets pointer-specific fields.
struct DecomposedValue {
  Type originalType;
  SmallVector<Value> components;
  SmallVector<Value> invariants;
  SmallVector<Attribute> attributes;
};

/// Read-only view of the SSA mapping and decompositions visible at the current
/// rewrite point. Policies use it while recursively analyzing pointer
/// producers; the state exists only for one control-flow rewrite attempt.
class ControlFlowRewriteContext {
public:
  ControlFlowRewriteContext(
      const IRMapping &valueMapping,
      const llvm::DenseMap<Value, DecomposedValue> &decomposedValues)
      : valueMapping(valueMapping), decomposedValues(decomposedValues) {}

  Value remap(Value value) const;
  const DecomposedValue *lookup(Value value) const;

private:
  const IRMapping &valueMapping;
  const llvm::DenseMap<Value, DecomposedValue> &decomposedValues;
};

/// Pointer-semantics interface implemented by each decomposition policy.
///
/// The policy decides how its value is decomposed and rebuilt, which components
/// cross loop/if boundaries, and whether two decompositions share a compatible
/// invariant schema. It is not an IR marker and carries no state between
/// policy invocations.
class ControlFlowRewritePolicy {
public:
  virtual ~ControlFlowRewritePolicy() = default;

  virtual bool matches(Type type) const = 0;

  virtual FailureOr<DecomposedValue>
  decompose(Value value, const ControlFlowRewriteContext &context,
            OpBuilder &builder, Location loc) const = 0;

  virtual Value recompose(const DecomposedValue &value, OpBuilder &builder,
                          Location loc) const = 0;

  /// Returns component indices carried in place of a loop iter-argument.
  virtual FailureOr<SmallVector<unsigned>>
  getLoopComponentIndices(const DecomposedValue &value) const = 0;

  virtual bool areLoopStatesCompatible(const DecomposedValue &initial,
                                       const DecomposedValue &next) const = 0;

  /// Validates an if-merge and returns the components selected by the if.
  virtual FailureOr<SmallVector<unsigned>>
  getIfComponentIndices(const DecomposedValue &thenValue,
                        const DecomposedValue &elseValue) const = 0;

  /// Whether results of this ordinary operation need immediate decomposition
  /// after cloning so later operations can reuse their exact component state.
  virtual bool shouldDecomposeOperation(Operation *op) const = 0;

  /// Optional block-pointer closed form used for zero-based unit-step loops.
  virtual FailureOr<SmallVector<Value>>
  matchForInductionDeltas(scf::ForOp forOp,
                          const DecomposedValue &initial,
                          unsigned iterArgIndex, Value yieldOperand) const = 0;

  virtual FailureOr<DecomposedValue>
  applyForInductionDeltas(const DecomposedValue &initial,
                          ArrayRef<Value> deltas, Value inductionVar,
                          const ControlFlowRewriteContext &context,
                          OpBuilder &builder, Location loc) const = 0;
};

/// Rewrites supported SCF operations from outermost to innermost.
///
/// This function owns the mechanical control-flow work: signature expansion,
/// region cloning, terminator rewriting, nested recursion and result
/// replacement. Pointer semantics remain selected by `policy` so the two
/// policies do not duplicate the SCF plumbing.
LogicalResult rewriteControlFlow(ModuleOp module,
                                 const ControlFlowRewritePolicy &policy);

} // namespace mlir::triton::controlflow

#endif // TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H
