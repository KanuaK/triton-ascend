add_triton_library(TritonMemoryAccessCore
  OpFoldResultUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
)

add_triton_library(TritonMemoryAccessAnalysis
  MaskAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
  TritonMemoryAccessCore
)

# A consumer-independent numerical pointer decomposition. Both libraries are
# delivered together, but the ordinary analysis does not depend on block or
# conversion targets.
add_triton_library(TritonPointerAnalysis
  PointerArithmetic.cpp
  PointerAnalysisTransfer.cpp
  PointerAnalysis.cpp

  LINK_LIBS PUBLIC
  LLVMCore
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRSupport
  TritonIR
  TritonMemoryAccessCore
)

add_triton_library(TritonBlockPointerAnalysis
  BlockPointerAnalysis.cpp

  LINK_LIBS PUBLIC
  LLVMCore
  MLIRArithDialect
  MLIRIR
  MLIRSupport
  TritonIR
  TritonPointerAnalysis
)
