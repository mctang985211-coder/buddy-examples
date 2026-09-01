# DistilBert on the toy single-core chip (buckyball arch layout).
#
# toy core mounts the Gemmini ball (balldomain ballId 0), so the subgraph
# matmuls lower through the gemmini dialect (precedent: archs/gemmini/Bert),
# while everything the toy core runs on its scalar RISC-V pipeline --
# embedding lookups, softmax, layernorm, gelu, residual adds -- lowers to
# plain linalg loops (precedent: the forward.o pipeline shared by
# goban/Bert and gemmini/Bert).
#
# The run target name is the one bb-tests/workloads/scripts/build.py
# registers in _MODELS for the "distilbert" CLI model key.

set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_DISTILBERT_DIR ${MODEL_DIR}/DistilBert)
set(MODEL_DISTILBERT_GENERATED_DIR ${MODEL_DISTILBERT_DIR}/output)
set(BUCKYBALL_DISTILBERT_DIR ${BUCKYBALL_DIR}/DistilBert)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODELTEST_LIB_DIR})

# forward.o: TOSA -> linalg -> scalar loops on the toy core RISC-V pipeline.
add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DISTILBERT_GENERATED_DIR}/forward.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith), empty-tensor-to-alloc-tensor, convert-elementwise-to-linalg)" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -buffer-deallocation-simplification
            -convert-linalg-to-loops
            -eliminate-empty-tensors
            -llvm-request-c-wrappers
            -convert-math-to-llvm
            -convert-math-to-libm
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-arith-to-llvm
            -expand-strided-metadata
            -finalize-memref-to-llvm
            -convert-func-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=${BUCKYBALL_RISCV_MATTR} -float-abi=hard -o forward.o
        DEPENDS ${MODEL_DISTILBERT_GENERATED_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

# subgraph0.o: matmuls onto the toy chip's Gemmini ball via the gemmini
# dialect; non-matmul glue stays on scalar loops.
add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DISTILBERT_GENERATED_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -batchmatmul-optimize
            > subgraph0_loops.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DISTILBERT_GENERATED_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -lower-gemmini
            > subgraph0_gemmini.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DISTILBERT_GENERATED_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -convert-linalg-to-gemmini
            -expand-strided-metadata
            -convert-linalg-to-loops
            -convert-scf-to-cf
            -llvm-request-c-wrappers
            -lower-gemmini
            -ownership-based-buffer-deallocation
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=+xgemmini,+D -float-abi=hard -o subgraph0.o
        DEPENDS ${MODEL_DISTILBERT_GENERATED_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballDistilBert STATIC forward.o subgraph0.o)
set_target_properties(BuckyballDistilBert PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballDistilBert distilbert-model-build)

add_executable(buddy-buckyball-distilbert-run ${MODEL_DISTILBERT_DIR}/distilbert-main.cpp)
add_dependencies(buddy-buckyball-distilbert-run BuckyballDistilBert)

target_include_directories(buddy-buckyball-distilbert-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
set(BUDDY_DISTILBERT_LIBS BuckyballDistilBert ModelTestCRunnerUtils ModelTestDIP_riscv)
target_link_libraries(buddy-buckyball-distilbert-run -static ${BUDDY_DISTILBERT_LIBS})

set(DISTILBERT_OUTPUT_DIR ${BUCKYBALL_OUTPUT_DIR}/DistilBert)

add_custom_command(TARGET buddy-buckyball-distilbert-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DISTILBERT_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-distilbert-run>
          ${DISTILBERT_OUTPUT_DIR}/buddy-buckyball-distilbert-run
  COMMENT "Copying Buckyball DistilBert executable to output directory"
  VERBATIM
)

# Runtime deps: import artifacts (arg0/arg1.data) come from the model-side
# build; vocab.txt is committed in the model directory. bemu --sim runs with
# the binary's directory as cwd, so the artifacts land next to the ELF.
add_custom_target(sync-buckyball-distilbert-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DISTILBERT_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DISTILBERT_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_DISTILBERT_GENERATED_DIR}/arg0.data
          ${DISTILBERT_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_DISTILBERT_GENERATED_DIR}/arg1.data
          ${DISTILBERT_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_DISTILBERT_DIR}/vocab.txt
          ${DISTILBERT_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_DISTILBERT_GENERATED_DIR}/arg0.data
          ${MODEL_DISTILBERT_GENERATED_DIR}/arg1.data
          ${MODEL_DISTILBERT_DIR}/vocab.txt
  COMMENT "Syncing DistilBert runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-distilbert-run sync-buckyball-distilbert-deps)
