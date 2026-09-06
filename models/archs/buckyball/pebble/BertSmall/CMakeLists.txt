# BertSmall (prajjwal1/bert-small) layout for the pebble chip.
#
# Host-linalg form (upstream precedent: pebble/llama2/CMakeLists.txt and
# pebble/StableDiffusion/CMakeLists.txt, which never pass
# -convert-linalg-to-tile / -convert-tile-to-buckyball /
# buckyball_assign_physical_banks — the in-tree shape for a transformer whose
# ops stay on the host CPU). archs/buckyball/CMakeLists.txt defines the
# tile->buckyball machinery variables for goban and pebble, but this layout
# deliberately does not use them: the importer's subgraph0.mlir contains an
# i1 (attention-mask) transpose site that pebble's
# MemTransposeToBankSSAPatterns rejects (round-1 CI: "transpose only supports
# 8-bit or 32-bit elements"; derived spec D-tr-1 in PR #25). Under this form
# every op of both subgraphs lowers to riscv64 host loops; no buckyball
# dialect op is ever created.
#
# BertSmall's importer (import-bertsmall.py) emits exactly forward.mlir +
# subgraph0.mlir into <model>/output/, so unlike goban/Bert there is no
# BERT_CORE_COUNT subgraph loop: the driver is bertsmall-main.cpp and the
# subgraph count is one.

set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_BERTSMALL_DIR ${MODEL_DIR}/BertSmall)
set(MODEL_BERTSMALL_GENERATED_DIR ${MODEL_BERTSMALL_DIR}/output)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODEL_BERTSMALL_DIR}/include)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTSMALL_GENERATED_DIR}/forward.mlir
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
        DEPENDS ${MODEL_BERTSMALL_GENERATED_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTSMALL_GENERATED_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -eliminate-empty-tensors
            -convert-elementwise-to-linalg
            -convert-tensor-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-trace-to-llvm="cycle-trace"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-vector-to-scf
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -convert-vector-to-llvm
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=${BUCKYBALL_RISCV_MATTR} -float-abi=hard -o subgraph0.o
        DEPENDS ${MODEL_BERTSMALL_GENERATED_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballBERTSMALL STATIC forward.o subgraph0.o)
set_target_properties(BuckyballBERTSMALL PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballBERTSMALL bertsmall-model-build)
add_executable(buddy-buckyball-bertsmall-run ${MODEL_BERTSMALL_DIR}/bertsmall-main.cpp)
add_dependencies(buddy-buckyball-bertsmall-run BuckyballBERTSMALL)

target_include_directories(buddy-buckyball-bertsmall-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
set(BUDDY_BERTSMALL_LIBS BuckyballBERTSMALL ModelTestCRunnerUtils ModelTestDIP_riscv)
target_link_libraries(buddy-buckyball-bertsmall-run -static ${BUDDY_BERTSMALL_LIBS})

set(BERTSMALL_OUTPUT_DIR ${BUCKYBALL_OUTPUT_DIR}/BertSmall)

add_custom_command(TARGET buddy-buckyball-bertsmall-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTSMALL_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-bertsmall-run>
          ${BERTSMALL_OUTPUT_DIR}/buddy-buckyball-bertsmall-run
  COMMENT "Copying Buckyball BertSmall executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-bertsmall-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTSMALL_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTSMALL_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTSMALL_GENERATED_DIR}/arg0.data
          ${BERTSMALL_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTSMALL_GENERATED_DIR}/arg1.data
          ${BERTSMALL_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTSMALL_DIR}/vocab.txt
          ${BERTSMALL_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_BERTSMALL_GENERATED_DIR}/arg0.data
          ${MODEL_BERTSMALL_GENERATED_DIR}/arg1.data
          ${MODEL_BERTSMALL_DIR}/vocab.txt
  COMMENT "Syncing BertSmall runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-bertsmall-run sync-buckyball-bertsmall-deps)
