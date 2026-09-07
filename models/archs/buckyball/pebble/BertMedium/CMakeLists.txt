# BertMedium (prajjwal1/bert-medium) layout for the pebble chip.
#
# Template: pebble/BertSmall/CMakeLists.txt — the team's phase6 realization of
# the same family bind (there is NO upstream pebble layout for any Bert model:
# upstream has Bert only under goban/; pebble/BertSmall is the in-fork phase6
# realization this file mirrors). Structural reference: pebble/LeNet/
# CMakeLists.txt (riscv64-linux g++ + -static Linux-ABI ELF).
#
# Host-linalg form (upstream precedent: pebble/llama2/CMakeLists.txt and
# pebble/StableDiffusion/CMakeLists.txt, which never pass
# -convert-linalg-to-tile / -convert-tile-to-buckyball /
# buckyball_assign_physical_banks — the in-tree shape for a transformer whose
# ops stay on the host CPU). archs/buckyball/CMakeLists.txt defines the
# tile->buckyball machinery variables for goban and pebble, but this layout
# deliberately does not use them: the importer's subgraph0.mlir contains an
# i1 (attention-mask) transpose site that pebble's
# MemTransposeToBankSSAPatterns rejects (BertSmall round-1 CI: "transpose only
# supports 8-bit or 32-bit elements"; elemsPerBankRow() -> 0 for i1 at
# MemTransposeToBankSSAPatterns.cpp:145; derived spec D-tr-1, fork PR #25).
# BertMedium's subgraph0.mlir carries the same i1 mask-transpose shape
# (tosa.cast -> i1 at the mask compare) plus 82 f32 transpose sites and 18
# LayerNorm sites; TransposeBall (funct7 49) and LayerNormBall (funct7 73 /
# ballId 10) are registered on pebble at the ISA layer, but the model-path
# tile->ball trigger is unverified and this layout stays on the proven
# host-linalg form, so every op of both subgraphs lowers to riscv64 host
# loops and no buckyball dialect op is ever created.
#
# BertMedium differs from BertSmall only in encoder layer count (4 -> 8): the
# importer emits exactly forward.mlir + subgraph0.mlir into <model>/output/
# (same as BertSmall — no BERT_CORE_COUNT subgraph loop, one subgraph), and the
# packed parameter buffer is larger: arg0.data = 41,404,730 float32 =
# 165,618,920 B (BertMedium/HANDOFF.md), arg1.data = 512 int64 = 4,096 B. The
# driver is bertmedium-main.cpp (forward signature
# memref<41404730xf32> + memref<512xi64> + 3x memref<1x11xi64>
# -> memref<1x11x30522xf32>, fixed case SEQ_LEN = 11).

set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_BERTMEDIUM_DIR ${MODEL_DIR}/BertMedium)
set(MODEL_BERTMEDIUM_GENERATED_DIR ${MODEL_BERTMEDIUM_DIR}/output)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODEL_BERTMEDIUM_DIR}/include)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTMEDIUM_GENERATED_DIR}/forward.mlir
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
        DEPENDS ${MODEL_BERTMEDIUM_GENERATED_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTMEDIUM_GENERATED_DIR}/subgraph0.mlir
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
        DEPENDS ${MODEL_BERTMEDIUM_GENERATED_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballBERTMEDIUM STATIC forward.o subgraph0.o)
set_target_properties(BuckyballBERTMEDIUM PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballBERTMEDIUM bertmedium-model-build)
add_executable(buddy-buckyball-bertmedium-run ${MODEL_BERTMEDIUM_DIR}/bertmedium-main.cpp)
add_dependencies(buddy-buckyball-bertmedium-run BuckyballBERTMEDIUM)

target_include_directories(buddy-buckyball-bertmedium-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
set(BUDDY_BERTMEDIUM_LIBS BuckyballBERTMEDIUM ModelTestCRunnerUtils ModelTestDIP_riscv)
target_link_libraries(buddy-buckyball-bertmedium-run -static ${BUDDY_BERTMEDIUM_LIBS})

set(BERTMEDIUM_OUTPUT_DIR ${BUCKYBALL_OUTPUT_DIR}/BertMedium)

add_custom_command(TARGET buddy-buckyball-bertmedium-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTMEDIUM_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-bertmedium-run>
          ${BERTMEDIUM_OUTPUT_DIR}/buddy-buckyball-bertmedium-run
  COMMENT "Copying Buckyball BertMedium executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-bertmedium-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTMEDIUM_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTMEDIUM_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTMEDIUM_GENERATED_DIR}/arg0.data
          ${BERTMEDIUM_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTMEDIUM_GENERATED_DIR}/arg1.data
          ${BERTMEDIUM_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTMEDIUM_DIR}/vocab.txt
          ${BERTMEDIUM_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_BERTMEDIUM_GENERATED_DIR}/arg0.data
          ${MODEL_BERTMEDIUM_GENERATED_DIR}/arg1.data
          ${MODEL_BERTMEDIUM_DIR}/vocab.txt
  COMMENT "Syncing BertMedium runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-bertmedium-run sync-buckyball-bertmedium-deps)
