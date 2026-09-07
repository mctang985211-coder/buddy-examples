# Pythia (EleutherAI/pythia-70m) layout for the pebble chip.
#
# Template: pebble/BertMedium/CMakeLists.txt — the phase7 realization of the
# same family bind (there is NO upstream pebble layout for Pythia: the tree has
# only llama2/BertSmall/BertMedium host-linalg and DeepSeekR1 tile-lane
# transformer layouts; BertMedium is the closest importer/output shape).
#
# Host-linalg form (upstream precedent: pebble/llama2 and pebble/BertMedium,
# which never pass -convert-linalg-to-tile / -convert-tile-to-buckyball /
# buckyball_assign_physical_banks): every op of both generated graphs lowers to
# riscv64 host loops and no buckyball dialect op is ever created. archs/
# CMakeLists.txt defines the tile->buckyball machinery variables for pebble,
# but this layout deliberately does not use them. Grounding for that choice is
# the bind-round observation recorded in the PR body (phase8 bind):
#   - transpose model-path trigger: VERIFIED locally at compile level — the
#     pebble tile lane converts all 38 f32 transpose sites to
#     buckyball.mem_transpose and passes -lower-buckyball-to-bank-ssa +
#     -assign-physical-banks (no i1 site this time; the phase6 i1 breakpoint
#     does not apply), but the lane expands this 70M-param graph to ~1.0M
#     lines of bank-SSA IR with 87,790 memref.alloc / 43,597 mvin+mvout pairs
#     — an unshippable compile/runtime carrier for a host-resident parameter
#     graph without upstream compiler work (D-p-1), so the layout stays on the
#     proven host-linalg form (same call as phase6 round-2 / phase7).
#   - LayerNorm model-path trigger: BREAKPOINT — neither buddy-mlir's
#     -convert-linalg-to-tile (no layernorm pattern) nor pebble's
#     tile->buckyball pass (transpose/quant/matmul/mega only) can raise the 13
#     LayerNorm sites to LayerNormBall (funct7 73); the composite stays inside
#     linalg.generic and falls to host loops (D-p-2, core-compiler territory).
# The driver's per-position argmax fail-hard gate is the layout's correctness
# contract on the host carrier; the canonical reference lives in the workload
# dir (reference/pythia_logits_f32.bin + HANDOFF.md).
#
# Pythia differs from BertMedium in importer output placement: the generated
# artifacts land in the model directory ROOT (models/models/Pythia/forward.mlir,
# subgraph0.mlir, arg0.data — see Pythia/CMakeLists.txt pythia-model-build),
# not in an output/ subdir, and there is no arg1.data blob: the driver
# buddy-pythia-main.cpp hardcodes the fixed 16 input ids (no C++ gpt-neox
# tokenizer in buddy). Runtime deps are arg0.data + vocab.txt only.
# @forward signature (read from the emitted forward.mlir):
#   memref<70426632xf32> + memref<1x16xi64> -> memref<1x16x50304xf32>
# arg0.data = 70,426,632 float32 (281,706,528 B) = 70,426,624 trainable params
# + 8-elem rotary inv_freq buffer (Pythia/HANDOFF.md).

set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_PYTHIA_DIR ${MODEL_DIR}/Pythia)
set(BUCKYBALL_PYTHIA_DIR ${BUCKYBALL_DIR}/Pythia)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)

add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_PYTHIA_DIR}/forward.mlir
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
  DEPENDS ${MODEL_PYTHIA_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

add_custom_command(
  OUTPUT subgraph0.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_PYTHIA_DIR}/subgraph0.mlir
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
  DEPENDS ${MODEL_PYTHIA_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballPYTHIA STATIC forward.o subgraph0.o)
set_target_properties(BuckyballPYTHIA PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballPYTHIA pythia-model-build)
add_executable(buddy-buckyball-pythia-run ${MODEL_PYTHIA_DIR}/buddy-pythia-main.cpp)
add_dependencies(buddy-buckyball-pythia-run BuckyballPYTHIA)

target_include_directories(buddy-buckyball-pythia-run PRIVATE ${INTERFACE_DIR})
set(BUDDY_PYTHIA_LIBS BuckyballPYTHIA ModelTestCRunnerUtils)
target_link_libraries(buddy-buckyball-pythia-run -static ${BUDDY_PYTHIA_LIBS})

set(PYTHIA_OUTPUT_DIR ${BUCKYBALL_OUTPUT_DIR}/Pythia)

add_custom_command(TARGET buddy-buckyball-pythia-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHIA_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-pythia-run>
          ${PYTHIA_OUTPUT_DIR}/buddy-buckyball-pythia-run
  COMMENT "Copying Buckyball Pythia executable to output directory"
  VERBATIM
)

add_custom_target(sync-buckyball-pythia-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHIA_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHIA_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_PYTHIA_DIR}/arg0.data
          ${PYTHIA_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_PYTHIA_DIR}/vocab.txt
          ${PYTHIA_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_PYTHIA_DIR}/arg0.data
          ${MODEL_PYTHIA_DIR}/vocab.txt
  COMMENT "Syncing Pythia runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-pythia-run sync-buckyball-pythia-deps)
