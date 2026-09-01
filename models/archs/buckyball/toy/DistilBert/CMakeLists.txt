# DistilBert on the toy single-core chip (buckyball arch layout).
#
# toy is a buckyball bank target (pb_to_target_registry.py
# --print-bank-targets), so the subgraph matmuls lower through the buckyball
# tile -> bank-ssa -> bank-ssa-intrinsics pipeline. Precedents:
# archs/buckyball/pebble/LeNet (single-core model layout) and
# archs/buckyball/goban/Bert.
#
# The upstream gemmini dialect pipeline (-convert-linalg-to-gemmini
# -lower-gemmini) is NOT usable on this chip: it emits Berkeley-Gemmini ISA
# encodings (CONFIG=0, FLUSH=7, LOOP_WS=8, LOOP_WS_CONFIG_BOUNDS=9, ...,
# see buddy-mlir llvm/.../RISCVInstrInfoXGemmini.td) that toy's GemminiBall
# registry (funct7 2/3/53/66/67/80-87/96-105,
# examples/cores/toy/configs/balldomains/default.toml) does not declare, and
# its WS-dataflow config semantics (addrs_ab/dc packing, nonzero D) differ
# from the ball's bemu contract (OS dataflow only, zero D, one address per
# config instruction).
#
# The run target name is the one bb-tests/workloads/scripts/build.py
# registers in _MODELS for the "distilbert" CLI model key.

set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_DISTILBERT_DIR ${MODEL_DIR}/DistilBert)
set(MODEL_DISTILBERT_GENERATED_DIR ${MODEL_DISTILBERT_DIR}/output)
set(BUCKYBALL_DISTILBERT_DIR ${BUCKYBALL_DIR}/DistilBert)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODELTEST_LIB_DIR})

# forward.o: TOSA -> linalg -> scalar loops on the toy core RISC-V pipeline
# (embedding lookup, softmax, layernorm, gelu, residual adds).
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

# subgraph0.o: scalar host fallback on the toy single core. Neither
# compiler path to toy's mounted balls is usable today: the upstream gemmini
# dialect pipeline emits Berkeley-Gemmini encodings toy's registry does not
# declare (funct7 9 panic in bemu), and toy mounts no ball with a tile
# lowering hook (BuckyballBallLoweringHooks.inc TILE section is empty for
# toy; smatmul/transpose/relu own them), so the buckyball tile pipeline
# cannot lower here either (tile.tile_transpose survives to
# buddy-translate). Matmuls therefore lower to plain scalar loops next to
# the other host-fallback ops — see the ball/core follow-up specs recorded
# in the PR description for restoring an accelerator path.
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DISTILBERT_GENERATED_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
          > subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_DISTILBERT_GENERATED_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))" |
          ${BUDDY_BINARY_DIR}/buddy-opt
            -extend-trace-to-linalg
            -eliminate-empty-tensors
            -convert-tensor-to-linalg
            -convert-elementwise-to-linalg
            -one-shot-bufferize="bufferize-function-boundaries"
            -llvm-request-c-wrappers
            -convert-trace-to-llvm="cycle-trace"
            -expand-strided-metadata
            -convert-linalg-to-loops
            -lower-affine
            -convert-scf-to-cf
            -convert-cf-to-llvm
            -buffer-deallocation-simplification
            -bufferization-lower-deallocations
            -convert-math-to-llvm
            -convert-math-to-libm
            -convert-arith-to-llvm
            -convert-func-to-llvm
            -finalize-memref-to-llvm
            -reconcile-unrealized-casts |
          ${BUDDY_BINARY_DIR}/buddy-translate --buddy-to-llvmir |
          ${BUDDY_BINARY_DIR}/buddy-llc -filetype=obj -mtriple=riscv64 -O2 -mattr=${BUCKYBALL_RISCV_MATTR} -float-abi=hard -o subgraph0.o
        DEPENDS ${MODEL_DISTILBERT_GENERATED_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballDistilBert STATIC forward.o subgraph0.o)
set_target_properties(BuckyballDistilBert PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballDistilBert distilbert-model-build)

add_executable(buddy-buckyball-distilbert-run ${MODEL_DISTILBERT_DIR}/distilbert-main.cpp)
add_dependencies(buddy-buckyball-distilbert-run BuckyballDistilBert)

target_include_directories(buddy-buckyball-distilbert-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
# DistilBert uses no DIP (resize) ops: ModelTestDIP_riscv is deliberately not
# linked, so the DIP.mlir toolchain dependency stays out of this target's
# ninja closure.
set(BUDDY_DISTILBERT_LIBS BuckyballDistilBert ModelTestCRunnerUtils)
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

# Perfetto evidence chain (design doc §4.7) is intentionally NOT wired here
# yet: -extend-trace-to-linalg only extends trace.start scopes, which the
# model-side import (import-distilbert.py) must insert from a trace TOML —
# LeNet precedent: models/LeNet/trace/trace.toml feeding its import. Until
# the model side re-imports with trace scopes, no cycle files are produced
# and a perfetto target would fail with "no cycle trace files found". The
# bbdev PERFETTO_TARGETS mapping for buddy-buckyball-distilbert-run is
# therefore also absent for now. Follow-up (workload line): re-import with a
# trace TOML, then re-add add_buckyball_perfetto_target(...) mirroring
# pebble/LeNet and restore the bbdev mapping.
