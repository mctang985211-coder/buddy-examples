# BertTiny on the toy single-core chip (buckyball arch layout).
#
# toy is a buckyball bank target (pb_to_target_registry.py
# --print-bank-targets) and mounts GemminiBall / TraceBall / Mxfp2IntBall /
# GeluBall in examples/cores/toy/configs/balldomains/default.toml
# (GELU funct7=56 is registered for GeluBall, ballId=3). Precedent:
# archs/buckyball/toy/DistilBert (single-core model layout copied below) and
# archs/buckyball/goban/Bert.
#
# GELU reachability on this chip (stage 0 evidence, PR #7 for DistilBert and
# this BertTiny bind): the graph-level exact-erf GELU can NOT reach the
# mounted GeluBall. Three concrete break points in the toy pipeline:
#
# 1) e2e/models/archs/buckyball/CMakeLists.txt:19-29 only sets
#    BUCKYBALL_CONVERT_TILE_TO_BUCKYBALL / -assign-physical-banks under
#    `if(BUCKYBALL_ARCH STREQUAL "goban" OR ... "pebble")`; the toy layout
#    never runs -convert-tile-to-buckyball, so subgraph lowering goes straight
#    to scalar host RISC-V loops (`-convert-linalg-to-loops` ->
#    `-convert-math-to-libm` -> `erff`).
# 2) The toy core-local tile pass
#    examples/cores/toy/compiler/src/Conversion/LowerTileToBuckyball/
#    LowerTileToBuckyball.cpp:44 only registers TileConv2dLowering (the conv2d
#    fallback). Its generated chip `BUCKYBALL_TILE_HOOK` section
#    (compiler/thirdparty/buddy-mlir/build/toy/external_dialects/
#    BuckyballBallLoweringHooks.inc) is empty, and examples/balls/gelu has no
#    compiler/src/Conversion/LowerTileToBuckyball/ directory, so GeluBall has
#    no populate<tile> hook: even with the tile pipeline the erf is not fused.
# 3) toy bank-SSA / assign populate are empty stubs:
#    Examples/.../LowerBuckyball/Patterns/ToyLowerBuckyballToBankSSAPatterns.cpp:9
#    and ToyAssignPhysicalBankPatterns.cpp:9-10 are `(void)` no-ops, so no
#    buckyball.ball op can pass those two lowerings either.
# On the graph side BertTiny's intermediate gelu is imported as a 5-op
# exact-erf TOSA sequence (scale x by 1/sqrt(2) -> math.erf -> +1 -> *0.5 ->
# *x, e.g. tosa.mul/tosa.add/math.erf in subgraph0.mlir around the layer/head
# boundaries); buddy-opt exposes no fusion/recognition pattern that folds the
# sequence back to a single buckyball.gelu on toy. subgraph0.o therefore lowers
# to plain scalar loops and the three math.erf become glibc erff calls (see
# the bound PR description for the D-class derivation specs).
#
# The run target name is the one bb-tests/workloads/scripts/build.py
# registers in _MODELS for the "berttiny" CLI model key.

set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_BERTTINY_DIR ${MODEL_DIR}/BertTiny)
set(MODEL_BERTTINY_GENERATED_DIR ${MODEL_BERTTINY_DIR}/output)
set(BUCKYBALL_BERTTINY_DIR ${BUCKYBALL_DIR}/BertTiny)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODELTEST_LIB_DIR})

# forward.o: forward.mlir is a thin wrapper that unpacks the packed arg0 into
# private subviews and calls the private @subgraph0 symbol (defined by
# subgraph0.o). It contains no compute itself; the scalar passes below are
# no-ops on the memref slice/cast ops and just hand the module to LLVM.
add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTTINY_GENERATED_DIR}/forward.mlir
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
        DEPENDS ${MODEL_BERTTINY_GENERATED_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

# subgraph0.o: scalar host fallback on the toy single core (same routing as
# the toy/DistilBert round-2 layout). Neither compiler path to toy's mounted
# balls reaches GeluBall for the exact-erf gelu sequence (see header): toy
# mounts no ball with a tile lowering hook (BUCKYBALL_TILE_HOOK TILE section
# is empty for toy), and toy's bank-SSA / assign populate hooks are empty
# stubs, so the buckyball ball pipeline cannot lower this module either.
# Every op - the three math.erf exact gelu terminations and the matmuls -
# therefore lowers to plain scalar loops on the host RISC-V core, with the
# gelu falling to glibc erff via -convert-math-to-libm. The accelerator
# follow-up (tile hook + toy bank-SSA/assign) is recorded as derived specs
# D-gelu-* in the binding PR description.
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTTINY_GENERATED_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
          > subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTTINY_GENERATED_DIR}/subgraph0.mlir
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
        DEPENDS ${MODEL_BERTTINY_GENERATED_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballBertTiny STATIC forward.o subgraph0.o)
set_target_properties(BuckyballBertTiny PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballBertTiny berttiny-model-build)

add_executable(buddy-buckyball-berttiny-run ${MODEL_BERTTINY_DIR}/berttiny-main.cpp)
add_dependencies(buddy-buckyball-berttiny-run BuckyballBertTiny)

target_include_directories(buddy-buckyball-berttiny-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
# BertTiny uses no DIP (resize) ops: ModelTestDIP_riscv is deliberately not
# linked, so the DIP.mlir toolchain dependency stays out of this target's
# ninja closure.
set(BUDDY_BERTTINY_LIBS BuckyballBertTiny ModelTestCRunnerUtils)
target_link_libraries(buddy-buckyball-berttiny-run -static ${BUDDY_BERTTINY_LIBS})

set(BERTTINY_OUTPUT_DIR ${BUCKYBALL_OUTPUT_DIR}/BertTiny)

add_custom_command(TARGET buddy-buckyball-berttiny-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTTINY_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-berttiny-run>
          ${BERTTINY_OUTPUT_DIR}/buddy-buckyball-berttiny-run
  COMMENT "Copying Buckyball BertTiny executable to output directory"
  VERBATIM
)

# Runtime deps: import artifacts (arg0/arg1.data) come from the model-side
# build; vocab.txt is committed in the model directory. bemu --sim runs with
# the binary's directory as cwd, so the artifacts land next to the ELF.
add_custom_target(sync-buckyball-berttiny-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTTINY_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTTINY_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTTINY_GENERATED_DIR}/arg0.data
          ${BERTTINY_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTTINY_GENERATED_DIR}/arg1.data
          ${BERTTINY_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTTINY_DIR}/vocab.txt
          ${BERTTINY_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_BERTTINY_GENERATED_DIR}/arg0.data
          ${MODEL_BERTTINY_GENERATED_DIR}/arg1.data
          ${MODEL_BERTTINY_DIR}/vocab.txt
  COMMENT "Syncing BertTiny runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-berttiny-run sync-buckyball-berttiny-deps)

# Perfetto evidence chain (design doc §4.7) is intentionally NOT wired here
# yet: -extend-trace-to-linalg only extends trace.start scopes, which the
# model-side import (import-berttiny.py) must insert from a trace TOML.
# Until the model side re-imports with trace scopes, no cycle files are
# produced and a perfetto target would fail with "no cycle trace files found".
# The bbdev PERFETTO_TARGETS mapping for buddy-buckyball-berttiny-run is
# therefore also absent.
