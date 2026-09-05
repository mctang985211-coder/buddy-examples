# BertMini (prajjwal1/bert-mini) on the toy single-core chip (buckyball arch
# layout). Precedent: archs/buckyball/toy/BertTiny (copy source) and
# archs/buckyball/toy/DistilBert (single-core model layout).
#
# toy mounts GemminiBall / TraceBall / Mxfp2IntBall / GeluBall / LayerNormBall
# in examples/cores/toy/configs/balldomains/default.toml (GELU funct7=56
# ballId=3; LAYERNORM funct7=69 ballId=4, inBW=2/outBW=1). LayerNormBall
# landed in S3 (examples/balls/layernorm, buckyball.layernorm op contract:
# per-row fp32 LN over x_bank(BANK0)+param_bank(BANK1, gamma[0,C) beta[C,2C)),
# out_bank(BANK2), N=R*C in BB_ITER, C in rs2, eps=1e-12 fixed in the op).
#
# BALL UNREACHABILITY on this chip (stage-0 evidence re-grounded this bind;
# BertMini shares the BertTiny verdict for gelu and extends it to layernorm):
# neither the gelu nor the layernorm of this model graph can reach the mounted
# GeluBall / LayerNormBall. Concrete break points in the toy pipeline:
#
# 1) The toy layout never runs -convert-tile-to-buckyball:
#    e2e/models/archs/buckyball/CMakeLists.txt:19-29 only defines
#    BUCKYBALL_CONVERT_TILE_TO_BUCKYBALL / -assign-physical-banks under
#    `if(BUCKYBALL_ARCH STREQUAL "goban" OR ... "pebble")`; toy layouts lower
#    subgraphs straight to scalar host RISC-V loops
#    (`-convert-linalg-to-loops` -> `-convert-math-to-libm`).
# 2) Even on a tile pipeline nothing would fuse: toy's core-local tile pass
#    examples/cores/toy/compiler/src/Conversion/LowerTileToBuckyball/
#    LowerTileToBuckyball.cpp registers only TileConv2dLowering (class :44,
#    patterns.add<TileConv2dLowering> :172-175). The generated toy hook file
#    compiler/thirdparty/buddy-mlir/build/toy/external_dialects/
#    BuckyballBallLoweringHooks.inc has an EMPTY BUCKYBALL_TILE_HOOK section
#    (:15-17), and the only balls owning populate*TileLoweringPatterns are
#    relu/smatmul/transpose (examples/balls/{relu,smatmul,transpose}/compiler/
#    src/Conversion/LowerTileToBuckyball/Tile*Patterns.cpp); the gelu and
#    layernorm ball compiler trees contain no Conversion/LowerTileToBuckyball/
#    directory at all.
# 3) toy bank-SSA / assign populate are empty stubs:
#    examples/cores/toy/compiler/src/Conversion/LowerBuckyball/Patterns/
#    ToyLowerBuckyballToBankSSAPatterns.cpp:8-9 and
#    ToyAssignPhysicalBankPatterns.cpp:8-10 are `(void)` no-ops, so no
#    buckyball.ball op survives those two lowerings either.
# 4) Graph side there is no op to attach: subgraph0.mlir (source of record:
#    model-side import, models/models/BertMini/output/subgraph0.mlir) carries
#    gelu as an exact-erf sequence terminating in 5 math.erf ops (4x
#    tensor<1x11x1024xf32> FFN sites at :261/:455/:649/:843 and 1x
#    tensor<1x11x256xf32> prediction-head site at :911) and layernorm as a
#    tosa.reduce_sum/reciprocal/rsqrt decomposition (~30 lines per site, 24
#    tosa.reduce_sum / 20 tosa.reciprocal / 10 tosa.rsqrt total, eps const
#    dense<9.99999996E-13> i.e. 1e-12 matching the LayerNormBall contract);
#    buddy-opt exposes no recognition pattern that folds either back into a
#    single buckyball op. Dry-run of this layout's scalar pipeline on
#    subgraph0.mlir (toy buddy-opt build/toy/bin): 0 buckyball.* ops survive,
#    the 5 erf become llvm.call @erff via -convert-math-to-libm (verified
#    round trip: pipeline1 0 buckyball + 5 math.erf; pipeline2 0 buckyball +
#    5 calls @erff).
#
# Both balls stay reachable only through direct ball ctests (S3); restoring
# a model-graph path is recorded as derived specs D-ln-1/D-ln-2 (this bind's
# PR description; D-gelu-* remain on file from PR #15). subgraph0.o therefore
# is a full scalar host fallback, same routing as toy/BertTiny.
#
# The run target name is the one bb-tests/workloads/scripts/build.py
# registers in _MODELS for the "bertmini" CLI model key.

set(CMAKE_CXX_COMPILER ${RISCV_GNU_TOOLCHAIN}/bin/riscv64-unknown-linux-gnu-g++)

set(MODEL_BERTMINI_DIR ${MODEL_DIR}/BertMini)
set(MODEL_BERTMINI_GENERATED_DIR ${MODEL_BERTMINI_DIR}/output)
set(BUCKYBALL_BERTMINI_DIR ${BUCKYBALL_DIR}/BertMini)
set(INTERFACE_DIR ${BUDDY_MLIR_DIR}/frontend/Interfaces)
set(INCLUDE_DIR ${MODELTEST_LIB_DIR})

# forward.o: forward.mlir is a thin wrapper that unpacks the packed arg0 into
# private subviews and calls the private @subgraph0 symbol (defined by
# subgraph0.o). It contains no compute itself; the scalar passes below are
# no-ops on the memref slice/cast ops and just hand the module to LLVM.
add_custom_command(
  OUTPUT forward.o
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTMINI_GENERATED_DIR}/forward.mlir
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
        DEPENDS ${MODEL_BERTMINI_GENERATED_DIR}/forward.mlir
  COMMENT "Building forward.o"
  VERBATIM)

# subgraph0.o: full scalar host fallback on the toy single core (same routing
# as the toy/BertTiny layout). Neither compiler path to toy's mounted balls
# reaches GeluBall / LayerNormBall for this graph (see header): toy never runs
# -convert-tile-to-buckyball, its tile pass registers no gelu/layernorm hook
# (BUCKYBALL_TILE_HOOK empty for toy), and its bank-SSA / assign populate
# hooks are empty stubs, so the buckyball ball pipeline cannot lower this
# module either. Every op - the 5 math.erf exact-gelu terminations, the
# layernorm reduce_sum/rsqrt sites and the matmuls - lowers to plain scalar
# loops on the host RISC-V core, with gelu falling to glibc erff via
# -convert-math-to-libm. The accelerator follow-up (tile hook + toy
# bank-SSA/assign) is recorded as derived specs D-ln-1/D-ln-2 in the binding
# PR description.
add_custom_command(
  OUTPUT subgraph0.o
  BYPRODUCTS subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTMINI_GENERATED_DIR}/subgraph0.mlir
            -pass-pipeline "builtin.module(func.func(tosa-to-linalg-named, tosa-to-linalg, tosa-to-tensor, tosa-to-arith))"
          > subgraph0_linalg.mlir
  COMMAND ${BUDDY_BINARY_DIR}/buddy-opt ${MODEL_BERTMINI_GENERATED_DIR}/subgraph0.mlir
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
        DEPENDS ${MODEL_BERTMINI_GENERATED_DIR}/subgraph0.mlir
  COMMENT "Building subgraph0.o"
  VERBATIM)

add_library(BuckyballBertMini STATIC forward.o subgraph0.o)
set_target_properties(BuckyballBertMini PROPERTIES LINKER_LANGUAGE CXX)
add_dependencies(BuckyballBertMini bertmini-model-build)

add_executable(buddy-buckyball-bertmini-run ${MODEL_BERTMINI_DIR}/bertmini-main.cpp)
add_dependencies(buddy-buckyball-bertmini-run BuckyballBertMini)

target_include_directories(buddy-buckyball-bertmini-run PRIVATE ${INTERFACE_DIR} ${INCLUDE_DIR})
# BertMini uses no DIP (resize) ops: ModelTestDIP_riscv is deliberately not
# linked, so the DIP.mlir toolchain dependency stays out of this target's
# ninja closure.
set(BUDDY_BERTMINI_LIBS BuckyballBertMini ModelTestCRunnerUtils)
target_link_libraries(buddy-buckyball-bertmini-run -static ${BUDDY_BERTMINI_LIBS})

set(BERTMINI_OUTPUT_DIR ${BUCKYBALL_OUTPUT_DIR}/BertMini)

add_custom_command(TARGET buddy-buckyball-bertmini-run POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTMINI_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:buddy-buckyball-bertmini-run>
          ${BERTMINI_OUTPUT_DIR}/buddy-buckyball-bertmini-run
  COMMENT "Copying Buckyball BertMini executable to output directory"
  VERBATIM
)

# Runtime deps: import artifacts (arg0/arg1.data) come from the model-side
# build; vocab.txt is committed in the model directory. bemu --sim runs with
# the binary's directory as cwd, so the artifacts land next to the ELF.
add_custom_target(sync-buckyball-bertmini-deps
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTMINI_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BERTMINI_OUTPUT_DIR}/trace/cycle
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTMINI_GENERATED_DIR}/arg0.data
          ${BERTMINI_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTMINI_GENERATED_DIR}/arg1.data
          ${BERTMINI_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MODEL_BERTMINI_DIR}/vocab.txt
          ${BERTMINI_OUTPUT_DIR}/vocab.txt
  DEPENDS
          ${MODEL_BERTMINI_GENERATED_DIR}/arg0.data
          ${MODEL_BERTMINI_GENERATED_DIR}/arg1.data
          ${MODEL_BERTMINI_DIR}/vocab.txt
  COMMENT "Syncing BertMini runtime deps to output directory"
  VERBATIM
)

add_dependencies(buddy-buckyball-bertmini-run sync-buckyball-bertmini-deps)

# Perfetto evidence chain (design doc §4.7) is intentionally NOT wired here
# yet: -extend-trace-to-linalg only extends trace.start scopes, which the
# model-side import (import-bertmini.py) must insert from a trace TOML.
# Until the model side re-imports with trace scopes, no cycle files are
# produced and a perfetto target would fail with "no cycle trace files found".
# The bbdev PERFETTO_TARGETS mapping for buddy-buckyball-bertmini-run is
# therefore also absent.
