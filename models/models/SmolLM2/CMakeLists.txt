# SmolLM2 (HuggingFaceTB/SmolLM2-135M) model-side workload.
#
# Only the buddy frontend import is bound here: `smollm2-model-build` runs
# import-smollm2.py and produces the AOT artifacts that the later (chip-bound)
# stages consume. No arch, core-count, quant or trace binding.
set(MODEL_SMOLLM2_OUTPUT_DIR ${MODEL_SMOLLM2_DIR})

add_custom_command(
  OUTPUT ${MODEL_SMOLLM2_DIR}/forward.mlir
         ${MODEL_SMOLLM2_DIR}/subgraph0.mlir
         ${MODEL_SMOLLM2_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_SMOLLM2_DIR}/import-smollm2.py
          --output-dir ${MODEL_SMOLLM2_DIR}
  COMMENT "Generating SmolLM2 forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(smollm2-model-build DEPENDS
  ${MODEL_SMOLLM2_DIR}/forward.mlir
  ${MODEL_SMOLLM2_DIR}/subgraph0.mlir
  ${MODEL_SMOLLM2_DIR}/arg0.data
)
