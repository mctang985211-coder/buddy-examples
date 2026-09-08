# SmolLM (HuggingFaceTB/SmolLM-135M) model-side workload.
#
# Only the buddy frontend import is bound here: `smollm-model-build` runs
# import-smollm.py and produces the AOT artifacts that the later (chip-bound)
# stages consume. No arch, core-count, quant or trace binding.
set(MODEL_SMOLLM_OUTPUT_DIR ${MODEL_SMOLLM_DIR})

add_custom_command(
  OUTPUT ${MODEL_SMOLLM_DIR}/forward.mlir
         ${MODEL_SMOLLM_DIR}/subgraph0.mlir
         ${MODEL_SMOLLM_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_SMOLLM_DIR}/import-smollm.py
          --output-dir ${MODEL_SMOLLM_DIR}
  COMMENT "Generating SmolLM forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(smollm-model-build DEPENDS
  ${MODEL_SMOLLM_DIR}/forward.mlir
  ${MODEL_SMOLLM_DIR}/subgraph0.mlir
  ${MODEL_SMOLLM_DIR}/arg0.data
)
