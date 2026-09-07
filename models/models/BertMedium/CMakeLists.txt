# BertMedium (prajjwal1/bert-medium) model-side workload.
#
# Only the buddy frontend import is bound here: `bertmedium-model-build` runs
# import-bertmedium.py and produces the AOT artifacts that the later
# (chip-bound) stages consume. No arch, core-count, quant or trace binding.
set(MODEL_BERTMEDIUM_OUTPUT_DIR ${MODEL_BERTMEDIUM_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_BERTMEDIUM_OUTPUT_DIR}/forward.mlir
         ${MODEL_BERTMEDIUM_OUTPUT_DIR}/subgraph0.mlir
         ${MODEL_BERTMEDIUM_OUTPUT_DIR}/arg0.data
         ${MODEL_BERTMEDIUM_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_BERTMEDIUM_DIR}/import-bertmedium.py
          --output-dir ${MODEL_BERTMEDIUM_OUTPUT_DIR}
  COMMENT "Generating BertMedium forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(bertmedium-model-build DEPENDS
  ${MODEL_BERTMEDIUM_OUTPUT_DIR}/forward.mlir
  ${MODEL_BERTMEDIUM_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_BERTMEDIUM_OUTPUT_DIR}/arg0.data
  ${MODEL_BERTMEDIUM_OUTPUT_DIR}/arg1.data
)
