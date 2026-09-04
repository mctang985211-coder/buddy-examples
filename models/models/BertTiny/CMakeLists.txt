# BertTiny (prajjwal1/bert-tiny) model-side workload.
#
# Only the buddy frontend import is bound here: `berttiny-model-build` runs
# import-berttiny.py and produces the AOT artifacts that the later
# (chip-bound) stages consume. No arch, core-count, quant or trace binding.
set(MODEL_BERTTINY_OUTPUT_DIR ${MODEL_BERTTINY_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_BERTTINY_OUTPUT_DIR}/forward.mlir
         ${MODEL_BERTTINY_OUTPUT_DIR}/subgraph0.mlir
         ${MODEL_BERTTINY_OUTPUT_DIR}/arg0.data
         ${MODEL_BERTTINY_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_BERTTINY_DIR}/import-berttiny.py
          --output-dir ${MODEL_BERTTINY_OUTPUT_DIR}
  COMMENT "Generating BertTiny forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(berttiny-model-build DEPENDS
  ${MODEL_BERTTINY_OUTPUT_DIR}/forward.mlir
  ${MODEL_BERTTINY_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_BERTTINY_OUTPUT_DIR}/arg0.data
  ${MODEL_BERTTINY_OUTPUT_DIR}/arg1.data
)
