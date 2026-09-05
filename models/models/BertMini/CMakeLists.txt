# BertMini (prajjwal1/bert-mini) model-side workload.
#
# Only the buddy frontend import is bound here: `bertmini-model-build` runs
# import-bertmini.py and produces the AOT artifacts that the later
# (chip-bound) stages consume. No arch, core-count, quant or trace binding.
set(MODEL_BERTMINI_OUTPUT_DIR ${MODEL_BERTMINI_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_BERTMINI_OUTPUT_DIR}/forward.mlir
         ${MODEL_BERTMINI_OUTPUT_DIR}/subgraph0.mlir
         ${MODEL_BERTMINI_OUTPUT_DIR}/arg0.data
         ${MODEL_BERTMINI_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_BERTMINI_DIR}/import-bertmini.py
          --output-dir ${MODEL_BERTMINI_OUTPUT_DIR}
  COMMENT "Generating BertMini forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(bertmini-model-build DEPENDS
  ${MODEL_BERTMINI_OUTPUT_DIR}/forward.mlir
  ${MODEL_BERTMINI_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_BERTMINI_OUTPUT_DIR}/arg0.data
  ${MODEL_BERTMINI_OUTPUT_DIR}/arg1.data
)
