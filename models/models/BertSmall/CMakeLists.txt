# BertSmall (prajjwal1/bert-small) model-side workload.
#
# Only the buddy frontend import is bound here: `bertsmall-model-build` runs
# import-bertsmall.py and produces the AOT artifacts that the later
# (chip-bound) stages consume. No arch, core-count, quant or trace binding.
set(MODEL_BERTSMALL_OUTPUT_DIR ${MODEL_BERTSMALL_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_BERTSMALL_OUTPUT_DIR}/forward.mlir
         ${MODEL_BERTSMALL_OUTPUT_DIR}/subgraph0.mlir
         ${MODEL_BERTSMALL_OUTPUT_DIR}/arg0.data
         ${MODEL_BERTSMALL_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_BERTSMALL_DIR}/import-bertsmall.py
          --output-dir ${MODEL_BERTSMALL_OUTPUT_DIR}
  COMMENT "Generating BertSmall forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(bertsmall-model-build DEPENDS
  ${MODEL_BERTSMALL_OUTPUT_DIR}/forward.mlir
  ${MODEL_BERTSMALL_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_BERTSMALL_OUTPUT_DIR}/arg0.data
  ${MODEL_BERTSMALL_OUTPUT_DIR}/arg1.data
)
