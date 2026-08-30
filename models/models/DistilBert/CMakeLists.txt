# DistilBert (distilbert-base-uncased) model-side workload.
#
# Only the buddy frontend import is bound here: `distilbert-model-build` runs
# import-distilbert.py and produces the AOT artifacts that the later
# (chip-bound) stages consume. No arch, core-count, quant or trace binding.
set(MODEL_DISTILBERT_OUTPUT_DIR ${MODEL_DISTILBERT_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_DISTILBERT_OUTPUT_DIR}/forward.mlir
         ${MODEL_DISTILBERT_OUTPUT_DIR}/subgraph0.mlir
         ${MODEL_DISTILBERT_OUTPUT_DIR}/arg0.data
         ${MODEL_DISTILBERT_OUTPUT_DIR}/arg1.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_DISTILBERT_DIR}/import-distilbert.py
          --output-dir ${MODEL_DISTILBERT_OUTPUT_DIR}
  COMMENT "Generating DistilBert forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(distilbert-model-build DEPENDS
  ${MODEL_DISTILBERT_OUTPUT_DIR}/forward.mlir
  ${MODEL_DISTILBERT_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_DISTILBERT_OUTPUT_DIR}/arg0.data
  ${MODEL_DISTILBERT_OUTPUT_DIR}/arg1.data
)
