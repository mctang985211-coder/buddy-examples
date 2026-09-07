# Pythia (EleutherAI/pythia-70m) model-side workload.
#
# Only the buddy frontend import is bound here: `pythia-model-build` runs
# import-pythia.py and produces the AOT artifacts that the later (chip-bound)
# stages consume. No arch, core-count, quant or trace binding.
set(MODEL_PYTHIA_OUTPUT_DIR ${MODEL_PYTHIA_DIR})

add_custom_command(
  OUTPUT ${MODEL_PYTHIA_DIR}/forward.mlir
         ${MODEL_PYTHIA_DIR}/subgraph0.mlir
         ${MODEL_PYTHIA_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_PYTHIA_DIR}/import-pythia.py
          --output-dir ${MODEL_PYTHIA_DIR}
  COMMENT "Generating Pythia forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(pythia-model-build DEPENDS
  ${MODEL_PYTHIA_DIR}/forward.mlir
  ${MODEL_PYTHIA_DIR}/subgraph0.mlir
  ${MODEL_PYTHIA_DIR}/arg0.data
)
