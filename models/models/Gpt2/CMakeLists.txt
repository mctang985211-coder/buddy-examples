# Gpt2 (openai-community/gpt2) model-side workload.
#
# Only the buddy frontend import is bound here: `gpt2-model-build` runs
# import-gpt2.py and produces the AOT artifacts that the later
# (chip-bound) stages consume. No arch, core-count, quant or trace binding.
set(MODEL_GPT2_OUTPUT_DIR ${MODEL_GPT2_DIR}/output)

add_custom_command(
  OUTPUT ${MODEL_GPT2_OUTPUT_DIR}/forward.mlir
         ${MODEL_GPT2_OUTPUT_DIR}/subgraph0.mlir
         ${MODEL_GPT2_OUTPUT_DIR}/arg0.data
  COMMAND ${CMAKE_COMMAND} -E env
          BUDDY_MLIR_BUILD_DIR=$ENV{BUDDY_MLIR_BUILD_DIR}
          PYTHONPATH=${BUDDY_BINARY_DIR}/../python_packages
          ${Python3_EXECUTABLE} ${MODEL_GPT2_DIR}/import-gpt2.py
          --output-dir ${MODEL_GPT2_OUTPUT_DIR}
  COMMENT "Generating Gpt2 forward.mlir, subgraph0.mlir and parameters"
)

add_custom_target(gpt2-model-build DEPENDS
  ${MODEL_GPT2_OUTPUT_DIR}/forward.mlir
  ${MODEL_GPT2_OUTPUT_DIR}/subgraph0.mlir
  ${MODEL_GPT2_OUTPUT_DIR}/arg0.data
)
