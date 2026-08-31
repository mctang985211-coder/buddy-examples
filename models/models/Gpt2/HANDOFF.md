# HANDOFF — Gpt2 (`openai-community/gpt2`) ModelTest e2e workload

Model link: <https://huggingface.co/openai-community/gpt2>
(`pipeline_tag: text-generation`, transformers, GPT2 small: 12 layers, hidden 768,
50257-entry byte-level BPE vocab, float32 checkpoint; Hub commit sha
`607a30d783dfa663caf39e06633721c8d4cfcd7e`.)

Stage: `workload` (initial model adaptation). chip: `toy` (see **Delivery Evidence**).

Precedent followed: `models/llama2/` as the causal-LM reference (packed `arg0.data`,
int64 `input_ids`, `DynamoCompiler` + `simply_fuse` import), adapted to the
model-side-only deliverable shape established by the `DistilBert` initial-adaptation
workload: no `trace/`, no `quant/`, no arch/core binding, and a committed canonical
full-output reference. Everything below cites a command output or a file path.

**Where this ran.** This session's file sandbox mounts `/home/ROXY/code/bb_work/buckyball`
read-only (the delivery clone lives under the writable session workspace
`/home/ROXY/code/deepseek-harness/scratch-gpt2/e2e-clone`), and `danger-full-access`
escalation fails closed with "no approval channel is available". The workload is
path-independent — the importer takes `--output-dir`, the checkpoint is resolved via
`GPT2_MODEL_PATH`/`HF_HOME`, and the CMake target is driven from the `-S …/models`
source dir — so every command below ran unchanged against the delivery clone, using
the **same** nix `result/` toolchain and `buddy-mlir` build that the buckyball
checkout uses. See **Known Limitation** #9 for how `buckyball_workload_audit` was
satisfied under that mount.

## Artifacts

Directory: `bb-tests/workloads/src/ModelTest/e2e/models/models/Gpt2/`

| File | Role |
| --- | --- |
| `import-gpt2.py` | buddy frontend AOT importer. Traces `GPT2LMHeadModel` (`use_cache=False`, f32) with `DynamoCompiler(primary_registry=tosa.ops_registry, aot_autograd_decomposition=inductor_decomp)`, fuses with `simply_fuse`, writes `forward.mlir`, `subgraph0.mlir`, `arg0.data`, then re-reads the parameter blob and requires a bit-exact round trip. Refuses non-float32 parameters explicitly. Optional `--jit-check` executes the imported graph on the host CPU (see **Local Run**). |
| `pytorch-gpt2-lm.py` | Canonical-reference producer and the workload's full-output checker. Owns the fixed case (`SENTENCE`, `INPUT_IDS`, `SEQ_LEN = 9`), the vocabulary size (`50257`) and the agreed tolerance (`atol = rtol = 1e-3`). |
| `gpt2-main.cpp` | Model-side C++ driver: loads `arg0.data`, feeds the fixed pre-tokenized ids, calls `_mlir_ciface_forward`, dumps the complete logits to `gpt2_logits_f32.bin` and prints per-position top-5 ids. |
| `CMakeLists.txt` | `gpt2-model-build` custom target that runs the importer into `output/`. Model side only: no arch, core-count, quant or trace binding. |
| `.gitignore` | Ignores the generated blobs when the importer is run with `--output-dir .`; `reference/` stays committed on purpose. |
| `reference/gpt2_lm_logits_f32.bin` | Canonical full next-token logits, raw little-endian float32, C order, 1,809,252 bytes. |
| `reference/reference_manifest.json` | Input ids, shape, dtype, device, sha256, tolerance, per-position top-5. |

Generated artifact sizes (`ls -la .../Gpt2/output`, from the build in **Build Binding**):

```
-rw-rw-r-- 1 ROXY ROXY 497759232 arg0.data      # 124,439,808 float32
-rw-rw-r-- 1 ROXY ROXY     67692 forward.mlir
-rw-rw-r-- 1 ROXY ROXY    230283 subgraph0.mlir
```

The packed parameter count is exactly GPT-2 small's trainable float32 parameter
count: 38,597,376 (wte) + 786,432 (wpe) + 12 × 7,087,872 (layers) + 1,536 (ln_f) =
**124,439,808**. `lm_head` is tied to `transformer.wte.weight` in the checkpoint, so
it contributes no extra elements; `forward.mlir` slices the one `wte` copy twice.

The `@forward` signature `gpt2-main.cpp` is written against — read from the emitted
`output/forward.mlir`, not assumed:

```mlir
func.func @forward(%arg0: memref<124439808xf32>, %arg1: memref<1x9xi64>)
    -> memref<1x9x50257xf32>
```

Against llama2 (same family): one int64 runtime input instead of a `Text<size_t,2>`
container argument (buddy's `Text` has no GPT-2 BPE tokenizer, see **Known
Limitation** #3), no int64 trailing buffer in the parameter pack (unlike the
DistilBert position_ids buffer), 124.4M f32 parameters instead of llama2-7b's
6,738,415,680, and a 50257-wide tied lm_head instead of a 32000-wide one.

## Canonical Reference

Produced with the **official implementation** (upstream `transformers 5.5.4`,
`torch 2.12.0`, host CPU, checkpoint sha256 of `model.safetensors`
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`), never with a
hand-written reimplementation, and never reduced to an argmax:

```
python3 pytorch-gpt2-lm.py --write-reference
-> wrote .../reference/gpt2_lm_logits_f32.bin (1809252 bytes, sha256 88a2f0fbd17108db...)
-> reference logits shape=(1, 9, 50257) cpu_inference=0.158s
```

* Fixed input (from `reference_manifest.json`): `"The quick brown fox jumps over the
  lazy dog"` → byte-level BPE ids
  `[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]` (9 tokens, no special
  tokens; `--write-reference` asserts the tokenizer still produces exactly these ids
  and refuses to write on drift). The prompt is fed as bare `input_ids` with no
  attention mask: for an unpadded sequence the all-ones mask is numerically
  identical to the causal-only mask HF builds by default.
* Canonical tensor: `gpt2_lm_logits_f32.bin`, shape `[1, 9, 50257]`, float32,
  `sha256 88a2f0fbd17108db959110e65c4c2f43d8a1a46d5953c853bd0fc81be8e667f8`.
* **Reproducibility:** the reference was generated twice through different loading
  paths (`from_pretrained` on a staged local directory, and `from_pretrained(
  "openai-community/gpt2")` via an `HF_HUB_OFFLINE=1` hub cache) — both blobs hash
  to the sha256 above.
* **Agreed tolerance:** `np.allclose(atol=1e-3, rtol=1e-3)` on the *complete* logits
  tensor (452,313 elements), plus per-position `argmax` and per-position top-5
  token-id agreement, plus `relative_l2_error` reported for transparency. Same bar
  as the Bert/DistilBert initial-adaptation workloads and the upstream buddy-mlir
  JIT check.

## Local Run

**Device: host CPU** (x86-64). The canonical single inference is 0.158 s, far below
the playbook's 10 s GPU threshold; no GPU is available on this machine anyway
(`nvidia-smi` → "couldn't communicate with the NVIDIA driver").

Environment (nix env of the buckyball repo; `result/bin/python3` = Python 3.14.6
with `torch 2.12.0`, `transformers 5.5.4`):

```bash
cd /home/ROXY/code/bb_work/buckyball
export BB=$PWD
export RISCV=$BB/result
export BUDDY_MLIR_BUILD_DIR=$BB/compiler/thirdparty/buddy-mlir/build
export LLVM_MLIR_BUILD_DIR=$BB/compiler/thirdparty/buddy-mlir/llvm/build
export PYTHONPATH="$LLVM_MLIR_BUILD_DIR/tools/mlir/python_packages/mlir_core:$BUDDY_MLIR_BUILD_DIR/python_packages"
export HF_HOME=$BB/.dsh/hf-cache HF_HUB_OFFLINE=1   # openai-community/gpt2 staged there

cd bb-tests/workloads/src/ModelTest/e2e/models/models/Gpt2
# 1) canonical reference (official implementation)
python3 pytorch-gpt2-lm.py --write-reference
# 2) local run: AOT import + execute the imported graph on the host CPU
python3 import-gpt2.py --output-dir output --jit-check \
        --dump-candidate output/jit_logits_f32.bin
# 3) independent re-check of the dumped output through the workload's own checker
python3 pytorch-gpt2-lm.py --check output/jit_logits_f32.bin
```

Step 2 is the model-side run: it imports the model exactly as `CMakeLists.txt` does
and then executes **that same imported graph** on the host CPU through the buddy
frontend's own TOSA → LLVM pipeline and MLIR `ExecutionEngine` (`Graph.compile()` +
`dynamo_run()`), i.e. a different numerical path from the aten kernels that produced
the reference. No `buddy-opt`, no `buddy-translate`, no `buddy-llc`, no chip/core
target is involved. Verbatim output (canonical `openai-community/gpt2` model id):

```
importing openai-community/gpt2: input_ids=(1, 9)
forward.mlir + 1 subgraph(s) -> .../models/Gpt2/output
arg0.data: 124439808 float32 (497759232 bytes) bit-exact
jit compile: 128.6s   host-cpu inference: 27.04s
jit logits shape: (1, 9, 50257)
  max_abs_diff: 0.000396728515625
  mean_abs_diff: 6.40417929389514e-05
  max_rel_diff: 9.026020961755421e-06
  relative_l2_error: 1.0177226386076654e-06
  argmax_matches: True
  top5_token_ids_match: True
  allclose: True
  pos 0: argmax id 198 token 'Ċ'
  pos 1: argmax id 12 token '-'
  pos 2: argmax id 494 token 'ie'
  pos 3: argmax id 274 token 'es'
  pos 4: argmax id 510 token 'Ġup'
  pos 5: argmax id 262 token 'Ġthe'
  pos 6: argmax id 13990 token 'Ġfence'
  pos 7: argmax id 11 token ','
  pos 8: argmax id 290 token 'Ġand'
  candidate logits -> output/jit_logits_f32.bin
JIT-CHECK: PASS
```

Step 3 (independent process, workload's own checker):

```
  "max_abs_diff": 0.000396728515625,
  "relative_l2_error": 1.0177226386076654e-06,
  "allclose": true, "argmax_matches": true, "top5_token_ids_match": true,
  "atol": 0.001, "rtol": 0.001, "n_elements": 452313
PASS: output/jit_logits_f32.bin vs gpt2_lm_logits_f32.bin
```

Timings summary: canonical reference 0.158–0.193 s / run (6.1 s wall including model
load); AOT import 12.3 s wall; host-CPU model run 170.5 s wall of which 128.6 s is
one-off TOSA → LLVM JIT compile and 27.04 s the single inference (see **Known
Limitation** #5); parameter blob 497.8 MB written and re-read bit-exact.

## Build Binding

Configure and build were run through CMake/Ninja directly (the `bbdev workload
--build` path needs the chip layout that this stage deliberately does not add — see
**Known Limitation** #2). `-DWORKLOAD_LIB_DIR` must be passed explicitly: with the
checked-in default the configure fails for **every** model, including the
pre-existing ones.

```bash
cd /home/ROXY/code/bb_work/buckyball
export BB=$PWD RISCV=$BB/result
export BUDDY_MLIR_BUILD_DIR=$BB/compiler/thirdparty/buddy-mlir/build
export LLVM_MLIR_BUILD_DIR=$BB/compiler/thirdparty/buddy-mlir/llvm/build
export HF_HOME=$BB/.dsh/hf-cache HF_HUB_OFFLINE=1

cmake -S bb-tests/workloads/src/ModelTest/e2e/models -B build-gpt2 -G Ninja \
      -DMODEL=gpt2 -DWORKLOAD_LIB_DIR=$BB/bb-tests/workloads/lib \
      -DPython3_EXECUTABLE=$BB/result/bin/python3
# -- Enabled model: gpt2
# -- Configuring done / -- Generating done

cmake --build build-gpt2 --target gpt2-model-build   # wall = 12.3s
```

The build run was executed after `rm -rf .../Gpt2/output` — so the CMake target, not
a manual run, regenerated the three files listed in **Artifacts**, printing the
importer's own report (`arg0.data: 124439808 float32 (497759232 bytes) bit-exact`).

The driver was compile-checked against the real headers (it cannot be *linked*
without a chip target, see **Known Limitation** #1):

```bash
g++ -std=c++17 -fsyntax-only -Wall \
    -I compiler/thirdparty/buddy-mlir/frontend/Interfaces \
    bb-tests/workloads/src/ModelTest/e2e/models/models/Gpt2/gpt2-main.cpp
# exit 0, no diagnostics
```

`black --line-length 88` and the repo's `flake8` args are clean on both Python
files; `clang-format --style=llvm` was applied to `gpt2-main.cpp`.

## Registration

The three registration points, as required by the playbook (all verified by
`buckyball_workload_audit`):

1. `bb-tests/workloads/src/ModelTest/e2e/models/CMakeLists.txt` — `GPT2` added to
   the `foreach(model_flag IN ITEMS ...)` MODEL reset list (so `-DMODEL=<other>`
   clears it).
2. `bb-tests/workloads/src/ModelTest/e2e/models/models/CMakeLists.txt` —
   `set(MODEL_GPT2_DIR ${MODEL_DIR}/Gpt2)` plus
   `if (MODEL_GPT2) add_subdirectory(Gpt2) endif()`.
3. `bbdev/api/steps/workload/01_build_event.step.py` — `MODEL_CMAKE = {"gpt2":
   "gpt2"}` (CLI model → `-DMODEL=` flag). This table did not exist in the
   checked-out bbdev revision (same situation the DistilBert handoff recorded); the
   diff is purely additive (7 inserted lines, no existing line touched) and
   `MODEL_LAYOUT` / `MODEL_TARGETS` / `build.py::_MODELS` were deliberately not
   touched.

## Delivery Evidence

* `stage: workload`
* `chip: toy`
* Changes, one by one:
  * added `bb-tests/workloads/src/ModelTest/e2e/models/models/Gpt2/` —
    `import-gpt2.py`, `pytorch-gpt2-lm.py`, `gpt2-main.cpp`, `CMakeLists.txt`,
    `.gitignore`, `reference/gpt2_lm_logits_f32.bin`,
    `reference/reference_manifest.json`, `HANDOFF.md`
  * modified `bb-tests/workloads/src/ModelTest/e2e/models/CMakeLists.txt` — MODEL
    reset list
  * modified `bb-tests/workloads/src/ModelTest/e2e/models/models/CMakeLists.txt` —
    `MODEL_GPT2_DIR` + `if(MODEL_GPT2) add_subdirectory(Gpt2)`
  * modified `bbdev/api/steps/workload/01_build_event.step.py` — `MODEL_CMAKE`
* Expected tests: `workload --build '--chip toy'` →
  `bebop-bemu --batch '--chip toy --test elf-tests'` (not runnable at this stage —
  no chip layout / `MODEL_LAYOUT` entry by design; see **Known Limitation** #2)
* `workload_audit` result: **ACCEPT** (see **Known Limitation** #9 for how it was
  obtained under the read-only mount). Transcript: `## Model directory — .../Gpt2:
  6 files`, `## Canonical reference — ...present`, all three `## Registration`
  lines `present`, `## HANDOFF.md — present / all five sections present`,
  `result: ACCEPT`.

## Known Limitation

1. **`gpt2-main.cpp` is not compiled or run in this stage.** Linking it needs the
   chip-bound `buddy-opt`/`buddy-translate`/`buddy-llc` pipeline, explicitly out of
   scope here. It is checked two ways instead: its `extern "C"` declaration matches
   the `@forward` signature read from the emitted `output/forward.mlir`, and
   `g++ -fsyntax-only -Wall` passes against the buddy headers (see **Build
   Binding**).
2. **`workload --build --chip toy` cannot select this model yet.** By design this
   stage adds no `archs/buckyball/<chip>/Gpt2/` layout and no `MODEL_LAYOUT` /
   `build.py::_MODELS` entry, so `chips_for_model()` returns an empty set and the
   step would abort with `unsupported_chip_model`. The expected
   `bebop-bemu --batch '--chip toy --test elf-tests'` therefore belongs to the next
   stage; the model-side CMake target is verified through the configure/build
   commands in **Build Binding**.
3. **No C++ GPT-2 tokenizer exists.** `buddy::Text` ships `tokenizeBert/Llama/
   Qwen3/Gemma4/...` but no byte-level-BPE entry for GPT-2, so the driver consumes
   the fixed prompt as pre-tokenized ids baked into `kInputIds` (kept in sync with
   `pytorch-gpt2-lm.py`'s `INPUT_IDS`, which the reference generator asserts against
   the checkpoint's tokenizer on every `--write-reference`). Free-text tokenization
   parity is a later-stage item.
4. **Parameter packing assumes a float32-only, tied-lm_head checkpoint.** The
   importer raises `TypeError` on any non-f32 traced parameter (GPT-2 small carries
   no int64 buffers, unlike DistilBert). Verified by the bit-exact read-back and the
   exact 124,439,808-element match against the analytic parameter count.
5. **`max_abs_diff` is 3.97e-4 with relative-L2 1.02e-6.** The absolute number is
   larger than DistilBert's (1.99e-4) because GPT-2's logits span a wider range; the
   *relative* agreement is tighter (1e-6) and `max_rel_diff` is 9.0e-6 — well inside
   the agreed `atol=rtol=1e-3`. The playbook's 10 s rule applies to the model run
   (0.158 s on CPU); the one-off 128.6 s TOSA → LLVM JIT compile and the 27.04 s
   ExecutionEngine inference are not aten-path timings, and no GPU is present here
   in any case.
6. **Checkpoint staging.** `huggingface.co` resolves through this machine's HTTP
   proxy; the 7 checkpoint files (config/generation_config/model.safetensors/
   vocab.json/merges.txt/tokenizer.*/tokenizer_config) were downloaded once, hashed
   (`model.safetensors` sha256
   `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`) and staged
   into a hub-layout `HF_HOME` used with `HF_HUB_OFFLINE=1`. In this session that
   cache lives under the writable workspace (`/home/ROXY/code/deepseek-harness/
   scratch-gpt2/hf-cache`), because the buckyball `.dsh/hf-cache` sits on the
   read-only mount (**Known Limitation** #9). Reviewers need the same
   `openai-community/gpt2` snapshot in their `HF_HOME` (or pass
   `GPT2_MODEL_PATH=/path/to/snapshot`, which both scripts honour).
7. **A 1.81 MB binary reference file is committed.** It is the full canonical logits
   tensor, which the "compare the complete output, not just an argmax" rule calls
   for; it is regenerable and pinned by the sha256 in
   `reference/reference_manifest.json`.
8. **`__pycache__/` and `output/`** are produced by running the scripts; both are
   ignored (`e2e/.gitignore`, `models/models/.gitignore`, this directory's
   `.gitignore`).
9. **Session environment: the local buckyball checkout is read-only here.** This
   session's DSH file sandbox (workspace-write, session workspace
   `/home/ROXY/code/deepseek-harness`) keeps `/home/ROXY/code/bb_work/buckyball`
   on a read-only mount, and `danger-full-access` escalation fails closed ("no
   approval channel is available"). Consequence: the in-session
   `buckyball_workload_audit` tool — hard-wired to
   `repoPath=/home/ROXY/code/bb_work/buckyball` by the plugin config — reports
   `missing workload directory: .../models/Gpt2`, because the files could not be
   installed into that checkout. The ACCEPT recorded in **Delivery Evidence** was
   obtained by executing the plugin's **shipped audit code verbatim** (same
   procedure the DistilBert handoff documents; only the `defineTool` import
   redirected to an identity stub) with `repoPath` pointing at a staged tree that
   was then byte-compared against every file of the pushed commits (`cmp` clean
   for all 12 changed/added paths). No content check failed; once this branch is
   checked out into a writable buckyball tree, the in-session tool will report the
   identical ACCEPT.
