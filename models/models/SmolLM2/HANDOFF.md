# HANDOFF — SmolLM2 (`HuggingFaceTB/SmolLM2-135M`) ModelTest e2e workload

Model link: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M>
(`pipeline_tag: text-generation`, `library: transformers`, single
`model.safetensors` 269,060,552 B, hub commit
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` — reported by the HF Hub API via
`buckyball_model_info` and confirmed by the local cache snapshot directory).

Config (read from the checkpoint's own `config.json`): LlamaForCausalLM,
`model_type: llama`, hidden size 576, **30 hidden layers**, 9 attention heads,
**3 KV heads (GQA)**, SwiGLU MLP intermediate 1536, `hidden_act: "silu"`,
RMSNorm eps 1e-5, **RoPE theta 100000** (the in-repo SmolLM-135M has 10000),
**max positions 8192** (SmolLM-135M has 2048), vocab 49152,
`tie_word_embeddings: true`, `pretraining_tp: 1`, `transformers_version:
4.40.1`. Config declares `torch_dtype: bfloat16` — and unlike the SmolLM-135M
precedent (whose hub snapshot actually stores float32 despite the same
declaration), this snapshot's `model.safetensors` **does store bfloat16**:
all 272 tensors are BF16, measured with `safetensors.safe_open` on the cached
snapshot (`model.embed_tokens.weight` → `torch.bfloat16`, `[49152, 576]`;
dtype histogram BF16 × 272; payload 269,030,016 B = 134,515,008 × 2). The
fp32 load of this workload therefore converts bf16 → float32 exactly (the
arg0 blob is float32, as always). Trainable parameter count read from the
loaded checkpoint: **134,515,008 float32** (tied embeddings counted once; the
file has no separate `lm_head.weight`) plus the 32-float32 rotary `inv_freq`
buffer, for a 134,515,040-element arg0.data blob.

Model key: **SmolLM2** (this directory / `SMOLLM2` CMake flag /
`MODEL_SMOLLM2_DIR`). It is a distinct key from the in-repo `SmolLM`
(`SMOLLM`, directory `SmolLM`, key `smollm`): same Llama-family architecture
and the same GPT-2-style BPE tokenizer (the committed `vocab.txt` is
byte-identical, sha256 `19f2e5a6b1c4d8bdcef27c5bcf0139e3d4c3f2ca2a52b5aceb88ca12c451a238`),
but different weights (this is the SmolLM2-135M checkpoint) and different
config (rope_theta / max positions). The bare-uppercase flag derivation
(`string(TOUPPER "smollm2")` → `SMOLLM2`) does not collide with `SMOLLM`, so
no variant name was needed; the future bind-stage `build.py::_MODELS` key
will be lowercase `smollm2` (not written at this stage).

Stage: `workload` (initial model adaptation). chip: `pebble` — **intended future
bind target, NOT yet bound** (this stage adds no chip layout; the e2e
`models/archs/buckyball/pebble/SmolLM2/` layout, the bbdev `MODEL_LAYOUT` entry
and the parent `bb-tests/workloads/scripts/build.py::_MODELS` entry do not exist
— see **Known Limitation** #2).

Precedents followed: the in-repo `models/SmolLM/` realization (gitlink
5588b46213c3e282f9273974c362959bd5af933b, phase9 delivery) is the master
template — same LlamaForCausalLM + DynamoCompiler frontend pathway, reference/
artifacts, fail-hard expectations, HANDOFF shape; the upstream `models/llama2/`
directory was read for the Llama-family import and vocab.txt conventions.
Everything below was executed on this checkout; every claim cites a command
output or a file path.

## Artifacts

Directory: `bb-tests/workloads/src/ModelTest/e2e/models/models/SmolLM2/`

| File | Role |
| --- | --- |
| `import-smollm2.py` | buddy frontend AOT importer. Traces `AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)` (env override `SMOLLM2_135M_MODEL_PATH`) with `DynamoCompiler(primary_registry=tosa.ops_registry, aot_autograd_decomposition=inductor_decomp)` over one full forward of the fixed 16-token input (no KV cache), fuses with `simply_fuse`, writes `forward.mlir`, `subgraph0.mlir`, `arg0.data`, asserts the single-f32-blob packing rule fail-hard (`check_param_layout`) and requires a bit-exact read-back. Optional `--jit-check` executes the imported graph on the host CPU (see **Local Run**; team-added switch, not an upstream form). |
| `smollm2-ppl.py` | Canonical-reference producer and the workload's full-output checker (team-added companion of the SmolLM `smollm-ppl.py --weights <arg0.data>` shape). Owns the fixed case (`TEXT`, `SEQ_LEN = 16`, `VOCAB_SIZE = 49152`), the agreed logits tolerance (`atol = rtol = 1e-3`) and the recon-weights ppl tolerance (`PPL_TOL = 0.05`). |
| `buddy-smollm2-main.cpp` | Model-side C++ driver: loads `arg0.data`, feeds the hardcoded fixed input ids, calls `_mlir_ciface_forward`, dumps the complete logits to `smollm2_driver_logits_f32.bin`, prints per-position argmax landing points, and **fails hard (returns 1)** when a landing point deviates from the hardcoded expectation `kExpectedArgmaxIds` (canonical top-1 token ids, see **Canonical Reference**). |
| `CMakeLists.txt` | `smollm2-model-build` custom target that runs the importer. Model side only: no arch, core-count, quant or trace binding. |
| `vocab.txt` | Byte-level BPE vocabulary of the official SmolLM2 (GPT-2-style) tokenizer, 49152 lines (418,322 B), one token per line ordered by id; ids 0–16 are the 17 special tokens (`<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<repo_name>`, `<reponame>`, … `<empty_output>`). Generated from the checkpoint's own `vocab.json` (sort by id; no token contains a raw newline/CR — byte-level BPE maps them to visible chars; head ids asserted). Byte-identical to the SmolLM `vocab.txt` (same tokenizer). |
| `reference/smollm2_logits_f32.bin` | Canonical full logits tensor, raw little-endian float32, C order, 3,145,728 bytes, sha256 `58983c9054aab21a30063fc10450777c1b056323e76fbbd9be5d5d7c0c745f1c`. |
| `reference/reference_manifest.json` | Input ids, shape, dtype, causal ppl, sha256, tolerance, per-position top-5. |
| `.gitignore` | Ignores generated blobs (`arg0.data`, `*.mlir`, `/smollm2_driver_logits_f32.bin`, `output/`, `__pycache__/`); `reference/` stays committed (verified with `git check-ignore`: `arg0.data`/`forward.mlir`/driver bin/`output/jit_logits_f32.bin` ignored with exit 0; `reference/smollm2_logits_f32.bin`, `reference/reference_manifest.json` and `vocab.txt` NOT ignored). |
| `forward.mlir`, `subgraph0.mlir`, `arg0.data`, `output/` | Generated by the importer / CMake target / local runs; not tracked. |

Generated artifact sizes (from the direct importer run in **Local Run**):

```
-rw-rw-r-- 1 ROXY ROXY 538060160 arg0.data      # 134,515,040 float32 = 134,515,008 trainable params + 32-elem inv_freq buffer
-rw-rw-r-- 1 ROXY ROXY    154230 forward.mlir
-rw-rw-r-- 1 ROXY ROXY    648391 subgraph0.mlir
```

The `@forward` signature `buddy-smollm2-main.cpp` is written against — read
from the emitted `forward.mlir`, not assumed:

```mlir
func.func @forward(%arg0: memref<134515040xf32>, %arg1: memref<1x16xi64>)
  -> memref<1x16x49152xf32>
```

Llama takes a single runtime input (`input_ids`); everything else is lifted:
embed_in (tied with the lm_head region; the packed main graph addresses both
49152x576 views as subviews of the same `arg0.data` prefix), the 30 blocks'
q/k/v/o projections, SwiGLU gate/up/down projections, the two RMSNorm weights
per block, the final norm and the rotary `inv_freq` buffer — lifted tensors in
`subgraph0.mlir`, all float32, verified by the importer's layout check and
bit-exact read-back.

## Canonical Reference

Produced with the **official implementation** (upstream `transformers 5.5.4`,
`torch 2.12.0`, CPU), never with a hand-written reimplementation:

```
python3 smollm2-ppl.py --write-reference
-> wrote .../reference/smollm2_logits_f32.bin (3145728 bytes, sha256 58983c9054aab21a...)
-> wrote .../reference/reference_manifest.json
-> reference logits shape=(1, 16, 49152) cpu_inference=0.037s causal_ppl=5.052739
```

* Fixed input (from `reference_manifest.json`): the sentence
  `"Once upon a time there was a little girl who lived in a small village."`
  tokenizes with the official SmolLM2 GPT-2 BPE tokenizer to exactly 16 ids
  (no padding; the script fails hard otherwise):
  `input_ids = [6403, 1980, 253, 655, 665, 436, 253, 1838, 8180, 617, 4161,
  281, 253, 1165, 6560, 30]` = `['Once', 'Ġupon', 'Ġa', 'Ġtime', 'Ġthere',
  'Ġwas', 'Ġa', 'Ġlittle', 'Ġgirl', 'Ġwho', 'Ġlived', 'Ġin', 'Ġa', 'Ġsmall',
  'Ġvillage', '.']` (same 16 ids as the SmolLM round — the tokenizer is
  identical; the weights are not).
* Canonical tensor: `reference/smollm2_logits_f32.bin`, shape `[1, 16, 49152]`,
  float32, sha256 `58983c9054aab21a30063fc10450777c1b056323e76fbbd9be5d5d7c0c745f1c`.
* **Canonical ppl (causal-LM loss over the shifted ids)**: `causal_ppl =
  5.052739` (official `labels=` forward).
* **Discrete expectation** (the driver's fail-hard contract,
  `kExpectedArgmaxIds` in `buddy-smollm2-main.cpp`): the per-position argmax
  token ids of the canonical tensor are
  `[346, 253, 655, 28, 436, 253, 1838, 8180, 3365, 4161, 281, 253, 1165,
  6560, 281, 2306]` — the ids predicted after each prompt token (tokens
  `Ġyou`, `Ġa`, `Ġtime`, `,`, `Ġwas`, `Ġa`, `Ġlittle`, `Ġgirl`, `Ġnamed`,
  `Ġlived`, `Ġin`, `Ġa`, `Ġsmall`, `Ġvillage`, `Ġin`, `ĠShe`). These
  landing points are *measured from the official implementation*; they are this
  checkpoint's own deterministic predictions, not a quality claim. They
  differ from the SmolLM-135M expectations at positions 6/9/12/14 — expected,
  since the weights differ.
* **Reproduce / compare commands** (each verified below in **Local Run**):
  `python3 smollm2-ppl.py --write-reference` regenerates the reference; the
  jit/offline candidates are checked with `python3 smollm2-ppl.py --check
  output/jit_logits_f32.bin`; the weight-blob packing order is checked with
  `python3 smollm2-ppl.py --weights arg0.data`.
* **Agreed tolerance:** `np.allclose(atol=1e-3, rtol=1e-3)` on the complete
  logits tensor plus exact per-position argmax agreement (metrics in the
  manifest), and recon-weights ppl within `PPL_TOL = 0.05` of the canonical
  ppl. Element-wise tolerance is the auxiliary gate of this stage — the
  conclusion is the discrete argmax / ppl agreement.

## Local Run

**Device: host CPU** (x86-64). The host has an NVIDIA A800, but no CUDA-enabled
Python toolchain exists in this environment (`torch.cuda.is_available()` →
`False` in `result/bin/python3`, and the buddy ExecutionEngine is a host-CPU
LLVM JIT with no GPU codegen path — see **Known Limitation** #4). The official
single forward is **0.037 s** on CPU; the full TOSA→LLVM jit inference of the
imported graph is reported below.

Environment (nix develop of the buckyball repo; `result/bin/python3` = Python
3.14.6 with `torch 2.12.0`, `transformers 5.5.4`, `huggingface_hub 1.16.0`):

```bash
cd /home/ROXY/code/bb_work/buckyball
export BB=$PWD RISCV=$BB/result
export BUDDY_MLIR_BUILD_DIR=$BB/compiler/thirdparty/buddy-mlir/build
export LLVM_MLIR_BUILD_DIR=$BB/compiler/thirdparty/buddy-mlir/llvm/build
export PYTHONPATH="$LLVM_MLIR_BUILD_DIR/tools/mlir/python_packages/mlir_core:$BUDDY_MLIR_BUILD_DIR/python_packages"
export HF_HUB_OFFLINE=1
# this host's harness *_proxy vars break httpx URL parsing; keep http(s)_proxy
# or strip them all for the offline runs below (same quirk as the SmolLM round)

cd bb-tests/workloads/src/ModelTest/e2e/models/models/SmolLM2
# 1) canonical reference (official implementation)
python3 smollm2-ppl.py --write-reference
# 2) local run: AOT import + execute the imported graph on the host CPU
python3 import-smollm2.py --output-dir . --jit-check \
        --dump-candidate output/jit_logits_f32.bin
# 3) independent re-check of the dumped output through the workload's own checker
python3 smollm2-ppl.py --check output/jit_logits_f32.bin
# 4) packing-order check: inject arg0.data into a fresh official model
python3 smollm2-ppl.py --weights arg0.data
```

The **local carrier chosen for this stage is the importer's own host execution
path** (step 2, a `--jit-check` host run of the imported graph) plus the
workload's own checker (step 3) and the recon-weights ppl (step 4). This
carrier is a **team-side addition, not a general upstream form** (upstream's
chip-agnostic host form is the `.rax` + `buddy-cli` runner-plugin packaging,
which SmolLM2 does not carry); the `.rax` form was not chosen because it
requires a runner plugin/spec manifest this model lacks. Same carrier decision
as the SmolLM / BertMedium / Pythia rounds.

Step 2 is the model-side run: it imports the model exactly as `CMakeLists.txt`
does and then executes **that same imported graph** on the host CPU through the
buddy frontend's own TOSA → LLVM pipeline and MLIR `ExecutionEngine`, i.e. a
different numerical path from the aten kernels that produced the reference. No
`buddy-opt`, no `buddy-translate`, no `buddy-llc`, no chip/core target is
involved. Verbatim tail of step 2:

```
importing HuggingFaceTB/SmolLM2-135M: input_ids=(1, 16) ([6403, ..., 30])
forward.mlir + 1 subgraph(s) -> .../models/SmolLM2
arg0.data: 134515040 float32 (538060160 bytes) bit-exact
jit compile: 1966.4s   host-cpu inference: 176.83s
jit logits shape: (1, 16, 49152)
  max_abs_diff: 0.00042819976806640625
  mean_abs_diff: 3.3305692340945825e-05
  max_rel_diff: 2.5671701431274414
  relative_l2_error: 7.443965841957834e-06
  argmax_matches: True
  per_position_argmax_matches: True
  allclose: True
  per-position argmax ids: [346, 253, 655, 28, 436, 253, 1838, 8180, 3365, 4161, 281, 253, 1165, 6560, 281, 2306]
  candidate logits -> output/jit_logits_f32.bin
JIT-CHECK: PASS
```

Step 3 (independent process, workload's own checker) — verbatim:

```
  "max_abs_diff": 0.00042819976806640625,
  "mean_abs_diff": 3.3305692340945825e-05,
  "max_rel_diff": 2.5671701431274414,
  "relative_l2_error": 7.443965841957834e-06,
  "argmax_matches": true,
  "per_position_argmax_matches": true,
  "allclose": true,
  "atol": 0.001, "rtol": 0.001, "n_elements": 786432,
  "candidate_ppl": 5.052755,
  "canonical_ppl": 5.052739
PASS: output/jit_logits_f32.bin vs smollm2_logits_f32.bin
```

Step 4 (recon-weights ppl; bit-exact blob ⇒ identical ppl, which is exactly the
packing-order check) — verbatim:

```
canonical ppl : 5.052739
recon ppl     : 5.052739
recon args    : 134515040 float32 elements consumed exactly
RECON-PPL: PASS
```

Timings summary: canonical reference 0.037 s / run (~10 s wall including model
load); local CPU model run splits as 1966.4 s one-off TOSA→LLVM jit compile
plus 176.83 s single jit inference of the imported graph; the weight blob
538.1 MB is written and re-read bit-exact. `g++ -std=c++17 -fsyntax-only
-Wall` passes on the driver against the buddy headers (exit 0); `python3 -m
py_compile` passes on the two Python files.

## Build Binding

Configure and build were run through CMake/Ninja directly (the
`bbdev workload --build` path needs the chip layout that this stage
deliberately does not add — see **Known Limitation** #2). The upstream
`e2e/models/lib/CMakeLists.txt` gate requires per-chip compiler targets, so a
chip-less configure needs the same shim bbdev injects: `BUCKYBALL_TARGETS` plus
an ISA dir generated from the chip.pb (`pb_to_target_registry.py --isa-dir`, the
pebble chip.pb already installed under
`examples/chips/pebble/configs/generated/chip.pb`).

```bash
cd /home/ROXY/code/bb_work/buckyball
export BB=$PWD RISCV=$BB/result
export BUDDY_MLIR_BUILD_DIR=$BB/compiler/thirdparty/buddy-mlir/build
export LLVM_MLIR_BUILD_DIR=$BB/compiler/thirdparty/buddy-mlir/llvm/build
export PYTHONPATH="$LLVM_MLIR_BUILD_DIR/tools/mlir/python_packages/mlir_core:$BUDDY_MLIR_BUILD_DIR/python_packages"
export HF_HUB_OFFLINE=1
python3 compiler/scripts/pb_to_target_registry.py --repo $PWD \
  --chip-pb examples/chips/pebble/configs/generated/chip.pb \
  --isa-dir .dsh/smollm2-isa --print-targets   # prints: pebble

cmake -S bb-tests/workloads/src/ModelTest/e2e/models -B .dsh/build-smollm2 -G Ninja \
      -DMODEL=smollm2 -DWORKLOAD_LIB_DIR=$BB/bb-tests/workloads/lib \
      -DPython3_EXECUTABLE=$BB/result/bin/python3 \
      -DBUCKYBALL_TARGETS=pebble -DBUCKYBALL_ISA_DIR=$BB/.dsh/smollm2-isa
# -- Enabled model: smollm2
# -- Configuring done (0.8s) / -- Generating done (0.0s)

rm -f bb-tests/workloads/src/ModelTest/e2e/models/models/SmolLM2/{forward.mlir,subgraph0.mlir,arg0.data}
cmake --build .dsh/build-smollm2 --target smollm2-model-build
# [1/1] Generating SmolLM2 forward.mlir, subgraph0.mlir and parameters
# importing HuggingFaceTB/SmolLM2-135M: input_ids=(1, 16) ([6403, ..., 30])
# forward.mlir + 1 subgraph(s) -> .../models/SmolLM2
# arg0.data: 134515040 float32 (538060160 bytes) bit-exact
```

The two registration points required at this stage (verified by
`buckyball_workload_audit`):

1. `bb-tests/workloads/src/ModelTest/e2e/models/CMakeLists.txt` — `SMOLLM2`
   appended to the `foreach(model_flag IN ITEMS ...)` MODEL reset list (so
   `-DMODEL=<other>` clears it).
2. `bb-tests/workloads/src/ModelTest/e2e/models/models/CMakeLists.txt` —
   `set(MODEL_SMOLLM2_DIR ${MODEL_DIR}/SmolLM2)` plus
   `if (MODEL_SMOLLM2) add_subdirectory(SmolLM2) endif()`.

The model directory's CMakeLists produces no executable (`*-run` targets live
in `archs/` and belong to the bind round).

## Known Limitation

1. **`buddy-smollm2-main.cpp` is not compiled or run in this stage.** Linking it
   needs the chip-bound `buddy-opt`/`buddy-translate`/`buddy-llc` pipeline,
   explicitly out of scope here. It is checked two ways instead: its
   `extern "C"` declaration and sizes match the `@forward` signature read from
   the emitted `forward.mlir` (`memref<134515040xf32>`, `memref<1x16xi64>` in,
   `memref<1x16x49152xf32>` out), and `g++ -std=c++17 -fsyntax-only -Wall`
   passes against the buddy headers (see **Local Run**).
2. **SmolLM2 has no pebble-bind assets yet — it is NOT bound.** By design this
   stage adds no `archs/buckyball/pebble/SmolLM2/` layout, no bbdev
   `MODEL_LAYOUT` entry and no parent `bb-tests/workloads/scripts/build.py::_MODELS`
   entry, so `workload --build '--chip pebble'` (the expected CI test of this
   round) cannot select this model yet. The chip-bind round (later stage,
   chip-lead territory) owns that work; `chip: pebble` in this round's checklist
   is an intention, not a binding. The `_MODELS` key reserved for that round
   is lowercase `smollm2` (the existing `smollm` key maps to SmolLM-135M).
3. **The driver takes fixed input ids, not a C++ tokenization.** SmolLM2 uses
   the GPT-2-style byte-level BPE tokenizer, which buddy's `TextContainer` does
   not implement (it has Bert/Llama/Qwen3/Gemma4 tokenizers only). The ids are
   hardcoded from the official tokenizer's output (committed in
   `reference/reference_manifest.json`); `vocab.txt` is committed for id→token
   display. A C++ SmolLM2 tokenizer, if ever needed by a later stage, would be
   an upstream toolchain addition, not part of this workload.
4. **Host-CPU jit inference exceeds the 10 s per-inference rule; GPU is not
   available for this path.** The playbook says to switch to local GPU when a
   single forward exceeds 10 s; this environment cannot:
   `torch.cuda.is_available()` is False in the only Python toolchain (a
   CPU-only nix torch), and the buddy ExecutionEngine lowers to host-CPU LLVM
   JIT with no GPU codegen path. The canonical reference producer (official
   aten forward) is 0.037 s on CPU, well under the rule. Recorded here rather
   than silently switching carriers (same finding as the SmolLM /
   BertMedium / Pythia rounds).
5. **The importer traces one full forward (no KV-cache decode).** The Qwen3
   precedent additionally imports prefill/decode graphs with `StaticCache` and
   carries `trace/` tomls — those are the later quant / cycle-trace / bind
   rounds' assets of the same model, deliberately not copied at this stage. The
   `quant/` and `trace/` dirs do not exist under `SmolLM2/` yet.
6. **Hub packaging nuance (measured, differs from the SmolLM-135M
   precedent): this snapshot's `model.safetensors` genuinely stores
   bfloat16.** All 272 tensors are BF16 (file is 269,060,552 B = 30,528 B
   safetensors header + 269,030,016 B payload + 8 B alignment padding), even
   though the same-dtype declaration exists in both checkpoints; SmolLM-135M's
   snapshot stored float32 despite it. The fp32 load of this workload
   converts bf16 → float32 (a value-preserving upcast), so the arg0 blob
   packing convention is unaffected — but a brief based on the SmolLM-135M
   precedent or on the config declaration alone would misreport this
   checkpoint's storage layout. The 134,515,008 trainable-parameter count is
   identical in both checkpoints (same architecture).
7. **`max_rel_diff` is 2.57 while `atol` is 1e-3.** Reference logits near zero
   make the plain relative bound meaningless at a handful of positions; the
   operative metric is `np.allclose(atol=1e-3, rtol=1e-3)` (passes;
   `max_abs_diff` is 4.3e-4, relative-L2 is 7.4e-6) plus exact per-position
   argmax agreement — that is the agreed metric (manifest `tolerance`).
   (Same shape as the SmolLM round's finding #7; re-measured for this
   checkpoint in **Local Run**.)
8. **The per-position argmaxes are this checkpoint's measured values.** They
   are deterministic outputs of the official implementation (this is the base
   135M model, not an instruct variant — its greedy continuations are plain
   text, not a quality claim), and they are stable across the two numerical
   paths (jit argmax list is identical to the official one).
9. **Hub access and proxies.** This host reaches HuggingFace through a local
   HTTP proxy; the harness's `no_proxy`/`NODE_USE_ENV_PROXY` vars break httpx
   URL parsing in this Python (the `[::1]` no_proxy entry → `httpx.InvalidURL:
   Invalid port: ':1]'`; same quirk as the SmolLM / Pythia rounds). Downloads
   ran with `HF_ENDPOINT=https://hf-mirror.com` (which 308-redirects to
   huggingface.co for this repo) or unset, keeping only `http(s)_proxy`. The
   `hf download` CLI additionally stalled on the Xet-backed
   `model.safetensors`; the blob was fetched with `curl -L` through the proxy
   to the exact hub cache path and the snapshot symlink created, then
   `HF_HUB_OFFLINE=1` verified the load. `smollm2-ppl.py` loads with
   `local_files_only=True` (the model must be in the hub cache at the snapshot
   commit `93efa2f097...`, or reachable via the `SMOLLM2_135M_MODEL_PATH`
   override); the committed workload consumes only committed/regenerable
   assets with `HF_HUB_OFFLINE=1`.
10. **A 3.1 MB binary reference file is committed.** It is the full canonical
    logits tensor, which the "compare the complete output, not just an argmax"
    rule calls for; it is regenerable via `smollm2-ppl.py --write-reference` and
    pinned by the sha256 in `reference/reference_manifest.json`. The 538.1 MB
    `arg0.data` blob and the `.mlir` artifacts are generated and gitignored.
