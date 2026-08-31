#!/usr/bin/env python3
# ===- import-gpt2.py ---------------------------------------------------------===
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# Ahead-of-time importer for the Gpt2 (openai-community/gpt2) e2e workload,
# model side only. Same contract as the llama2 precedent, condensed to the
# no-trace form used by the DistilBert initial-adaptation workload:
#
#   <output-dir>/forward.mlir    packed-parameter main graph (`@forward`)
#   <output-dir>/subgraph0.mlir  the single fused TOSA subgraph
#   <output-dir>/arg0.data       float32 parameters, concatenated in trace order
#
# GPT-2 differences against llama2 that matter for the signature:
#   * GPT2LMHeadModel (12 layers, 768 hidden, 50257 BPE vocab, 124M params)
#     instead of LlamaForCausalLM (32 layers, 4096 hidden, 32000 SentencePiece),
#   * the lm_head weight is tied to transformer.wte.weight in the checkpoint
#     (one copy appears in the parameter pack; forward.mlir slices it twice),
#   * the traced forward takes only int64 `input_ids` of shape [1, 9]; the
#     causal mask and position ids are static and fold into the graph,
#   * no int64 buffers ride along in the parameter pack (contrast the
#     DistilBert position_ids buffer).
#
# Optional `--jit-check` executes the *same imported graph* on the host CPU
# through the buddy frontend's own TOSA -> LLVM pipeline + MLIR ExecutionEngine
# (no buddy-opt / no chip target) and compares the complete logits tensor with
# the canonical reference written by pytorch-gpt2-lm.py.
#
# ===---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.graph import GraphDriver
from buddy.compiler.graph.transform import simply_fuse
from buddy.compiler.ops import tosa
from torch._inductor.decomposition import decompositions as inductor_decomp
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).resolve().parent


def load_reference_module():
    """Reuse the fixed-input / tolerance definitions of the reference script."""
    path = HERE / "pytorch-gpt2-lm.py"
    spec = importlib.util.spec_from_file_location("gpt2_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(model_id: str):
    """Official GPT2LMHeadModel in float32, eval mode, cache disabled."""
    model = GPT2LMHeadModel.from_pretrained(model_id, dtype=torch.float32).eval()
    model.config.use_cache = False
    return model


def import_graph(model, inputs):
    """Trace the model with the buddy frontend and fuse it into one subgraph."""
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    with torch.no_grad():
        graphs = dynamo_compiler.importer(model, **inputs)
    if len(graphs) != 1:
        raise RuntimeError(
            f"expected a single graph without breaks, got {len(graphs)}; "
            "the model-side import contract does not support graph breaks"
        )
    graph = graphs[0]
    graph.fuse_ops([simply_fuse])
    return dynamo_compiler, graph


def write_artifacts(graph, params, output_dir: Path) -> dict:
    """Emit forward.mlir, subgraph0.mlir and arg0.data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    driver = GraphDriver(graph)
    for subgraph in driver.subgraphs:
        subgraph.lower_to_top_level_ir()
    for core_id, subgraph in enumerate(driver.subgraphs):
        with open(output_dir / f"subgraph{core_id}.mlir", "w") as module_file:
            print(subgraph._imported_module, file=module_file)
    with open(output_dir / "forward.mlir", "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)

    # GPT-2's traced parameter list is float32 only (the wte weight is tied to
    # the lm_head, so the checkpoint's 124,411,200 parameters appear once).
    for param in params:
        if param.dtype != torch.float32:
            raise TypeError(
                f"unexpected parameter dtype {param.dtype}; the single-pack "
                "contract assumes float32-only parameters"
            )
    float32_param = np.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    float32_param.astype("<f4").tofile(output_dir / "arg0.data")
    return {
        "subgraphs": len(driver.subgraphs),
        "arg0_float32_elements": int(float32_param.size),
    }


def verify_param_files(params, output_dir: Path) -> None:
    """Read the emitted blob back and require a bit-exact round trip."""
    flat_float = np.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params]
    )
    on_disk = np.frombuffer((output_dir / "arg0.data").read_bytes(), dtype="<f4")
    if on_disk.size != flat_float.size or not np.array_equal(on_disk, flat_float):
        raise RuntimeError("arg0.data does not round-trip the float32 parameters")
    print(
        f"arg0.data: {on_disk.size} float32 "
        f"({(output_dir / 'arg0.data').stat().st_size} bytes) bit-exact"
    )


def jit_check(model, inputs, tokenizer, dump_candidate=None) -> int:
    """Run the imported graph on the host CPU and compare full logits."""
    reference = load_reference_module()
    reference_path = reference.DEFAULT_REFERENCE_DIR / "gpt2_lm_logits_f32.bin"
    if not reference_path.is_file():
        raise SystemExit(
            f"missing canonical reference {reference_path}; run "
            "python3 pytorch-gpt2-lm.py --write-reference first"
        )

    # A fresh compiler instance, unfused: the ExecutionEngine then takes
    # (parameters..., runtime inputs...) exactly in trace order.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    with torch.no_grad():
        graphs = dynamo_compiler.importer(model, **inputs)
    graph = graphs[0]
    graph_params = dynamo_compiler.imported_params[graph]
    started = time.perf_counter()
    execute = dynamo_compiler.dynamo_run()
    compiled_in = time.perf_counter() - started
    started = time.perf_counter()
    with torch.no_grad():
        outputs = execute(*graph_params, *graph._runtime_inputs_ref)
    ran_in = time.perf_counter() - started

    candidate = outputs[0]
    candidate = (
        candidate.detach().cpu().numpy()
        if torch.is_tensor(candidate)
        else np.asarray(candidate)
    )
    candidate = candidate.astype(np.float32)
    canonical = reference.load_logits(
        reference_path, [1, reference.SEQ_LEN, reference.VOCAB_SIZE]
    )
    report = reference.compare(candidate, canonical)
    print(f"jit compile: {compiled_in:.1f}s   host-cpu inference: {ran_in:.2f}s")
    print("jit logits shape:", tuple(candidate.shape))
    for key in (
        "max_abs_diff",
        "mean_abs_diff",
        "max_rel_diff",
        "relative_l2_error",
        "argmax_matches",
        "top5_token_ids_match",
        "allclose",
    ):
        print(f"  {key}: {report[key]}")
    for position in range(candidate.shape[1]):
        top = int(candidate[0, position].argmax())
        print(
            f"  pos {position}: argmax id {top} "
            f"token {tokenizer.convert_ids_to_tokens([top])[0]!r}"
        )
    ok = (
        report["allclose"]
        and report["argmax_matches"]
        and report["top5_token_ids_match"]
    )
    if dump_candidate is not None:
        dump_candidate = Path(dump_candidate)
        dump_candidate.parent.mkdir(parents=True, exist_ok=True)
        dump_candidate.write_bytes(candidate.astype("<f4").tobytes(order="C"))
        print(f"  candidate logits -> {dump_candidate}")
    print("JIT-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AOT importer for the Gpt2 e2e workload"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./", help="directory for the AOT artifacts"
    )
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument(
        "--jit-check",
        action="store_true",
        help="also run the imported graph on the host CPU and compare with the "
        "canonical reference (non-zero exit on mismatch)",
    )
    parser.add_argument(
        "--dump-candidate",
        type=str,
        default=None,
        help="with --jit-check, write the host-CPU logits to this .bin so that "
        "pytorch-gpt2-lm.py --check can re-verify it independently",
    )
    args = parser.parse_args()

    reference = load_reference_module()
    model_id = args.model_id or reference.resolve_model_id()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    model = load_model(model_id)
    inputs = reference.fixed_inputs()
    print(f"importing {model_id}: input_ids={tuple(inputs['input_ids'].shape)}")

    dynamo_compiler, graph = import_graph(model, inputs)
    params = dynamo_compiler.imported_params[graph]
    stats = write_artifacts(graph, params, output_dir)
    print(f"forward.mlir + {stats['subgraphs']} subgraph(s) -> {output_dir}")
    verify_param_files(params, output_dir)

    if args.jit_check:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        return jit_check(model, inputs, tokenizer, args.dump_candidate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
