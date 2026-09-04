#!/usr/bin/env python3
# ===- import-berttiny.py ----------------------------------------------------===
#
# Licensed under the Apache Version 2.0 (the "License");
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
# Ahead-of-time importer for the BertTiny (prajjwal1/bert-tiny) e2e workload,
# model side only. Same contract as the Bert / DistilBert precedents:
#
#   <output-dir>/forward.mlir    packed-parameter main graph (`@forward`)
#   <output-dir>/subgraph0.mlir  the single fused TOSA subgraph
#   <output-dir>/arg0.data       float32 parameters, concatenated in trace order
#   <output-dir>/arg1.data       the trailing int64 parameter (position_ids)
#
# BERT differences against DistilBERT that matter for the signature:
#   * `token_type_ids` is a real runtime input (traced between input_ids
#     and attention_mask, mirroring the Bert precedent's driver order),
#   * 2 hidden layers, hidden size 128 instead of 768 / 6 layers,
#   * same 30522-wide masked-LM logits (embedding/decoder weights tied).
#
# Optional `--jit-check` executes the *same imported graph* on the host CPU
# through the buddy frontend's own TOSA -> LLVM pipeline + MLIR ExecutionEngine
# (no buddy-opt / no chip target) and compares the complete logits tensor with
# the canonical reference written by pytorch-berttiny-mlm.py.
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
from transformers import BertForMaskedLM, BertTokenizerFast

MODEL_ID = "prajjwal1/bert-tiny"
HERE = Path(__file__).resolve().parent


def load_reference_module():
    """Reuse the fixed-input / tolerance definitions of the reference script."""
    path = HERE / "pytorch-berttiny-mlm.py"
    spec = importlib.util.spec_from_file_location("berttiny_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_inputs(model_id: str):
    """Return (inputs, tokenizer) for the fixed workload sentence.

    The dict's insertion order — input_ids, token_type_ids, attention_mask —
    is the traced graph's runtime-input order (`import_graph` expands it as
    kwargs), and it matches the Bert precedent's `_mlir_ciface_forward`
    argument order.
    """
    reference = load_reference_module()
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    return reference.fixed_inputs(tokenizer), tokenizer


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


def check_param_layout(params) -> None:
    """Fail hard unless every parameter but the last is float32 and the last
    is the int64 position_ids buffer (the Bert/DistilBert packing rule)."""
    for index, param in enumerate(params[:-1]):
        if param.detach().numpy().dtype != np.dtype("float32"):
            raise RuntimeError(
                f"param {index} is {param.detach().numpy().dtype}, expected "
                "float32; the arg0/arg1 packing convention does not hold"
            )
    last = params[-1].detach().numpy()
    if last.dtype != np.dtype("int64"):
        raise RuntimeError(
            f"trailing param is {last.dtype}, expected int64 (position_ids); "
            "the arg0/arg1 packing convention does not hold"
        )


def write_artifacts(graph, params, output_dir: Path) -> dict:
    """Emit forward.mlir, subgraph<i>.mlir, arg0.data and arg1.data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    check_param_layout(params)
    driver = GraphDriver(graph)
    for subgraph in driver.subgraphs:
        subgraph.lower_to_top_level_ir()
    for core_id, subgraph in enumerate(driver.subgraphs):
        with open(output_dir / f"subgraph{core_id}.mlir", "w") as module_file:
            print(subgraph._imported_module, file=module_file)
    with open(output_dir / "forward.mlir", "w") as module_file:
        print(driver.construct_main_graph(True), file=module_file)

    # Bert convention: every parameter but the trailing int64 buffer is packed
    # into one float32 blob; the trailing int64 buffer keeps its own file.
    float32_param = np.concatenate(
        [param.detach().numpy().reshape([-1]) for param in params[:-1]]
    )
    float32_param.astype("<f4").tofile(output_dir / "arg0.data")
    int64_param = params[-1].detach().numpy().reshape([-1])
    int64_param.astype("<i8").tofile(output_dir / "arg1.data")
    return {
        "subgraphs": len(driver.subgraphs),
        "arg0_float32_elements": int(float32_param.size),
        "arg1_int64_elements": int(int64_param.size),
    }


def verify_param_files(params, output_dir: Path) -> None:
    """Read the emitted blobs back and require a bit-exact round trip."""
    flat_float = np.concatenate(
        [param.detach().numpy().reshape([-1]).astype("<f4") for param in params[:-1]]
    )
    on_disk = np.frombuffer((output_dir / "arg0.data").read_bytes(), dtype="<f4")
    if on_disk.size != flat_float.size or not np.array_equal(on_disk, flat_float):
        raise RuntimeError("arg0.data does not round-trip the float32 parameters")
    int64_flat = params[-1].detach().numpy().reshape([-1]).astype("<i8")
    int64_disk = np.frombuffer((output_dir / "arg1.data").read_bytes(), dtype="<i8")
    if not np.array_equal(int64_disk, int64_flat):
        raise RuntimeError("arg1.data does not round-trip the int64 parameter")
    print(
        f"arg0.data: {on_disk.size} float32 ({output_dir.joinpath('arg0.data').stat().st_size} bytes) bit-exact"
    )
    print(
        f"arg1.data: {int64_disk.size} int64 ({output_dir.joinpath('arg1.data').stat().st_size} bytes) bit-exact"
    )


def jit_check(model, inputs, tokenizer, dump_candidate=None) -> int:
    """Run the imported graph on the host CPU and compare full logits."""
    reference = load_reference_module()
    reference_path = reference.DEFAULT_REFERENCE_DIR / "berttiny_mlm_logits_f32.bin"
    if not reference_path.is_file():
        raise SystemExit(
            f"missing canonical reference {reference_path}; run "
            "python3 pytorch-berttiny-mlm.py --write-reference first"
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
    top = np.argsort(candidate.reshape(-1))[-5:][::-1]
    print(
        "  top5 tokens:",
        [
            tokenizer.convert_ids_to_tokens([int(i)])[0]
            for i in top % reference.VOCAB_SIZE
        ],
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
        description="AOT importer for the BertTiny e2e workload"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./", help="directory for the AOT artifacts"
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
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
        "pytorch-berttiny-mlm.py --check can re-verify it independently",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    model = BertForMaskedLM.from_pretrained(args.model_id).eval()
    inputs, tokenizer = build_inputs(args.model_id)
    print(
        f"importing {args.model_id}: input_ids={tuple(inputs['input_ids'].shape)} "
        f"token_type_ids={tuple(inputs['token_type_ids'].shape)} "
        f"attention_mask={tuple(inputs['attention_mask'].shape)}"
    )

    dynamo_compiler, graph = import_graph(model, inputs)
    params = dynamo_compiler.imported_params[graph]
    stats = write_artifacts(graph, params, output_dir)
    print(f"forward.mlir + {stats['subgraphs']} subgraph(s) -> {output_dir}")
    verify_param_files(params, output_dir)

    if args.jit_check:
        return jit_check(model, inputs, tokenizer, args.dump_candidate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
