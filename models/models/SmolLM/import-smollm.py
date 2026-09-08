#!/usr/bin/env python3
# ===- import-smollm.py -----------------------------------------------------===
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
# Ahead-of-time importer for the SmolLM (HuggingFaceTB/SmolLM-135M) e2e
# workload, model side only. Same AutoModelForCausalLM + DynamoCompiler
# pathway as the Qwen3 / Pythia precedents (LlamaForCausalLM falls into the
# same shape), same artifact contract as the BertMedium precedent:
#
#   <output-dir>/forward.mlir    packed-parameter main graph (`@forward`)
#   <output-dir>/subgraph0.mlir  the single fused TOSA subgraph
#   <output-dir>/arg0.data       float32 parameters, concatenated in trace order
#
# SmolLM-135M is LlamaForCausalLM (model_type=llama): hidden 576, 30 layers,
# 9 query heads / 3 KV heads (GQA), SwiGLU MLP 1536, RMSNorm 1e-5, RoPE
# theta 10000, vocab 49152, tied embeddings, bf16 checkpoint loaded float32.
# The importer traces one full forward over the fixed 16-token workload input
# (no KV cache), which exercises every network op; the fixed input ids and the
# expected per-position argmax ids come from smollm-ppl.py / reference/.
#
# Optional `--jit-check` executes the *same imported graph* on the host CPU
# through the buddy frontend's own TOSA -> LLVM pipeline + MLIR ExecutionEngine
# (no buddy-opt / no chip target) and compares the complete logits tensor with
# the canonical reference written by smollm-ppl.py --write-reference.
#
# ===---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import importlib.util
import os
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
from transformers import AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM-135M"
HERE = Path(__file__).resolve().parent


def load_reference_module():
    """Reuse the fixed-input / tolerance definitions of the reference script."""
    path = HERE / "smollm-ppl.py"
    spec = importlib.util.spec_from_file_location("smollm_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_path() -> str:
    return os.environ.get("SMOLLM_135M_MODEL_PATH", MODEL_ID)


def build_inputs():
    """Return the input_ids tensor for the fixed workload text.

    The dict insertion order is the traced graph's runtime-input order. SmolLM
    takes a single runtime input (input_ids); every other tensor is lifted.
    """
    reference = load_reference_module()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path())
    input_ids = reference.fixed_input_ids(tokenizer)
    return input_ids


def import_graph(model, input_ids):
    """Trace the model with the buddy frontend and fuse it into one subgraph."""
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    with torch.no_grad():
        graphs = dynamo_compiler.importer(model, input_ids=input_ids)
    if len(graphs) != 1:
        raise RuntimeError(
            f"expected a single graph without breaks, got {len(graphs)}; "
            "the model-side import contract does not support graph breaks"
        )
    graph = graphs[0]
    graph.fuse_ops([simply_fuse])
    return dynamo_compiler, graph


def check_param_layout(params) -> None:
    """Fail hard unless every lifted parameter is float32.

    SmolLM packs a single arg0.data blob of float32 parameters (the bf16
    checkpoint is loaded float32 by import-smollm.py). Any other dtype breaks
    the one-blob packing convention and must fail loudly, not be silently
    packed.
    """
    for index, param in enumerate(params):
        if param.detach().numpy().dtype != np.dtype("float32"):
            raise RuntimeError(
                f"param {index} is {param.detach().numpy().dtype}, expected "
                "float32; the single-arg0 packing convention does not hold"
            )


def write_artifacts(graph, params, output_dir: Path) -> dict:
    """Emit forward.mlir, subgraph<i>.mlir and arg0.data."""
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

    # SmolLM convention: every lifted tensor is float32; one concatenated blob.
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
        [param.detach().numpy().reshape([-1]).astype("<f4") for param in params]
    )
    on_disk = np.frombuffer((output_dir / "arg0.data").read_bytes(), dtype="<f4")
    if on_disk.size != flat_float.size or not np.array_equal(on_disk, flat_float):
        raise RuntimeError("arg0.data does not round-trip the float32 parameters")
    print(
        f"arg0.data: {on_disk.size} float32 "
        f"({output_dir.joinpath('arg0.data').stat().st_size} bytes) bit-exact"
    )


def jit_check(model, input_ids, dump_candidate=None) -> int:
    """Run the imported graph on the host CPU and compare full logits."""
    refmod = load_reference_module()
    reference_path = refmod.DEFAULT_REFERENCE_DIR / "smollm_logits_f32.bin"
    if not reference_path.is_file():
        raise SystemExit(
            f"missing canonical reference {reference_path}; run "
            "python3 smollm-ppl.py --write-reference first"
        )

    # A fresh compiler instance, unfused: the ExecutionEngine then takes
    # (parameters..., runtime inputs...) exactly in trace order.
    dynamo_compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    with torch.no_grad():
        graphs = dynamo_compiler.importer(model, input_ids=input_ids)
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
    canonical = refmod.load_logits(
        reference_path, [1, refmod.SEQ_LEN, refmod.VOCAB_SIZE]
    )
    report = refmod.compare(candidate, canonical)
    print(f"jit compile: {compiled_in:.1f}s   host-cpu inference: {ran_in:.2f}s")
    print("jit logits shape:", tuple(candidate.shape))
    for key in (
        "max_abs_diff",
        "mean_abs_diff",
        "max_rel_diff",
        "relative_l2_error",
        "argmax_matches",
        "per_position_argmax_matches",
        "allclose",
    ):
        print(f"  {key}: {report[key]}")
    argmax = candidate.reshape(-1, refmod.VOCAB_SIZE).argmax(axis=1)
    print("  per-position argmax ids:", argmax.tolist())
    ok = report["allclose"] and report["per_position_argmax_matches"]
    if dump_candidate is not None:
        dump_candidate = Path(dump_candidate)
        dump_candidate.parent.mkdir(parents=True, exist_ok=True)
        dump_candidate.write_bytes(candidate.astype("<f4").tobytes(order="C"))
        print(f"  candidate logits -> {dump_candidate}")
    print("JIT-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AOT importer for the SmolLM e2e workload"
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
        "smollm-ppl.py --check can re-verify it independently",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    model_id = args.model_id if args.model_id else model_path()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    model.config.use_cache = False
    input_ids = build_inputs()
    print(
        f"importing {model_id}: input_ids={tuple(input_ids.shape)} "
        f"({[int(v) for v in input_ids.reshape(-1)]})"
    )

    dynamo_compiler, graph = import_graph(model, input_ids)
    params = dynamo_compiler.imported_params[graph]
    stats = write_artifacts(graph, params, output_dir)
    print(f"forward.mlir + {stats['subgraphs']} subgraph(s) -> {output_dir}")
    verify_param_files(params, output_dir)

    if args.jit_check:
        return jit_check(model, input_ids, args.dump_candidate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
