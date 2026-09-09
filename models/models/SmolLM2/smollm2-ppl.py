#!/usr/bin/env python3
# ===- smollm2-ppl.py ---------------------------------------------------------===
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
# Canonical (official-implementation) reference for the SmolLM2 e2e workload.
#
# Runs `HuggingFaceTB/SmolLM2-135M` (LlamaForCausalLM, model_type=llama: hidden
# 576, 30 layers, 9 query heads, 3 KV heads (GQA), SwiGLU MLP 1536, RMSNorm
# 1e-5, RoPE theta 100000, max positions 8192, vocab 49152, tied embeddings,
# bf16 checkpoint loaded float32 here) with the upstream HuggingFace
# transformers package on the fixed sentence below (exactly SEQ_LEN GPT-2 BPE
# tokens, no padding) and writes the *complete* logits tensor, not just an
# argmax:
#
#   reference/smollm2_logits_f32.bin   raw little-endian float32, C order
#   reference/reference_manifest.json shape/dtype/sha256/top-k/ppl + input ids
#
# The same script verifies a candidate logits file against the canonical
# reference in the agreed tolerance, and re-measures the causal-LM perplexity
# after injecting the exported weight blob (`arg0.data`) back into a fresh
# official model — the packing-order check of this workload:
#
#   python3 smollm2-ppl.py --write-reference             # generate reference/
#   python3 smollm2-ppl.py --check <candidate.bin>       # verify full logits
#   python3 smollm2-ppl.py --weights <arg0.data>         # recon-weights ppl
#
# ===---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
# Fixed workload input: tokenizes (GPT-2 byte-level BPE, 49152 vocab) to
# exactly SEQ_LEN ids without padding — verified by --write-reference, which
# fails otherwise.
TEXT = "Once upon a time there was a little girl who lived in a small village."
SEQ_LEN = 16
VOCAB_SIZE = 49152


def model_path() -> str:
    """Model id or directory; SMOLLM2_135M_MODEL_PATH overrides the hub id."""
    return os.environ.get("SMOLLM2_135M_MODEL_PATH", MODEL_ID)


# Agreed tolerance for float32 logits produced by a different summation order
# (upstream aten kernels vs. the TOSA/LLVM codegen of this workload).
ATOL = 1e-3
RTOL = 1e-3
# Tolerance for the causal-LM perplexity of the recon-weights model vs. the
# canonical perplexity (fp32 CPU numerics of the same torch build).
PPL_TOL = 0.05

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = HERE / "reference"


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path(), local_files_only=True)


def load_model() -> "torch.nn.Module":
    from transformers import AutoModelForCausalLM

    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path(), dtype=torch.float32, local_files_only=True
    ).eval()
    model.config.use_cache = False
    return model


def fixed_input_ids(tokenizer) -> torch.Tensor:
    """Tokenize the fixed sentence; fail hard unless it is exactly SEQ_LEN ids."""
    ids = tokenizer(TEXT, add_special_tokens=False)["input_ids"]
    if len(ids) != SEQ_LEN:
        raise SystemExit(
            f"TEXT tokenizes to {len(ids)} ids, expected exactly {SEQ_LEN}: "
            f"{tokenizer.convert_ids_to_tokens(ids)}"
        )
    return torch.tensor([ids], dtype=torch.int64)


def causal_ppl(logits: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy ppl over the shifted logits/labels (C-order float32)."""
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels.astype(np.int64))
    shift_logits = logits_t[..., :-1, :].reshape(-1, VOCAB_SIZE)
    shift_labels = labels_t[..., 1:].reshape(-1)
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
    return float(math.exp(loss))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_reference() -> tuple[np.ndarray, dict]:
    """Full-output reference from the official implementation, on CPU."""
    tokenizer = load_tokenizer()
    input_ids = fixed_input_ids(tokenizer)
    model = load_model()
    started = time.perf_counter()
    logits = model(input_ids=input_ids).logits
    elapsed = time.perf_counter() - started
    logits_np = logits.detach().cpu().contiguous().numpy().astype(np.float32)
    ppl = causal_ppl(logits_np, input_ids.numpy())
    meta = {
        "model_id": MODEL_ID,
        "text": TEXT,
        "input_ids": input_ids.reshape(-1).tolist(),
        "device": "cpu",
        "dtype": "float32",
        "shape": list(logits_np.shape),
        "causal_ppl": round(ppl, 6),
        "inference_seconds": round(elapsed, 6),
    }
    return logits_np, meta


def top_k(logits: np.ndarray, tokenizer, k: int = 5) -> list[dict]:
    """Per-position top-k predictions (kept in the manifest for eyeball checks)."""
    per_position = logits.reshape(logits.shape[1], logits.shape[2])
    report = []
    for position in range(per_position.shape[0]):
        indices = np.argsort(per_position[position])[-k:][::-1]
        report.append(
            {
                "position": position,
                "top": [
                    {
                        "token_id": int(tid),
                        "token": tokenizer.convert_ids_to_tokens([int(tid)])[0],
                        "logit": float(per_position[position][tid]),
                    }
                    for tid in indices
                ],
            }
        )
    return report


def compare(candidate: np.ndarray, reference: np.ndarray) -> dict:
    diff = np.abs(candidate - reference)
    denom = np.maximum(np.abs(reference), 1e-12)
    return {
        "shape": list(candidate.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float((diff / denom).max()),
        "relative_l2_error": float(
            np.linalg.norm(candidate - reference) / np.linalg.norm(reference)
        ),
        "argmax_matches": bool(candidate.argmax() == reference.argmax()),
        "per_position_argmax_matches": bool(
            np.array_equal(
                candidate.reshape(-1, VOCAB_SIZE).argmax(axis=1),
                reference.reshape(-1, VOCAB_SIZE).argmax(axis=1),
            )
        ),
        "allclose": bool(
            np.allclose(candidate, reference, atol=ATOL, rtol=RTOL, equal_nan=False)
        ),
        "atol": ATOL,
        "rtol": RTOL,
        "n_elements": int(candidate.size),
    }


def load_logits(path: Path, shape: list[int]) -> np.ndarray:
    raw = path.read_bytes()
    expected = int(np.prod(shape)) * 4
    if len(raw) != expected:
        raise SystemExit(
            f"{path}: expected {expected} bytes for shape {shape}, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()


def recon_ppl_check(weights: Path, expected_ppl: float) -> int:
    """Inject arg0.data into a fresh official model and re-measure the ppl.

    arg0.data is the concatenation of the lifted graph parameters in trace
    order (import-smollm2.py). A wrong packing order would scramble tensors and
    blow the perplexity up; the check fails hard when the recon model's ppl
    deviates from the canonical ppl by more than PPL_TOL.
    """
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as inductor_decomp

    tokenizer = load_tokenizer()
    input_ids = fixed_input_ids(tokenizer)
    model = load_model()
    raw = weights.read_bytes()
    if len(raw) % 4 != 0:
        raise SystemExit(f"{weights}: size {len(raw)} not a multiple of 4")
    recon = np.frombuffer(raw, dtype="<f4").copy()

    # Same trace as import-smollm2.py: parameter order is the packing order.
    dynamo = DynamoCompiler(
        primary_registry=tosa.ops_registry,
        aot_autograd_decomposition=inductor_decomp,
    )
    with torch.no_grad():
        graphs = dynamo.importer(model, input_ids=input_ids)
    assert len(graphs) == 1
    params = dynamo.imported_params[graphs[0]]
    cursor = 0
    with torch.no_grad():
        for param in params:
            n = param.numel()
            if cursor + n > recon.size:
                raise SystemExit(
                    f"{weights}: truncated at element {cursor}+{n} "
                    f"(blob has {recon.size})"
                )
            chunk = torch.from_numpy(recon[cursor : cursor + n].reshape(param.shape))
            param.data.copy_(chunk)
            cursor += n
    if cursor != recon.size:
        raise SystemExit(
            f"{weights}: {recon.size - cursor} leftover elements beyond the "
            "lifted parameter count"
        )
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    logits_np = logits.detach().cpu().numpy().astype(np.float32)
    ppl = causal_ppl(logits_np, input_ids.numpy())
    ok = abs(ppl - expected_ppl) <= PPL_TOL
    print(f"canonical ppl : {expected_ppl:.6f}")
    print(f"recon ppl     : {ppl:.6f}")
    print(f"recon args    : {recon.size} float32 elements consumed exactly")
    print("RECON-PPL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SmolLM-135M canonical reference generator / checker"
    )
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="run the official model and write reference/ artifacts",
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="candidate logits .bin to verify against the canonical reference",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="arg0.data blob to inject and re-measure the causal-LM ppl",
    )
    parser.add_argument(
        "--print-top-k", type=int, default=5, help="positions to report (-1 = none)"
    )
    args = parser.parse_args()

    reference_path = args.reference_dir / "smollm2_logits_f32.bin"
    manifest_path = args.reference_dir / "reference_manifest.json"

    if args.weights is not None:
        if not args.weights.is_file():
            raise SystemExit(f"missing {args.weights}")
        if not manifest_path.is_file():
            raise SystemExit(
                f"missing canonical manifest {manifest_path}; run --write-reference"
            )
        expected_ppl = float(json.loads(manifest_path.read_text())["causal_ppl"])
        return recon_ppl_check(args.weights, expected_ppl)

    tokenizer = load_tokenizer()

    if args.write_reference:
        logits, meta = run_reference()
        args.reference_dir.mkdir(parents=True, exist_ok=True)
        reference_path.write_bytes(logits.astype("<f4").tobytes(order="C"))
        manifest = {
            **meta,
            "logits_file": reference_path.name,
            "logits_sha256": sha256_file(reference_path),
            "logits_bytes": reference_path.stat().st_size,
            "tolerance": {"atol": ATOL, "rtol": RTOL, "metric": "np.allclose"},
            "ppl_tolerance": {"atol": PPL_TOL, "metric": "abs(causal_ppl - expected)"},
            "top_k_per_position": (
                top_k(logits, tokenizer, args.print_top_k)
                if args.print_top_k > 0
                else []
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"wrote {reference_path} ({reference_path.stat().st_size} bytes, "
            f"sha256 {manifest['logits_sha256'][:16]}...)"
        )
        print(f"wrote {manifest_path}")
        print(
            f"reference logits shape={tuple(logits.shape)} "
            f"cpu_inference={meta['inference_seconds']:.3f}s "
            f"causal_ppl={meta['causal_ppl']:.6f}"
        )
        return 0

    if args.check is None:
        raise SystemExit("nothing to do: pass --write-reference, --check or --weights")

    if not reference_path.is_file() or not manifest_path.is_file():
        raise SystemExit(f"missing canonical reference: {reference_path}")
    manifest = json.loads(manifest_path.read_text())
    reference = load_logits(reference_path, manifest["shape"])
    candidate = load_logits(args.check, list(reference.shape))
    report = compare(candidate, reference)
    report["candidate_ppl"] = round(
        causal_ppl(candidate, np.array(manifest["input_ids"], dtype=np.int64)), 6
    )
    report["canonical_ppl"] = manifest["causal_ppl"]
    print(json.dumps(report, indent=2))
    ok = report["allclose"] and report["per_position_argmax_matches"]
    print(f"{'PASS' if ok else 'FAIL'}: {args.check} vs {reference_path.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
