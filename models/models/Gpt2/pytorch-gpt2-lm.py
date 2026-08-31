#!/usr/bin/env python3
# ===- pytorch-gpt2-lm.py ----------------------------------------------------===
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
# Canonical (official-implementation) reference for the Gpt2 e2e workload.
#
# Runs `openai-community/gpt2` (GPT2LMHeadModel) with the upstream HuggingFace
# transformers package on the fixed token sequence below and writes the
# *complete* next-token logits tensor, not just an argmax:
#
#   reference/gpt2_lm_logits_f32.bin   raw little-endian float32, C order
#   reference/reference_manifest.json  shape/dtype/sha256/top-k + input ids
#
# The same script also verifies a candidate logits file against the canonical
# reference in the agreed tolerance, which is how the local CPU run and every
# later stage are checked:
#
#   python3 pytorch-gpt2-lm.py --write-reference            # generate
#   python3 pytorch-gpt2-lm.py --check <candidate.bin>      # verify
#
# ===---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "openai-community/gpt2"
# Fixed workload input: one plain-ASCII sentence with no special-token
# substrings. GPT-2's byte-level BPE is deterministic and adds no special
# tokens here, so these ids are the whole (unpadded) prompt.
SENTENCE = "The quick brown fox jumps over the lazy dog"
INPUT_IDS = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290]
SEQ_LEN = len(INPUT_IDS)
VOCAB_SIZE = 50257
# Agreed tolerance for float32 logits produced by a different summation order
# (upstream aten kernels vs. the TOSA/LLVM codegen of this workload). Same bar
# as the Bert/DistilBert precedent and the upstream buddy-mlir JIT check.
ATOL = 1e-3
RTOL = 1e-3

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = HERE / "reference"


def resolve_model_id() -> str:
    """Model repo id, or a pre-staged local directory via GPT2_MODEL_PATH."""
    return os.environ.get("GPT2_MODEL_PATH", MODEL_ID)


def fixed_inputs() -> dict[str, torch.Tensor]:
    """The workload's fixed case: one int64 input_ids tensor, no padding."""
    return {
        "input_ids": torch.tensor([INPUT_IDS], dtype=torch.int64),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_reference(model_id: str) -> tuple[np.ndarray, dict]:
    """Full-output reference from the official implementation, on CPU."""
    from transformers import GPT2LMHeadModel

    torch.set_grad_enabled(False)
    model = GPT2LMHeadModel.from_pretrained(model_id, dtype=torch.float32).eval()
    model.config.use_cache = False
    inputs = fixed_inputs()
    started = time.perf_counter()
    logits = model(**inputs).logits
    elapsed = time.perf_counter() - started
    logits_np = logits.detach().cpu().contiguous().numpy().astype(np.float32)
    meta = {
        "model_id": model_id,
        "sentence": SENTENCE,
        "input_ids": INPUT_IDS,
        "device": "cpu",
        "dtype": "float32",
        "shape": list(logits_np.shape),
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
        "argmax_matches": bool(
            np.array_equal(candidate.argmax(axis=-1), reference.argmax(axis=-1))
        ),
        "top5_token_ids_match": bool(
            all(
                np.array_equal(
                    np.argsort(candidate[0, p])[-5:][::-1],
                    np.argsort(reference[0, p])[-5:][::-1],
                )
                for p in range(reference.shape[1])
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gpt2 language-model canonical reference generator / checker"
    )
    parser.add_argument("--model-id", default=None)
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
        "--print-top-k", type=int, default=5, help="positions to report (-1 = none)"
    )
    args = parser.parse_args()

    model_id = args.model_id or resolve_model_id()
    reference_path = args.reference_dir / "gpt2_lm_logits_f32.bin"
    manifest_path = args.reference_dir / "reference_manifest.json"

    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    if args.write_reference:
        encoded = tokenizer(SENTENCE, add_special_tokens=False)["input_ids"]
        if encoded != INPUT_IDS:
            raise SystemExit(
                f"fixed-case drift: tokenizer produced {encoded}, manifest "
                f"expects {INPUT_IDS}"
            )
        logits, meta = run_reference(model_id)
        args.reference_dir.mkdir(parents=True, exist_ok=True)
        reference_path.write_bytes(logits.astype("<f4").tobytes(order="C"))
        manifest = {
            **meta,
            "logits_file": reference_path.name,
            "logits_sha256": sha256_file(reference_path),
            "logits_bytes": reference_path.stat().st_size,
            "tolerance": {"atol": ATOL, "rtol": RTOL, "metric": "np.allclose"},
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
            f"cpu_inference={meta['inference_seconds']:.3f}s"
        )
        return 0

    if args.check is None:
        raise SystemExit("nothing to do: pass --write-reference or --check")

    if not reference_path.is_file():
        raise SystemExit(f"missing canonical reference: {reference_path}")
    reference = load_logits(
        reference_path, json.loads(manifest_path.read_text())["shape"]
    )
    candidate = load_logits(args.check, list(reference.shape))
    report = compare(candidate, reference)
    print(json.dumps(report, indent=2))
    ok = (
        report["allclose"]
        and report["argmax_matches"]
        and report["top5_token_ids_match"]
    )
    print(f"{'PASS' if ok else 'FAIL'}: {args.check} vs {reference_path.name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
