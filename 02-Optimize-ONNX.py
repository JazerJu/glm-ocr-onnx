# coding=utf-8
"""Optimize exported GLM-OCR ONNX graphs with ONNX Runtime.

The script overwrites the exported ONNX files in place so the runtime keeps using
the same model paths after optimization.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import onnx
from onnxruntime.transformers.optimizer import optimize_model


TARGETS = [
    "vision_encoder_fp32.onnx",
    "embed_tokens_fp32.onnx",
    "merger_fp16.onnx",
    "merger_fp32.onnx",
]

Q4_TARGETS = [
    "vision_encoder_q4.onnx",
    "embed_tokens_q4.onnx",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize exported GLM-OCR ONNX graphs.")
    parser.add_argument("--onnx-dir", type=Path, default=Path("models/export/onnx"))
    parser.add_argument("--opt-level", type=int, default=2, help="ONNX Runtime graph optimization level")
    parser.add_argument("--include-q4", action="store_true", help="Also optimize pre-quantized q4 ONNX files")
    return parser.parse_args()


def save_optimizer_model(optimizer, output_path: Path) -> None:
    model = optimizer.model
    data_path = output_path.with_name(output_path.name + ".data")
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=1024 * 1024,
    )


def remove_stale_external_data(path: Path) -> None:
    keep = path.with_name(path.name + ".data")
    for candidate in [
        path.with_name(path.name + "_data"),
        path.with_name(path.stem + ".onnx_data"),
        path.with_name(path.stem + ".data"),
    ]:
        if candidate != keep and candidate.exists():
            candidate.unlink()


def print_model_summary(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)

    domain_ops: dict[str, set[str]] = defaultdict(set)
    for node in model.graph.node:
        domain = node.domain if node.domain else "ai.onnx"
        domain_ops[domain].add(node.op_type)

    print("  ops:")
    for domain, ops in sorted(domain_ops.items()):
        print(f"    [{domain}] {', '.join(sorted(ops))}")

    graph_mb = os.path.getsize(path) / (1024 * 1024)
    data_path = Path(str(path) + ".data")
    data_mb = os.path.getsize(data_path) / (1024 * 1024) if data_path.exists() else 0
    if data_mb:
        print(f"  size: {graph_mb:.1f} MB graph + {data_mb:.1f} MB external data")
    else:
        print(f"  size: {graph_mb:.1f} MB")


def optimize_one(input_path: Path, opt_level: int) -> None:
    output_path = input_path
    print(f"\nOptimizing: {input_path}")
    print(f"Overwrite:  {output_path}")

    optimizer = optimize_model(
        str(input_path),
        model_type="bert",
        num_heads=0,
        hidden_size=0,
        opt_level=opt_level,
        use_gpu=True,
    )
    save_optimizer_model(optimizer, output_path)
    remove_stale_external_data(output_path)
    print_model_summary(output_path)


def main() -> None:
    args = parse_args()
    if not args.onnx_dir.exists():
        raise SystemExit(f"ONNX directory does not exist: {args.onnx_dir}")

    found = False
    targets = TARGETS + (Q4_TARGETS if args.include_q4 else [])
    for name in targets:
        path = args.onnx_dir / name
        if not path.exists():
            print(f"Skip missing: {path}")
            continue
        found = True
        optimize_one(path, args.opt_level)

    if not found:
        raise SystemExit(f"No target ONNX files found in {args.onnx_dir}")


if __name__ == "__main__":
    main()
