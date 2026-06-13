# coding=utf-8
"""Create reproducible q4 ONNX artifacts for GLM-OCR.

The export script emits a stable fp32/fp16 baseline:

  - vision_encoder_fp32.onnx
  - embed_tokens_fp32.onnx
  - merger_fp16.onnx

This script derives the q4 runtime files from that baseline:

  - vision_encoder_q4.onnx
  - embed_tokens_q4.onnx

The vision graph exported by PyTorch uses Gemm for linear layers. ONNX Runtime's
MatMulNBits quantizer only handles MatMul with constant weights, so we first
rewrite Gemm(A, W, B, transB=1) into MatMul(A, W.T) + Add(B), then quantize.
"""

from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize GLM-OCR ONNX artifacts to q4.")
    parser.add_argument("--onnx-dir", type=Path, default=Path("models/export/onnx"))
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument(
        "--only",
        choices=["all", "vision", "embed"],
        default="all",
        help="Quantize only a subset of artifacts",
    )
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate Gemm-rewritten model")
    return parser.parse_args()


def save_model(model: onnx.ModelProto, output_path: Path) -> None:
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


def attr_value(node: onnx.NodeProto, name: str, default):
    for attr in node.attribute:
        if attr.name == name:
            return helper.get_attribute_value(attr)
    return default


def rewrite_gemm_to_matmul(input_path: Path, output_path: Path) -> None:
    print(f"Rewriting Gemm -> MatMul+Add: {input_path}")
    model = onnx.load(input_path, load_external_data=True)
    init_by_name = {init.name: init for init in model.graph.initializer}

    input_use_count: Counter[str] = Counter()
    for node in model.graph.node:
        input_use_count.update(name for name in node.input if name)

    removable_weights: set[str] = set()
    new_initializers: list[TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []
    converted = 0

    for node in model.graph.node:
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue

        trans_a = int(attr_value(node, "transA", 0))
        trans_b = int(attr_value(node, "transB", 0))
        alpha = float(attr_value(node, "alpha", 1.0))
        beta = float(attr_value(node, "beta", 1.0))
        if trans_a != 0 or trans_b != 1 or alpha != 1.0 or beta != 1.0:
            raise ValueError(f"Unsupported Gemm attrs in {node.name}: transA={trans_a} transB={trans_b} alpha={alpha} beta={beta}")
        if len(node.input) < 2 or node.input[1] not in init_by_name:
            raise ValueError(f"Gemm weight is not a constant initializer: {node.name}")

        a_name = node.input[0]
        w_name = node.input[1]
        b_name = node.input[2] if len(node.input) > 2 and node.input[2] else None
        out_name = node.output[0]

        w_array = numpy_helper.to_array(init_by_name[w_name])
        w_t_name = w_name + "_matmul_t"
        w_t = np.ascontiguousarray(w_array.T)
        new_initializers.append(numpy_helper.from_array(w_t, w_t_name))
        if input_use_count[w_name] == 1:
            removable_weights.add(w_name)

        matmul_out = out_name + "_matmul"
        new_nodes.append(
            helper.make_node(
                "MatMul",
                [a_name, w_t_name],
                [matmul_out if b_name else out_name],
                name=(node.name or out_name) + "_MatMul",
            )
        )
        if b_name:
            new_nodes.append(
                helper.make_node(
                    "Add",
                    [matmul_out, b_name],
                    [out_name],
                    name=(node.name or out_name) + "_BiasAdd",
                )
            )
        converted += 1

    kept_initializers = [init for init in model.graph.initializer if init.name not in removable_weights]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    model.graph.initializer.extend(new_initializers)

    print(f"Converted Gemm nodes: {converted}")
    save_model(model, output_path)
    remove_stale_external_data(output_path)
    print(f"Wrote intermediate: {output_path}")


def quantize_matmul_nbits(input_path: Path, output_path: Path, block_size: int, symmetric: bool, op_types: tuple[str, ...]) -> None:
    print(f"Quantizing: {input_path}")
    print(f"Output:     {output_path}")
    quantizer = MatMulNBitsQuantizer(
        str(input_path),
        bits=4,
        block_size=block_size,
        is_symmetric=symmetric,
        op_types_to_quantize=op_types,
    )
    quantizer.process()
    save_model(quantizer.model.model, output_path)
    remove_stale_external_data(output_path)
    print_model_summary(output_path)


def print_model_summary(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    ops = Counter((node.domain or "ai.onnx", node.op_type) for node in model.graph.node)
    print("  ops:")
    for (domain, op), count in sorted(ops.items()):
        print(f"    [{domain}] {op}: {count}")
    data_path = path.with_name(path.name + ".data")
    graph_mb = path.stat().st_size / (1024 * 1024)
    data_mb = data_path.stat().st_size / (1024 * 1024) if data_path.exists() else 0.0
    if data_mb:
        print(f"  size: {graph_mb:.1f} MB graph + {data_mb:.1f} MB external data")
    else:
        print(f"  size: {graph_mb:.1f} MB")


def main() -> None:
    args = parse_args()
    onnx_dir = args.onnx_dir
    if not onnx_dir.exists():
        raise SystemExit(f"ONNX directory does not exist: {onnx_dir}")

    if args.only in ("all", "embed"):
        src = onnx_dir / "embed_tokens_fp32.onnx"
        if not src.exists():
            raise SystemExit(f"Missing {src}")
        quantize_matmul_nbits(
            src,
            onnx_dir / "embed_tokens_q4.onnx",
            args.block_size,
            args.symmetric,
            ("Gather",),
        )

    if args.only in ("all", "vision"):
        src = onnx_dir / "vision_encoder_fp32.onnx"
        if not src.exists():
            raise SystemExit(f"Missing {src}")
        if args.keep_temp:
            temp_path = onnx_dir / "vision_encoder_matmul.onnx"
            rewrite_gemm_to_matmul(src, temp_path)
            quant_src = temp_path
        else:
            with tempfile.TemporaryDirectory(prefix="glm_ocr_quant_") as tmp:
                temp_path = Path(tmp) / "vision_encoder_matmul.onnx"
                rewrite_gemm_to_matmul(src, temp_path)
                quantize_matmul_nbits(
                    temp_path,
                    onnx_dir / "vision_encoder_q4.onnx",
                    args.block_size,
                    args.symmetric,
                    ("MatMul",),
                )
                return
        quantize_matmul_nbits(
            quant_src,
            onnx_dir / "vision_encoder_q4.onnx",
            args.block_size,
            args.symmetric,
            ("MatMul",),
        )


if __name__ == "__main__":
    main()
