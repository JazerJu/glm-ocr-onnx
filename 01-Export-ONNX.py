# coding=utf-8
"""Export GLM-OCR visual-side ONNX artifacts from the official HF checkpoint.

The official GGUF already covers the language decoder. This script exports only
the pieces needed by the hybrid ONNX + llama.cpp runtime:

  - onnx/vision_encoder_fp32.onnx: pixel patches + position_ids -> visual hidden states
  - onnx/merger_fp16.onnx: visual hidden states -> LLM-space visual embeddings
  - onnx/embed_tokens_fp32.onnx: token ids -> text embeddings

It also copies the small Hugging Face config/tokenizer files into the runtime
directory so `glm_ocr_onnx.py` and `glm_ocr_llama.py` can run without reading the
full safetensors checkpoint at inference time.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GLM-OCR visual-side ONNX artifacts.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/GLM-OCR"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/export"))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--export-fp32-merger", action="store_true", help="Also export merger_fp32.onnx for debugging")
    return parser.parse_args()


def require_export_deps():
    try:
        import torch
        from transformers import GlmOcrForConditionalGeneration
    except Exception as exc:  # pragma: no cover - user environment diagnostic
        raise SystemExit(
            "Export requires a recent Transformers build with GLM-OCR support.\n"
            "Install export dependencies, for example:\n"
            "  pip install torch onnx\n"
            "  pip install git+https://github.com/huggingface/transformers.git\n"
            f"\nOriginal import error: {exc}"
        ) from exc
    return torch, GlmOcrForConditionalGeneration


def copy_runtime_configs(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]:
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    processor_config = output_dir / "processor_config.json"
    if processor_config.exists():
        return

    preprocessor_path = output_dir / "preprocessor_config.json"
    if not preprocessor_path.exists():
        return
    with open(preprocessor_path, "r", encoding="utf-8") as f:
        image_processor = json.load(f)
    with open(processor_config, "w", encoding="utf-8") as f:
        json.dump(
            {
                "processor_class": image_processor.get("processor_class", "Glm46VProcessor"),
                "image_processor": image_processor,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def make_embed_module(torch, embeddings):
    class EmbedModule(torch.nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.emb = emb

        def forward(self, input_ids):
            return self.emb(input_ids)

    return EmbedModule(embeddings).eval()


def make_merger_module(torch, visual):
    class MergerModule(torch.nn.Module):
        def __init__(self, merger):
            super().__init__()
            self.merger = merger

        def forward(self, hidden_states):
            return self.merger(hidden_states)

    return MergerModule(visual.merger).eval()


def make_fp16_merger_module(torch, visual):
    module = make_merger_module(torch, visual)
    return module.half().eval()


def make_vision_module(torch, visual):
    class VisionNoMerger(torch.nn.Module):
        def __init__(self, visual_model):
            super().__init__()
            self.visual = visual_model
            self.spatial_merge_size = int(visual_model.spatial_merge_size)

        def forward(self, pixel_values, position_ids):
            hidden_states = self.visual.patch_embed(pixel_values)
            cu_seqlens = torch.tensor([0, hidden_states.shape[0]], dtype=torch.int32, device=hidden_states.device)

            rotary_emb = self.visual.rotary_pos_emb(position_ids)
            emb = torch.cat((rotary_emb, rotary_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

            for block in self.visual.blocks:
                hidden_states = block(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )

            hidden_states = self.visual.post_layernorm(hidden_states)
            hidden_states = hidden_states.view(
                -1,
                self.spatial_merge_size,
                self.spatial_merge_size,
                hidden_states.shape[-1],
            )
            hidden_states = hidden_states.permute(0, 3, 1, 2)
            return self.visual.downsample(hidden_states).view(-1, self.visual.config.out_hidden_size)

    return VisionNoMerger(visual).eval()


def main() -> None:
    args = parse_args()
    torch, GlmOcrForConditionalGeneration = require_export_deps()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    print(f"[1/5] Copying runtime config/tokenizer files to {output_dir}")
    copy_runtime_configs(model_dir, output_dir)

    print(f"[2/5] Loading GLM-OCR from {model_dir}")
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    visual = model.model.visual

    print("[3/5] Exporting embed_tokens_fp32.onnx")
    embed_module = make_embed_module(torch, model.get_input_embeddings())
    dummy_ids = torch.ones((1, 8), dtype=torch.long)
    torch.onnx.export(
        embed_module,
        (dummy_ids,),
        onnx_dir / "embed_tokens_fp32.onnx",
        input_names=["input_ids"],
        output_names=["output"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "output": {0: "batch", 1: "seq"}},
        opset_version=args.opset,
        dynamo=False,
    )

    print("[4/5] Exporting vision_encoder_fp32.onnx")
    vision_module = make_vision_module(torch, visual)
    patch_dim = int(visual.config.in_channels) * int(visual.config.temporal_patch_size) * int(visual.config.patch_size) ** 2
    dummy_pixels = torch.randn(4, patch_dim, dtype=torch.float32)
    dummy_pos = torch.zeros(4, 2, dtype=torch.long)
    torch.onnx.export(
        vision_module,
        (dummy_pixels, dummy_pos),
        onnx_dir / "vision_encoder_fp32.onnx",
        input_names=["pixel_values", "position_ids"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "num_patches"},
            "position_ids": {0: "num_tokens"},
            "image_features": {0: "num_tokens"},
        },
        opset_version=args.opset,
        dynamo=True,
    )

    print("[5/5] Exporting merger_fp16.onnx")
    merger_module = make_fp16_merger_module(torch, visual)
    dummy_hidden_fp16 = torch.randn(4, int(visual.config.out_hidden_size), dtype=torch.float16)
    torch.onnx.export(
        merger_module,
        (dummy_hidden_fp16,),
        onnx_dir / "merger_fp16.onnx",
        input_names=["hidden_states"],
        output_names=["projected"],
        dynamic_axes={"hidden_states": {0: "num_tokens"}, "projected": {0: "num_tokens"}},
        opset_version=args.opset,
        dynamo=False,
    )

    if args.export_fp32_merger:
        print("[extra] Exporting merger_fp32.onnx")
        merger_module_fp32 = make_merger_module(torch, visual)
        dummy_hidden_fp32 = torch.randn(4, int(visual.config.out_hidden_size), dtype=torch.float32)
        torch.onnx.export(
            merger_module_fp32,
            (dummy_hidden_fp32,),
            onnx_dir / "merger_fp32.onnx",
            input_names=["hidden_states"],
            output_names=["projected"],
            dynamic_axes={"hidden_states": {0: "num_tokens"}, "projected": {0: "num_tokens"}},
            opset_version=args.opset,
            dynamo=False,
        )

    print(f"Done. ONNX files written to: {onnx_dir}")


if __name__ == "__main__":
    main()
