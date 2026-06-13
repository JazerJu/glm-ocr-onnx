from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "runtime"))

from glm_ocr_onnx import GlmOcrOnnx


def make_test_image() -> Path:
    path = Path("/tmp/glm_ocr_onnx_test.png")
    image = Image.new("RGB", (640, 240), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 45), "GLM OCR TEST", fill="black", font=font)
    draw.text((40, 125), "Hello 2026", fill="black", font=font)
    image.save(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure ONNX Runtime GLM-OCR on one image.")
    parser.add_argument("image", nargs="?", type=Path, help="Path to an input image")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_DIR / "models" / "export",
        help="Directory containing config/tokenizer files and the onnx/ subdirectory",
    )
    parser.add_argument("--prompt", default="请识别图中的所有文字", help="OCR prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum generated tokens")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image if args.image is not None else make_test_image()
    print(f"Image: {image_path}")

    image = Image.open(image_path)
    model = GlmOcrOnnx(args.model_dir, max_tokens=args.max_tokens)
    print(f"Providers: {model.providers}")
    print("OCR: ", end="", flush=True)
    text = model.ocr(image, prompt=args.prompt, max_tokens=args.max_tokens, stream_callback=lambda s: print(s, end="", flush=True))
    print("\n\nFinal text:")
    print(text)


if __name__ == "__main__":
    main()
