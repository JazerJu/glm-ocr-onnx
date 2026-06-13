<p align="center">
  <a href="README.md">中文版</a> | English
</p>

# GLM-OCR-ONNX

This project focuses on one thing: running GLM-OCR locally with **ONNX Runtime + llama.cpp GGUF**.

GLM-OCR already provides an official GGUF decoder, so this repository does not convert the language model. It handles the ONNX visual path, image-token stitching, mRoPE position IDs, and passes visual embeddings into llama.cpp for text generation.

Typical use cases:

- local OCR
- screenshot or document image recognition
- adding a lightweight OCR tool to an AI agent
- reusing the official GGUF decoder without loading the full Transformers/PyTorch stack

## 1. Setup

**Requirements**: Python ≥ 3.10, Linux (Ubuntu 22.04+ recommended), CUDA 12.x (optional; CPU works but slower).

```bash
git clone https://github.com/JazerJu/glm-ocr-onnx.git
cd glm-ocr-onnx

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download Official Models

This project needs two official Hugging Face downloads:

- [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR): official safetensors, tokenizer, and config files for exporting the visual ONNX path and preparing runtime config
- [`ggml-org/GLM-OCR-GGUF`](https://huggingface.co/ggml-org/GLM-OCR-GGUF): official GGUF decoder for llama.cpp

Recommended layout:

```bash
mkdir -p models

hf download zai-org/GLM-OCR \
  --local-dir models/GLM-OCR

hf download ggml-org/GLM-OCR-GGUF \
  --include "GLM-OCR-Q8_0.gguf" \
  --local-dir models/GLM-OCR-GGUF
```

After download:

```text
models/
├── GLM-OCR/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── GLM-OCR-GGUF/
    └── GLM-OCR-Q8_0.gguf
```

## 3. Export ONNX

The export script is `01-Export-ONNX.py` in the repository root. It reads the official safetensors from `models/GLM-OCR` and writes ONNX plus runtime config into `models/export/`:

```bash
python3 01-Export-ONNX.py \
  --model-dir models/GLM-OCR \
  --output-dir models/export
```

After export:

```text
models/export/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── processor_config.json
├── preprocessor_config.json
├── generation_config.json
└── onnx/
    ├── vision_encoder_fp32.onnx
    ├── embed_tokens_fp32.onnx
    └── merger_fp16.onnx
```

`processor_config.json` is a compatibility config used by the current ONNX runtime. If the official directory does not provide it, the export script creates one from `preprocessor_config.json`.

After export, run ONNX Runtime graph optimization. It overwrites the ONNX files under `models/export/onnx/` in place:

```bash
python3 02-Optimize-ONNX.py --onnx-dir models/export/onnx
```

Then generate the Q4 vision encoder and Q4 token embedding:

```bash
python3 03-Quantize-ONNX.py --onnx-dir models/export/onnx
```

After that, the runtime directory contains:

```text
models/export/onnx/
├── vision_encoder_q4.onnx
├── vision_encoder_q4.onnx.data
├── embed_tokens_q4.onnx
├── embed_tokens_q4.onnx.data
└── merger_fp16.onnx
```

## 4. Get llama.cpp Runtime Libraries

This project calls llama.cpp's C API directly via ctypes. Place the `.so` files into `bin/`.

### Option 1: Download Pre-built Package (Recommended)

Download the tarball for your backend from [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases/latest):

| Backend | Download | Size |
|---------|----------|------|
| Vulkan (NVIDIA / AMD / Intel) | `llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz` | ~32 MB |
| CPU only | `llama-bXXXX-bin-ubuntu-x64.tar.gz` | ~15 MB |
| ROCm (AMD) | `llama-bXXXX-bin-ubuntu-rocm-x64.tar.gz` | ~128 MB |

> `bXXXX` is the build number; pick the latest version.

Extract and copy the `.so*` files into `bin/`:

```bash
tar xzf llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz
cp -a llama-bXXXX-bin-ubuntu-vulkan-x64/lib*.so* bin/
```

`libvulkan.so` is a system package: `apt install libvulkan1` on Ubuntu / Debian.

### Option 2: Build from Source

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp

# Vulkan (recommended, works on NVIDIA / AMD / Intel)
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build --config Release -j"$(nproc)"

# CPU only
# cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release

# CUDA
# cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

cp -a ../llama.cpp/build/bin/lib*.so* bin/
```

### Final `bin/` Directory

```text
bin/
├── libllama.so*
├── libggml.so*
├── libggml-base.so*
├── libggml-cpu.so*
└── libggml-vulkan.so*   # or libggml-cuda.so* (depending on backend)
```

Vulkan backend is recommended for NVIDIA users for best compatibility.

## 5. Runtime Entry

ONNX visual path + llama.cpp GGUF decoder:

```bash
python3 runtime/glm_ocr_llama.py \
  --image tests/example.png \
  --onnx-dir models/export \
  --gguf models/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf \
  --prompt "请识别图中的所有文字" \
  --max-tokens 512
```

Python usage:

```python
from PIL import Image
import sys
sys.path.insert(0, "runtime")

from glm_ocr_llama import GlmOcrLlama

engine = GlmOcrLlama(
    gguf_path="models/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf",
    onnx_dir="models/export",
)

image = Image.open("tests/example.png").convert("RGB")
print(engine.ocr(image, prompt="请识别图中的所有文字"))
```

## 6. Test

This repository includes `tests/example.png`. After preparing ONNX files and GGUF, run the full test:

```bash
python3 runtime/glm_ocr_llama.py \
  --image tests/example.png \
  --onnx-dir models/export \
  --gguf models/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf \
  --prompt "请识别图中的所有文字" \
  --max-tokens 512
```

> `tests/test_glm_ocr.py` is an experimental pure-ONNX test (without llama.cpp) that requires an additional decoder ONNX export. Not needed for normal usage.

## 7. How It Works

GLM-OCR inference is split into two parts:

```text
image
  -> image preprocess
  -> ONNX vision encoder / merger
  -> visual embeddings
  -> llama.cpp loads official GGUF decoder
  -> OCR text
```

The decoder is not exported to ONNX. It stays in the official GGUF format and runs through llama.cpp; ONNX is used for the visual side and embedding preparation.

This keeps deployment simple:

- reuse the official GGUF decoder
- optimize or quantize the visual ONNX path separately
- call llama.cpp directly from Python without starting `llama-server`

## 8. Local-Only Files

These paths are git-ignored:

- `models/` — original models and export artifacts
- `*.gguf` — GGUF decoder
- `bin/*.so*` — runtime libraries, user-provided

They are generated or downloaded artifacts and should not be committed to source control.

## License

[MIT](LICENSE)
