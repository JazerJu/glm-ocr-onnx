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

huggingface-cli download zai-org/GLM-OCR \
  --local-dir models/GLM-OCR

huggingface-cli download ggml-org/GLM-OCR-GGUF \
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

## 4. Prepare llama.cpp Shared Libraries

`runtime/llama_cpp_bindings.py` loads llama.cpp shared libraries from this repository's `bin/` directory. The source tree only keeps `bin/llama_wrap.c`; the following `.so` files are local build artifacts and are not committed to git:

```text
bin/libggml.so              # llama.cpp
bin/libggml-base.so         # llama.cpp dependency
bin/libggml-cpu.so          # llama.cpp dependency
bin/libggml-cuda.so         # CUDA backend, required for GPU inference
bin/libllama.so             # llama.cpp
bin/libllama_wrap.so        # ctypes wrapper from this repository
```

Build llama.cpp first:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build \
  -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build -j
```

Place the llama.cpp runtime libraries under this repository's `bin/`. For local development, symlinks are fine; for release bundles, use `cp -a` to preserve the symlink chain and versioned files:

```bash
mkdir -p bin

# Development: symlink to the local llama.cpp build
ln -sf "$(realpath ../llama.cpp/build/bin/libggml.so)" bin/libggml.so
ln -sf "$(realpath ../llama.cpp/build/bin/libllama.so)" bin/libllama.so

# Release bundle: copy the complete dependency chain
cp -a ../llama.cpp/build/bin/libggml*.so* bin/
cp -a ../llama.cpp/build/bin/libllama.so* bin/
```

Then build this repository's wrapper:

```bash
gcc -shared -fPIC -o bin/libllama_wrap.so bin/llama_wrap.c \
  -I../llama.cpp/include -I../llama.cpp/ggml/include \
  -Lbin -lllama -lggml \
  -Wl,-rpath,'$ORIGIN'
```

`$ORIGIN` tells the dynamic loader to resolve `libllama.so` / `libggml.so` from the same `bin/` directory as `libllama_wrap.so`, so users do not need to set `LD_LIBRARY_PATH` manually.

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

- `models/export/`
- `models/`
- `*.gguf`
- `bin/*.so`

They are generated or downloaded artifacts and should not be committed to source control.

## License

[MIT](LICENSE)
