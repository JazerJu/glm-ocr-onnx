<p align="center">
  中文版 | <a href="README.en.md">English</a>
</p>

# GLM-OCR-ONNX

这个项目专注于一件事：让 GLM-OCR 在本地用 **ONNX Runtime + llama.cpp GGUF** 跑起来。

GLM-OCR 已经有官方 GGUF decoder，所以这里不重新转换语言模型。仓库只负责视觉侧 ONNX 推理、图像 token 拼接、mRoPE 位置编码，以及把视觉 embedding 交给 llama.cpp 继续生成文字。

适合的场景：

- 本地 OCR
- 截图 / 文档图片识别
- 给自己的 AI agent 增加一个轻量 OCR 工具
- 不想启动完整 Transformers/PyTorch 大模型，只想复用llama.cpp

## 1. 准备环境

**环境要求**：Python ≥ 3.10、Linux（推荐 Ubuntu 22.04+）、CUDA 12.x（可选，CPU 也能跑但较慢）。

```bash
git clone https://github.com/JazerJu/glm-ocr-onnx.git
cd glm-ocr-onnx

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 下载官方模型

这个项目需要两份官方 Hugging Face 文件：

- [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR)：官方 safetensors、tokenizer、config，用于导出视觉侧 ONNX，也提供运行所需配置
- [`ggml-org/GLM-OCR-GGUF`](https://huggingface.co/ggml-org/GLM-OCR-GGUF)：官方 GGUF decoder，用于 llama.cpp 推理

建议路径：

```bash
mkdir -p models

hf download zai-org/GLM-OCR \
  --local-dir models/GLM-OCR

hf download ggml-org/GLM-OCR-GGUF \
  --include "GLM-OCR-Q8_0.gguf" \
  --local-dir models/GLM-OCR-GGUF
```

下载后目录类似：

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

## 3. 导出 ONNX

导出脚本是仓库根目录下的 `01-Export-ONNX.py`。它从 `models/GLM-OCR` 读取官方 safetensors，并把 ONNX 和运行配置写到 `models/export/`：

```bash
python3 01-Export-ONNX.py \
  --model-dir models/GLM-OCR \
  --output-dir models/export
```

导出后目录类似：

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

`processor_config.json` 是当前 ONNX runtime 使用的兼容配置；如果官方目录里没有，导出脚本会根据 `preprocessor_config.json` 生成一份。

导出完成后，接着跑 ONNX Runtime 图优化。它会直接覆盖 `models/export/onnx/` 下的同名 ONNX 文件：

```bash
python3 02-Optimize-ONNX.py --onnx-dir models/export/onnx
```

然后生成 Q4 视觉 encoder 和 Q4 token embedding：

```bash
python3 03-Quantize-ONNX.py --onnx-dir models/export/onnx
```

完成后，运行目录会包含：

```text
models/export/onnx/
├── vision_encoder_q4.onnx
├── vision_encoder_q4.onnx.data
├── embed_tokens_q4.onnx
├── embed_tokens_q4.onnx.data
└── merger_fp16.onnx
```

## 4. 获取 llama.cpp 运行库

本项目通过 ctypes 直接调用 llama.cpp 的 C API，需要把 `.so` 文件放入 `bin/`。

### 方式一：下载预编译包（推荐）

从 [llama.cpp Release](https://github.com/ggml-org/llama.cpp/releases/latest) 下载对应后端的 tarball：

| 后端 | 下载文件 | 大小 |
|------|---------|------|
| Vulkan（NVIDIA / AMD / Intel） | `llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz` | ~32 MB |
| CPU only | `llama-bXXXX-bin-ubuntu-x64.tar.gz` | ~15 MB |
| ROCm (AMD) | `llama-bXXXX-bin-ubuntu-rocm-x64.tar.gz` | ~128 MB |

> `bXXXX` 是 build 号，下载时选最新版本对应的文件名即可。

解压后把 `.so*` 复制到 `bin/`：

```bash
tar xzf llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz
cp -a llama-bXXXX-bin-ubuntu-vulkan-x64/lib*.so* bin/
```

`libvulkan.so` 是系统包，Ubuntu / Debian 用 `apt install libvulkan1` 安装。

### 方式二：从源码编译

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp

# Vulkan（推荐，NVIDIA / AMD / Intel 通用）
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build --config Release -j"$(nproc)"

# CPU only
# cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release

# CUDA
# cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

cp -a ../llama.cpp/build/bin/lib*.so* bin/
```

### 最终 `bin/` 目录

```text
bin/
├── libllama.so*
├── libggml.so*
├── libggml-base.so*
├── libggml-cpu.so*
└── libggml-vulkan.so*   # 或 libggml-cuda.so*（取决于编译后端）
```

NVIDIA 用户推荐 Vulkan 后端，兼容性最好。

## 5. 交互入口

ONNX 视觉侧 + llama.cpp GGUF decoder：

```bash
python3 runtime/glm_ocr_llama.py \
  --image tests/example.png \
  --onnx-dir models/export \
  --gguf models/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf \
  --prompt "请识别图中的所有文字" \
  --max-tokens 512
```

Python 中调用：

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

仓库提供 `tests/example.png`。准备好 ONNX 文件和 GGUF 后，运行完整测试：

```bash
python3 runtime/glm_ocr_llama.py \
  --image tests/example.png \
  --onnx-dir models/export \
  --gguf models/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf \
  --prompt "请识别图中的所有文字" \
  --max-tokens 512
```

> `tests/test_glm_ocr.py` 是纯 ONNX（不含 llama.cpp）的实验性测试，需要额外导出 decoder ONNX 才能运行，普通使用不需要。

## 7. 基本原理

GLM-OCR 的推理可以拆成两段：

```text
image
  -> image preprocess
  -> ONNX vision encoder / merger
  -> visual embeddings
  -> llama.cpp loads official GGUF decoder
  -> OCR text
```

也就是说，这个项目没有把 decoder 导出成 ONNX。decoder 仍然由 llama.cpp 运行官方 GGUF；ONNX 只负责视觉侧和 embedding 准备。

这样做的好处是：

- decoder 直接复用官方 GGUF，部署简单
- ONNX 视觉侧可以单独优化、量化和替换
- Python 只做预处理和调度，不需要启动 `llama-server`

## 8. 不进入仓库的文件

以下文件默认被 `.gitignore` 排除：

- `models/` — 原始模型和导出文件
- `*.gguf` — GGUF decoder
- `bin/*.so*` — 运行库，用户自行获取

这些都是本地生成或下载的大文件，不适合放进源码仓库。

## License

[MIT](LICENSE)
