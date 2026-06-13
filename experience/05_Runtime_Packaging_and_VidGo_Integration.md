# Runtime Packaging and VidGo Integration for GLM-OCR

本文记录 GLM-OCR-GGUF 从“单机脚本能跑”到“可以被 VidGo / 视频理解系统稳定调用”的整理过程。

前四篇分别解决了：

1. 为什么不用 llama-server 作为最终形态。
2. ctypes 注入 image embeddings 时 mRoPE 为什么会乱码。
3. 如何从官方 safetensors 导出并优化 ONNX baseline。
4. 如何从 FP32 baseline 得到干净的 Q4 runtime 文件。

这一篇只讲最后一公里：**文件怎么放、runtime 怎么找、下载清单怎么写、怎么验证用户机器上真的能跑**。

参考文件：

- `runtime/glm_ocr_llama.py`
- `runtime/glm_ocr_onnx.py`
- `runtime/llama_cpp_bindings.py`
- `runtime/llama_cpp_bindings.py`
- VidGo 集成路径（参考）：`<VIDGO_ROOT>/backend/vid_under/glm_ocr_llama.py`
- VidGo 集成路径（参考）：`<VIDGO_ROOT>/backend/vid_under/views_download.py`

---

## 1. 起点：能跑不等于能发布

最早验证时，命令里可以写绝对路径：

```bash
python3 runtime/glm_ocr_llama.py \
  --image tests/example.png \
  --onnx-dir models/export \
  --gguf /path/to/GLM-OCR-Q8_0.gguf
```

开发中没问题，但不能直接发布。因为用户侧会遇到四类问题：

| 问题 | 现象 |
|---|---|
| 模型目录不固定 | runtime 找不到 `config.json` 或 ONNX 文件 |
| ONNX external data 命名不一致 | `.onnx_data`、`_data`、`.onnx.data` 混用 |
| chat template 缺失 | prompt 构造失败或和官方模板不一致 |
| llama.cpp 动态库路径不一致 | `libllama.so` / `libggml.so` 加载失败 |

所以发布阶段要做的是，把运行边界拢成一个稳定目录，规定好协议。

---

## 2. 运行目录协议

最终 GLM-OCR runtime 只认一个 `onnx_dir`，它应该长这样：

```text
glm-ocr-onnx/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── processor_config.json
├── preprocessor_config.json
├── generation_config.json
├── chat_template.jinja
└── onnx/
    ├── embed_tokens_q4.onnx
    ├── embed_tokens_q4.onnx.data
    ├── vision_encoder_q4.onnx
    ├── vision_encoder_q4.onnx.data
    ├── merger_fp16.onnx
    └── merger_fp16.onnx.data
```

GGUF 单独传入：

```text
GLM-OCR-Q8_0.gguf
```

这样 runtime 初始化只需要两个显式参数：

```python
GlmOcrLlama(
    gguf_path="/path/to/GLM-OCR-Q8_0.gguf",
    onnx_dir="/path/to/glm-ocr-onnx",
)
```

不要让 runtime 在多个历史目录里猜文件。猜路径会让调试成本翻倍。

---

## 3. 为什么 `.onnx.data` 后缀要统一

ONNX external data 最容易产生“文件存在但加载失败”的问题。

早期可能出现过这些命名：

```text
vision_encoder_q4.onnx_data
vision_encoder_q4.onnx.data
vision_encoder_q4.data
```

但当前脚本保存时统一使用：

```text
<model>.onnx.data
```

例如：

```text
vision_encoder_q4.onnx
vision_encoder_q4.onnx.data
```

这点要同步到下载清单。否则用户下载了 `vision_encoder_q4.onnx`，但 external data 文件名不符合 ONNX 里记录的 `location`，加载时会报模型文件损坏或 initializer 缺失。

**下载清单必须和 ONNX 内部 external data location 完全一致，不能靠 runtime 自动修正。**

---

## 4. chat_template 是运行文件，不是文档(document)

GLM-OCR 的 prompt 构造依赖 chat template。不能简单手写：

```text
<image> 请识别图中的文字
```

当前 runtime 的逻辑是：

1. 读 `tokenizer_config.json`。
2. 如果里面有 `chat_template`，直接用。
3. 如果没有，读 `chat_template.jinja`。
4. 如果两个都没有，直接报错。

所以 `chat_template.jinja` 必须进入最小运行集。它不是 README，也不是示例文件，而是 prompt 构造的一部分。

---

## 5. llama.cpp 动态库边界

GLM-OCR runtime 需要两层 llama.cpp 动态库：

```text
libllama.so
libggml*.so
```


发布时要明确：

1. `libllama.so` 来自用户编译好的 llama.cpp。
2. `libggml.so` 来自


---

## 6. VidGo 接入时的路径约定

VidGo 侧不应该复制一份独立逻辑，而是按配置传路径。

推荐结构：

```text
VIDUNDER_MODEL_ROOT/
└── glm-ocr/
    ├── glm-ocr-onnx/
    │   ├── config.json
    │   └── onnx/...
    └── GLM-OCR-Q8_0.gguf
```

配置中只暴露：

```python
GLM_OCR_ONNX_DIR = MODEL_ROOT / "glm-ocr" / "glm-ocr-onnx"
GLM_OCR_GGUF_PATH = MODEL_ROOT / "glm-ocr" / "GLM-OCR-Q8_0.gguf"
```

调用时：

```python
ocr = GlmOcrLlama(
    gguf_path=GLM_OCR_GGUF_PATH,
    onnx_dir=GLM_OCR_ONNX_DIR,
    n_ctx=N_CTX,
)
```

这样下载模块、CLI 测试、Web 后端用的是同一套文件位置。

---

## 7. 下载清单的验收方式

下载清单不能只写 URL。每个文件至少需要：

```text
relative path
size
source url
```

GLM-OCR 这种 ONNX external data 模型尤其需要 size，因为 stale data 或旧命名很容易混进去。

当前 Q4 运行文件的参考大小：

```text
embed_tokens_q4.onnx        352K
embed_tokens_q4.onnx.data   47M
vision_encoder_q4.onnx      28M
vision_encoder_q4.onnx.data 209M
merger_fp16.onnx            12K
merger_fp16.onnx.data       45M
```

如果用户下载后文件大小明显不对，先不要跑 OCR，先修下载文件。

---

## 8. 验证命令必须只依赖最小运行集

发布前的 smoke test 应该模拟用户环境，而不是开发环境。

推荐命令：

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH \
  python3 runtime/glm_ocr_llama.py \
    --image tests/example.png \
    --onnx-dir models/export \
    --gguf models/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf \
    --prompt "请识别图中的所有文字" \
    --max-tokens 64
```

这里故意清掉：

```text
LD_LIBRARY_PATH
PYTHONPATH
```

目的是防止误用开发目录里的旧包、旧 so、旧 Python 模块。

验收输出应该包含测试图文字：

```text
GLM OCR TEST
Hello 2026
```

---

## 9. context reuse 与 batch 的真实收益

当时曾经把 `LlamaContext` 从每次 `ocr()` 内部创建，改成在 `GlmOcrLlama.__init__()` 中创建一次，然后每张图之间调用：

```python
self.ctx.clear_kv()
```

并增加：

```python
ocr_batch([img1, img2, ...])
```

这一步的初衷是减少重复初始化和为批量 OCR 提供接口。实测小图结果大致是：

| 模式 | 耗时/次 |
|---|---:|
| 复用 ctx | 105ms |
| `ocr_batch` 顺序处理 | 101ms |
| 每次 new ctx | 114ms |

结论要写清楚：这个 `ocr_batch` **不是 continuous batching**，也不是把多张图的 decoder 真正合成一个 GPU batch。它只是顺序处理多张图，同时复用同一个 context。

它的收益是：

1. 少做 context / KV-cache 反复创建和释放。
2. 降低 GPU 内存抖动。
3. 给 VidGo 上层一个批量接口，避免调用方自己管理 context 生命周期。

它不能解决的问题是：

1. 多张图的 LLM decode 仍然是串行。
2. llama.cpp 当前这条 ctypes 路线没有 vLLM 那种 continuous batching。
3. GLM-OCR 的 mRoPE 多 seq 并行 decode 还没有在这套 binding 里验证。

如果目标是多图吞吐，真实可行方向是：

| 方向 | 代价 | 预期收益 |
|---|---|---|
| ONNX vision encoder batch | 需要处理不同分辨率 padding / grid_thw | 只加速视觉侧，收益有限 |
| 多进程并行 | 多份 context，占更多显存 | 单卡也能提高吞吐，适合多帧 OCR |
| vLLM / SGLang 化 | 工程量大，可能放弃 ONNX 视觉侧控制 | continuous batching / paged KV 才是根本吞吐优化 |

所以这一步应该被描述为**运行时稳定性和接口整理**，不是大幅性能优化。

---

## 10. 不要把 FP32 baseline 当成 runtime

导出目录里可能有：

```text
vision_encoder_fp32.onnx
embed_tokens_fp32.onnx
merger_fp32.onnx
```

这些是调试和重新量化用的。普通运行时不要依赖它们。

原因：

1. 文件体积大。
2. 下载慢。
3. q4的精度已经足够了。
4. 用户很难判断到底 runtime 用的是 q4 还是 fp32。

发布包应该明确：

```text
q4 vision
q4 embed
fp16 merger
q8 gguf decoder
```

如果要保留 FP32，也应该放在 export/debug 说明里，而不是 runtime 默认路径。

---

## 11. 经验结论

GLM-OCR 的发布整理不是“把能跑的文件打包一下”。真正要固定的是四个边界：

1. **目录边界**：ONNX 目录只认一个根路径。
2. **文件边界**：external data 后缀和 ONNX 内部 location 必须一致。
3. **ABI 边界**：`libllama.so` / `libggml.so` 必须和当前 llama.cpp 版本匹配。
4. **验证边界**：smoke test 必须只依赖最小运行集，而不是开发缓存。

这四个边界固定以后，GLM-OCR 才能从“我的机器能跑”变成“别人下载后也能跑”。
