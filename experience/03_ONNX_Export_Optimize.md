# ONNX Export and Optimization for GLM-OCR

本文记录 GLM-OCR-GGUF 项目的第二条工程主线：**从官方 safetensors 导出可复现 ONNX，并完成 baseline 优化**。

重点不是"模型有哪些参数"，而是导出和优化中实际踩过的坑：文件命名误导、vision 导出方式、external data 旧尾巴。

Q4 量化的坑和方案在下一篇：**[04: ONNX Q4 Quantization](04_ONNX_Q4_Quantization.md)**。

参考记录：

- opencode `ses_178f9f5fdffeoFXwm1124bSeyw`: Implement GLM-OCR ONNX inference pipeline
- opencode `ses_1790940f6ffeLyDfNT2Zk7wrwA`: Explore GLM-OCR ONNX model structure
- opencode `ses_1796eededffezn1I53DOJXDC0G`: 优化视频理解项目-glmocr-纯onnx推理
- 当前脚本：`01-Export-ONNX.py`、`02-Optimize-ONNX.py`

---

## 1. 起点：不要依赖社区 ONNX 包的旧命名

早期参考过 `onnx-community/GLM-OCR-ONNX` 一类包，但里面有一个非常容易误导的历史问题：文件名和实际职责可能反过来。

当时记录中确认过：

```text
decoder_model_merged_q4.onnx = vision encoder
vision_encoder_q4.onnx      = LLM decoder
```

这不是适合长期维护的命名方式。对于一个要开源给别人复现的项目，文件名必须直接表达职责。

因此当前仓库改为从官方 `zai-org/GLM-OCR` safetensors 自己导出：

```text
vision_encoder_fp32.onnx
embed_tokens_fp32.onnx
merger_fp16.onnx
```

再从这些 baseline 派生：

```text
vision_encoder_q4.onnx
embed_tokens_q4.onnx
```

这样 README、runtime、download manifest 都不需要解释“这个文件名其实不是它的意思”。

---

## 2. 导出拆分：只导视觉侧，不导 decoder

当前导出脚本只导出三块：

| 文件 | 职责 |
|---|---|
| `vision_encoder_fp32.onnx` | pixel patches + position_ids -> visual hidden states |
| `merger_fp16.onnx` | visual hidden states -> LLM-space image embeddings |
| `embed_tokens_fp32.onnx` | text token ids -> text embeddings |

decoder 不导出 ONNX，而是交给官方 GGUF：

```text
GLM-OCR-Q8_0.gguf
```

原因不是“不能导”，而是这条混合路线更适合本地集成：

1. decoder ONNX with KV cache 会把运行时输入输出复杂度拉高。
2. llama.cpp 已经稳定支持 GLM-OCR decoder。
3. ONNX 的收益主要在视觉侧固定前向和 embedding 表，不在自回归 decoder。
4. 视觉侧 ONNX 可以单独 Q4，decoder 继续使用成熟 GGUF 量化。

---

## 3. 导出原理：子图拆分 + torch.onnx.export

导出不是简单的 key 匹配，而是**按功能拆出 PyTorch 子图，逐个 tracing 导出**。

完整模型 `GlmOcrForConditionalGeneration` 的结构：

```text
GlmOcrForConditionalGeneration
├── model.visual                  ← 视觉侧（导出）
│   ├── patch_embed               ┐
│   ├── rotary_pos_emb            │ vision_encoder_fp32.onnx
│   ├── blocks (transformer layers)│
│   ├── post_layernorm            │
│   └── downsample                ┘
│   └── merger                    ← merger_fp16.onnx
├── model.embed_tokens            ← embed_tokens_fp32.onnx
└── model.layers + lm_head        ← 不导出，用 GGUF decoder
```

### 3.1 embed_tokens：最简单的包一层

```python
class EmbedModule(nn.Module):
    def forward(self, input_ids):
        return self.emb(input_ids)
```

一个 Embedding lookup，`dynamo=False`（TorchScript trace）就够了。输入 `[batch, seq]`，输出 `[batch, seq, hidden]`。

### 3.2 vision_encoder：手动展开 forward 跳过 merger

```python
class VisionNoMerger(nn.Module):
    def forward(self, pixel_values, position_ids):
        hidden = self.visual.patch_embed(pixel_values)
        rotary_emb = self.visual.rotary_pos_emb(position_ids)
        emb = cat((rotary_emb, rotary_emb), dim=-1)  # cos/sin pair
        for block in self.visual.blocks:
            hidden = block(hidden, ..., position_embeddings)
        hidden = post_layernorm(hidden)
        return downsample(hidden)
```

- 输入：`pixel_values [num_patches, patch_dim]` + `position_ids [num_tokens, 2]`（2D 的 hpos/wpos，不是 4D）
- 输出：`image_features [num_tokens, out_hidden_size]`
- 权重从完整 safetensors 加载后"切"出，不是按 key 重新匹配

### 3.3 merger：先转 fp16 再导出

```python
MergerModule(visual.merger).half().eval()
```

merger 计算量小，直接 fp16 导出省空间。输入是 vision encoder 的输出，输出投影到 LLM embedding 空间。

### 3.4 `dynamo` 参数与导出的实质

`torch.onnx.export` 的 `dynamo` 参数决定了 trace 方式，但**导出的实质是一样的**：从 PyTorch nn.Module 调用一路展开到 ONNX 原子算子（MatMul、Add、Softmax、LayerNorm 等）。导出的 `.onnx` 文件里没有 `patch_embed`、`blocks`、`downsample` 这些模块概念，只有算子。

```text
PyTorch 侧                        ONNX 侧
─────────────                      ─────────
patch_embed()         →            Conv2D + Reshape
rotary_pos_emb()      →            MatMul + 三角函数
block(hidden, ...)    →            MatMul + Softmax + Add + LayerNorm (×N 层)
post_layernorm()      →            LayerNorm
downsample()          →            Conv2D + Reshape
```

两个 trace 后端的区别：

| | `dynamo=False` (TorchScript) | `dynamo=True` (PyTorch 2.x Dynamo) |
|---|---|---|
| 原理 | JIT trace，记录算子调用 | Python 字节码级编译，再转 ONNX |
| 动态 shape | **会把 dummy 输入的 shape 固化进图** | 正确处理动态轴 |
| 循环 | 展开为固定次数 | 保留为动态循环 |
| 适用 | 结构简单的模块 | 有循环或需要动态 shape 的模块 |

当前导出各模块的选择：

| 模块 | dynamo | 原因 |
|---|---|---|
| embed_tokens | `False` | 纯 Embedding lookup，无动态 shape 问题 |
| vision_encoder | `True` | 有 `for block in blocks` 循环，且必须支持任意 patch 数 |
| merger | `False` | 简单线性层，无动态 shape 问题 |

**vision_encoder 用 `dynamo=False` 会坠机**：dummy 输入的 patch 数会被固化进图，真实图片尺寸不同时直接报错或产出错误结果。

---

## 4. 导出阶段的真实坑

### 4.1 vision 不能随便用 legacy exporter

`embed_tokens` 和 `merger` 结构简单，用：

```python
dynamo=False
```

稳定。

但 `vision_encoder` 不能照搬。旧 exporter 容易把 dummy 输入的长度、切片或 reshape 细节固化进图里，真实图片换尺寸后会出问题。

当前脚本对 vision 使用：

```python
dynamo=True
```

并设置动态轴：

```python
dynamic_axes={
    "pixel_values": {0: "num_patches"},
    "position_ids": {0: "num_tokens"},
    "image_features": {0: "num_tokens"},
}
```

这是为了让导出的 vision 图接受真实图片产生的 patch 数，而不是只服务 dummy 输入。

### 4.2 attention 路径要固定为 eager

导出时加载 HF 模型：

```python
GlmOcrForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
```

这里使用 `attn_implementation="eager"` 是为了避免 export 过程进入更复杂的 attention kernel 路径。导出脚本不是性能运行时，优先目标是稳定、可追踪、可复现。

### 4.3 `chat_template.jinja` 不是可有可无

runtime 构造 prompt 时需要 chat template。

有些 tokenizer 配置里不内嵌 `chat_template`，所以导出时必须复制：

```text
chat_template.jinja
```

否则 runtime 只能靠手写 prompt，容易和官方模板漂移。

当前 runtime 的策略是：

1. 先读 `tokenizer_config.json` 里的 `chat_template`。
2. 如果没有，再读导出目录下的 `chat_template.jinja`。
3. 两者都没有才报错。

---

## 5. Optimize 原理：onnxruntime 内置图优化

`02-Optimize-ONNX.py` 没有自定义优化逻辑，核心就是调 `onnxruntime.transformers.optimizer`：

```python
optimizer = optimize_model(
    str(input_path),
    model_type="bert",    # 当作 BERT 类模型优化
    num_heads=0,           # 自动推断
    hidden_size=0,         # 自动推断
    opt_level=2,           # 默认 level 2
    use_gpu=True,          # 保留 GPU 节点
)
```

###  `model_type="bert"` 能用的原因

GLM-OCR 的 vision encoder 本质是 **ViT 架构**（patch embed → transformer blocks → layernorm），和 BERT 的计算图结构一致：
- Multi-Head Attention → 融合成单个 `MultiHeadAttention` 节点
- LayerNorm → 融合成单个 `LayerNormalization` 节点
- Skip connections → 识别残差模式并优化

### `opt_level=2` 的参数含义

| Level | 名称 | 做什么 |
|---|---|---|
| 0 | Basic | 常量折叠、死代码消除 |
| 1 | Extended | 算子融合（Conv+BN、MatMul+Add） |
| **2** | **Full** | Level 0+1 + 注意力融合、嵌套子图优化、节点重排 |

Level 2 对 ViT 的主要效果：Q/K/V MatMul + Softmax → 单个 `MultiHeadAttention`；冗余 Reshape/Transpose 消除；常量直接嵌入图。

### `num_heads=0, hidden_size=0`

传 0 不是"不需要"，是让 optimizer 扫描 ONNX graph 的维度信息自动推断。

### 保存方式

```python
onnx.save_model(model, output_path,
    save_as_external_data=True,        # 权重存外部 .data 文件
    all_tensors_to_one_file=True,      # 所有权重打包成一个 .data
    size_threshold=1024 * 1024,        # >1MB 的 tensor 外置
)
```

大模型权重拆到 `.onnx.data`，graph 本身保持精简可读。

---

## 6. Optimize 的坑：保存前必须清理 external data

ONNX 大权重会写到外部文件：

```text
*.onnx.data
```

第一次保存没问题，真正的坑发生在覆盖保存：

```text
旧 .onnx.data 比新权重大
onnx.save_model 覆盖时没有自动截断旧尾巴
du -sh 看到的文件大小明显偏大
```

这会直接造成误判：你以为 Q4 变大了，其实是旧 external data 残留。

当前 `02-Optimize-ONNX.py` 和 `03-Quantize-ONNX.py` 都采用同一策略：

```python
data_path = output_path.with_name(output_path.name + ".data")
if data_path.exists():
    data_path.unlink()
onnx.save_model(...)
```

同时会清理历史命名：

```text
*.onnx_data
*.data
*.onnx_data
```

经验：**判断 ONNX 文件是否干净，不能只看 `du -sh`，还要看 initializer 的 external data 引用。**

---

## 7. 复现流程（导出 + 优化）

从官方 safetensors 导出：

```bash
python3 01-Export-ONNX.py \
  --model-dir models/GLM-OCR \
  --output-dir models/export
```

优化 baseline：

```bash
python3 02-Optimize-ONNX.py \
  --onnx-dir models/export/onnx
```

导出 + 优化完成后，`models/export/onnx/` 应包含：

```text
vision_encoder_fp32.onnx
embed_tokens_fp32.onnx
merger_fp16.onnx
```

Q4 量化步骤见 **[04: ONNX Q4 Quantization](04_ONNX_Q4_Quantization.md)**。

---

## 8. 经验结论

GLM-OCR 的 ONNX 导出工作不难在"能导出"，难在三个细节：

1. **文件职责要重新命名清楚**：不要沿用历史上容易反转含义的社区 ONNX 文件名。
2. **vision 导出要支持真实动态 patch 数**：不能让 dummy 输入形状污染图。
3. **optimize 后必须清理旧 external data**：`du -sh` 不能代替 initializer 检查。

只要这三点处理好，GLM-OCR 的 ONNX 视觉侧可以稳定形成可复现的 baseline。Q4 量化在这个 baseline 之上进行。
