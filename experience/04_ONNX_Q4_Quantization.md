# ONNX Q4 Quantization for GLM-OCR

本文记录 GLM-OCR-GGUF 项目从 FP32 baseline ONNX 得到 Q4 runtime 文件的过程。

前置条件：**[03: ONNX Export and Optimization](03_ONNX_Export_Optimize.md)** 已完成，`models/export/onnx/` 里有干净的 FP32 ONNX。

随着大模型的发展，ONNX也开始支持Q4作为模型量化格式，因为Q4的信息损失远小于int4。 GLM-OCR vision 图量化时踩过的坑：普通动态量化不适合 fused QKV、Gemm 必须先转 MatMul+Add。

参考记录：

- opencode `ses_178f9f5fdffeoFXwm1124bSeyw`: Implement GLM-OCR ONNX inference pipeline
- opencode `ses_1790940f6ffeLyDfNT2Zk7wrwA`: Explore GLM-OCR ONNX model structure
- opencode `ses_1796eededffezn1I53DOJXDC0G`: 优化视频理解项目-glmocr-纯onnx推理
- 当前脚本：`03-Quantize-ONNX.py`

---

## 1. `quantize_dynamic` 可以跑 vision，但走不了 Q4

`quantize_dynamic`（INT8 动态量化）可以直接处理 `Gemm(transB=1)` 的 fused QKV，不需要 rewrite。

当前版本的 ONNX Runtime 测试验证：

```text
quantize_dynamic(model, weight_type=QInt8)  → 成功，~395 MB
quantize_dynamic(model, weight_type=QInt4)   → 失败："must be 8-bit before packing as 4-bit values"
```

所以 **fused QKV 的 Gemm 格式本身不是问题**，早期遇到过 `1024 vs 3072` shape inference 报错，即下一层需要长度为1024的向量，但上一层的结果维度为(3072,1024)，可能是特定版本或配置下触发的。

但 Q4 路线（`MatMulNBitsQuantizer`）只处理 `MatMul` 节点，不认 `Gemm`，这是onnx的支持问题，所以仍需 rewrite：

| | `quantize_dynamic` INT8 | `MatMulNBits` Q4 |
|---|---|---|
| 精度 | 8 bit/weight | 4 bit/weight |
| 大小 | ~395 MB（全内联） | ~236 MB（graph + data） |
| Gemm 兼容 | ✅ 直接处理 | ❌ 需先 rewrite 成 MatMul+Add |
| Q4 支持 | ❌ 不支持 | ✅ 原生 |

当前项目选 Q4 是因为视觉侧权重大，INT8 只压缩 4x（395 MB vs 原始 1.6 GB），Q4 压缩 ~7x（236 MB）。rewrite `Gemm→MatMul+Add` 是 Q4 路线的前置步骤，不是 vision 图本身的限制。

---

## 2. 为什么要先 Gemm -> MatMul + Add

`MatMulNBitsQuantizer` 主要处理 constant-weight `MatMul`，但 PyTorch 导出的 vision linear 经常是：

```text
Gemm(A, W, B, transB=1)
```

所以量化前必须把它改写成：

```text
MatMul(A, W.T) + Add(B)
```

当前 `03-Quantize-ONNX.py` 会：

1. 加载 `vision_encoder_fp32.onnx`
2. 找到支持的 `Gemm`
3. 把权重转置成新的 initializer
4. 写成 `MatMul + Add`
5. 再运行 `MatMulNBitsQuantizer`

这一步不是多余的格式整理，而是 Q4 路线能跑通的前提。

---

## 2.1 embed_tokens 和 merger 没有特别的坑

- **embed_tokens**：纯 `Gather`（embedding lookup），`MatMulNBitsQuantizer` 直接量化，没有 Gemm 要 rewrite。
- **merger**：保持 fp16 不量化。只有一个小线性层，Q4 省不了多少，不值得增加精度风险。

整个 Q4 流程的难点集中在 vision_encoder 的 Gemm→MatMul+Add rewrite，其余都是标准操作。

---

## 3. 当前确认过的干净 Q4 文件大小

当前本地确认过的最小运行文件大小：

```text
embed_tokens_q4.onnx        352K
embed_tokens_q4.onnx.data   47M
vision_encoder_q4.onnx      28M
vision_encoder_q4.onnx.data 209M
merger_fp16.onnx            12K
merger_fp16.onnx.data       45M
```

如果看到类似：

```text
embed_tokens_q4.onnx.data       100M+
vision_encoder_q4.onnx.data     400M+
```

优先检查是否有 stale external data，而不是立刻怀疑量化算法。判断 ONNX 文件是否干净，不能只看 `du -sh`，还要看 initializer 的 external data 引用。

---

## 4. 复现流程（量化）

在 03 的导出 + 优化完成后，运行：

```bash
python3 03-Quantize-ONNX.py \
  --onnx-dir models/export/onnx
```

完成后 `models/export/onnx/` 应包含：

```text
vision_encoder_q4.onnx
vision_encoder_q4.onnx.data
embed_tokens_q4.onnx
embed_tokens_q4.onnx.data
merger_fp16.onnx
merger_fp16.onnx.data
```

运行测试图验证：

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH \
  python3 runtime/glm_ocr_llama.py \
    --image tests/example.png \
    --onnx-dir models/export \
    --gguf models/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf \
    --prompt "请识别图中的所有文字" \
    --max-tokens 64
```

预期能识别测试图里的文字：

```text
GLM OCR TEST
Hello 2026
```

---

## 5. 最小发布集

最终 runtime 不需要带 FP32 baseline。最小集：

```text
models/export/
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

再加：

```text
GLM-OCR-Q8_0.gguf
```

FP32 文件只用于重新导出、debug 和对齐，不应该作为普通用户运行集的一部分。

---

## 6. 经验结论

GLM-OCR 的 Q4 量化工作不难在"能量化"，难在两个前提：

1. **Q4 必须走 weight-only 路线**：`quantize_dynamic` 能跑 INT8 但不支持 Q4；`MatMulNBitsQuantizer` 支持 Q4 但只处理 `MatMul` 节点，所以 fused QKV 的 `Gemm(transB=1)` 需要先 rewrite。
2. **必须先把 Gemm 转成 MatMul + Add**：`MatMulNBitsQuantizer` 只处理 constant-weight MatMul。

只要这两步处理好，加上 03 里清理 external data 的习惯，Q4 runtime 文件可以稳定复现。
