# Debugging mRoPE Embedding Injection in GLM-OCR

本文记录 GLM-OCR ONNX + GGUF 混合推理中最关键的 bug：**视觉特征正确，文本 embedding 正确，但注入 image embeddings 后输出乱码**。

“所有子模块看起来都正确”，“最终 batch layout 出错”，本文讲述逐步排查过程。

参考记录：

- opencode `ses_15a2a6bc0ffe1icaU1zLeNHRfr`: Debug GLM-OCR ctypes embd mRoPE
- opencode `ses_179019d36ffeK2INgwy5P1nmZV`: Find GLM-OCR mRoPE position ID logic
- opencode `ses_1796eededffezn1I53DOJXDC0G`: 优化视频理解项目-glmocr-纯onnx推理
- 当前代码：`runtime/glm_ocr_llama.py`

---

## 1. 现象：所有局部检查都过了，但端到端乱码

当时已经完成的局部验证：

| 检查 | 结果 |
|---|---|
| ONNX vision output vs PyTorch vision output | cosine=1.0 |
| ONNX token embedding vs GGUF token mode | text-only logits max diff=0.0 |
| 纯文本 embedding 模式 | 正常输出 |
| 不带图的 OCR prompt | 模型能理解任务 |
| 带 image embeddings 的完整 OCR | 乱码 |

这类 bug 最容易误判，因为每个独立模块都能自证。


但端到端仍然不对。

最后证明：问题不在 ONNX 视觉特征本身，而在 **image embeddings 被送进 llama.cpp 时的 mRoPE position layout**。

更关键的是，乱码不随这些尝试消失：

1. sequential position
2. row-major position
3. Transformers 风格 position
4. llama.cpp mtmd 风格 position
5. 单 batch / split batch
6. raw features / normalized features
7. PyTorch features / ONNX features

这说明问题不在"ONNX vision 输出差一点"，而在 **image embeddings 进入 llama.cpp 的方式不等价于 `mtmd-helper`**。

---

## 2. GLM-OCR 不是普通 1D RoPE

GLM-OCR 的 decoder 是 GLM4 类模型，使用 mRoPE。它不是每个 token 一个位置，而是多维位置。

Transformers 侧 `get_rope_index` 的语义更接近：

```text
position_ids shape = [3, batch, seq]
rows = temporal / height / width
```

llama.cpp 内部进一步把 mRoPE position 作为 4 组 position 存储：

```text
t / y / x / z
```

其中 `z` 对当前图像输入通常为 0。

---

## 3. 最容易写错的地方：位置数组布局

错误直觉是按 token 交错写：

```text
token0: t, y, x, z
token1: t, y, x, z
token2: t, y, x, z
```

也就是：

```text
pos[i * 4 + 0] = t
pos[i * 4 + 1] = y
pos[i * 4 + 2] = x
pos[i * 4 + 3] = z
```

但 llama.cpp `mtmd-helper` 对 embedding batch 使用的是分组布局：

```text
all_t: t0, t1, t2, ...
all_y: y0, y1, y2, ...
all_x: x0, x1, x2, ...
all_z: z0, z1, z2, ...
```

也就是：

```text
pos[0 * n + i] = t
pos[1 * n + i] = y
pos[2 * n + i] = x
pos[3 * n + i] = z
```

这点来自 llama.cpp 的 `mtmd-helper.cpp`：

```cpp
pos[i]                    = rel_pos[i].t;
pos[i + batch.n_tokens]   = rel_pos[i].y;
pos[i + batch.n_tokens*2] = rel_pos[i].x;
pos[i + batch.n_tokens*3] = rel_pos[i].z;
```

如果布局写成交错式，数值看起来都有，但 decoder 读到的是错位 position，结果就是乱码。

---

## 4. 第二个坑：batch 申请大小不是 `n_tokens`，而是4*`n_tokens`

普通文本 batch 只需要：

```text
n_tokens 个 pos
```

GLM-OCR image embedding batch 需要：

```text
4 * n_tokens 个 pos
```

当前 runtime 的做法：

```python
MROPE_DIMS = 4
batch = LlamaBatch(n_tokens * MROPE_DIMS, embd_dim, 1)
batch.n_tokens = n_tokens
```

这里看起来反直觉：`LlamaBatch` 初始化时按 `4*n` 分配空间，但真实 `batch.n_tokens` 要改回 `n`。

原因是：

1. 分配空间要够放 `4*n` 个 position。
2. 真实送入 decoder 的 token 数仍然是 `n`。
3. llama.cpp 内部根据 `n_pos_per_embd=4` 读取 position buffer。

如果只按 `n_tokens` 申请，Python 侧写入 `4*n` position 时要么越界，要么只写了第一组 position。

---

## 5. 第三个坑：image token 的位置推进不是 token 数

GLM-OCR prompt 中 image placeholder 会被替换成很多 `image_token_id`。但 image block 消耗的文本位置长度不是 `num_image_tokens`。

对于图像，位置推进是：

```text
current_pos += max(H, W) // spatial_merge_size
```

而不是：

```text
current_pos += num_image_tokens
```

这和 Transformers 的 `get_rope_index` 逻辑一致。它让二维图像区域在 mRoPE 空间里占据一个二维网格，而不是一条很长的一维 token 序列。

---

## 6. 修复后，实现的核心逻辑

当前 `runtime/glm_ocr_llama.py` 里实际做了三件事。

### 6.1 计算 image token 的 4D position

```python
t, h, w = image_grid_thw[image_idx]
llm_h = int(h) // self.spatial_merge_size
llm_w = int(w) // self.spatial_merge_size
n_img = int(t) * llm_h * llm_w

t_pos_img = np.full(n_img, current_pos, dtype=np.int32)
h_pos_img = np.array([j // llm_w for j in range(n_img)], dtype=np.int32) + current_pos
w_pos_img = np.array([j % llm_w for j in range(n_img)], dtype=np.int32) + current_pos
```

### 6.2 按分组布局拼接

```python
positions = np.concatenate([t_pos, h_pos, w_pos, x_pos])
```

### 6.3 写入 llama_batch

```python
ctypes.memmove(batch.embd, embeds.ctypes.data, embeds.nbytes)
ctypes.memmove(batch.struct.pos, pos_array, len(positions) * 4)
```

---

## 7. EOS 也要用 config 里的全部 token

调试时还遇到过一个次级问题：模型可能生成多个 EOS 之一。

GLM-OCR config 中：

```text
eos_token_id = [59246, 59253]
```

如果只检查 llama.cpp vocab 返回的单个 EOS，可能出现已经生成终止 token 但 runtime 不停，导致后面继续重复或多出 Markdown/code fence。

当前做法：

```python
self.eos_tokens = set(cfg["text_config"]["eos_token_id"])
```

decode 时只要 token 在这个集合里就停止。

---

## 8. 修复后的验证

修复 mRoPE 分组布局和 EOS 检查后，测试图输出恢复正常：

```text
GLM OCR TEST
Hello 2026
```

opencode 记录中的关键结果：

```text
修复 mRoPE 分组布局后，模型正确输出 "GLM OCR TEST\nHello 2026" 并在 12 步后命中 EOS 停止。
```

后续批量测试同一图像两次，也返回两个正确结果。

---

## 9. 排查顺序建议

以后如果 GLM-OCR 输出乱码，不要先改量化，也不要先怀疑 prompt。按下面顺序排：

1. llama-server + mmproj 是否能正确识别同一张图。
2. ONNX vision output shape 是否正确。
3. image token 数量是否等于 image features 数量。
4. image embeddings 是否替换到 prompt 中正确的位置。
5. `batch.embd` 是否 C-contiguous。
6. `batch.pos` 是否按 `[all_t, all_y, all_x, all_z]` 分组写入。
7. `LlamaBatch` 是否按 `4*n_tokens` 申请 position 空间。
8. `batch.n_tokens` 是否设置回真实 token 数。
9. EOS 是否检查了 config 中的全部 token。

---

## 10. 经验结论

GLM-OCR 的 ctypes 路线不是“把 embedding 塞进 llama.cpp”这么简单。

真正需要复刻的是 llama.cpp `mtmd-helper` 对 image embedding batch 的语义：

```text
embedding 数据本身
+ 4D mRoPE position buffer
+ 分组式 position layout
+ 正确的 image block position 推进
+ 正确的 EOS 集合
```

只要其中一个环节按普通文本模型处理，局部数值检查仍可能全部通过，但端到端 OCR 会变成乱码。
