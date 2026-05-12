# BPE（Byte Pair Encoding）Tokenizer 训练方式

## 核心思想

BPE 的本质是一个**贪心的数据压缩算法**，反复把语料中"出现最频繁的相邻符号对"合并成一个新符号，直到词表达到目标大小为止。

---

## 训练过程

### 第 0 步：初始化——字节级词表

现代 LLM 的 BPE（GPT-2 / GPT-4 / nanochat 均采用此方式）从 **256 个 UTF-8 字节**出发，而不是从字符或单词出发。

好处：词表天然覆盖所有语言和符号，不会出现 `<unk>`。

```
初始词表 = {0x00, 0x01, ..., 0xFF}   # 256 个字节
```

---

### 第 1 步：Pre-tokenize（预切分）

在学习合并规则之前，先用正则表达式把文本切成**不跨越语义边界的片段**（不会把单词和标点合并在一起）。

nanochat 使用与 GPT-4 相似的 pattern（见 `nanochat/tokenizer.py` 中的 `SPLIT_PATTERN`）：

```
'[sdmt] | 单词 | 1-2位数字 | 标点 | 空白 ...
```

例如 `"Hello, world!"` 会被切成 `["Hello", ",", " world", "!"]`，  
BPE **只在每个片段内部**学习合并规则，永远不会跨片段合并。

> nanochat 将数字分组改为最多 2 位（`\p{N}{1,2}`）而非 GPT-4 的 3 位，  
> 在 32K 词表下节省了 token 空间。

---

### 第 2 步：统计所有相邻字节对的频率

将每个片段拆成字节序列，统计所有相邻字节对在整个语料中出现的总次数：

```
片段: "low"   → [l, o, w]
片段: "lower" → [l, o, w, e, r]
片段: "new"   → [n, e, w]

统计相邻对频率:
  (l, o) → 2
  (o, w) → 2
  (w, e) → 1
  (n, e) → 1
  ...
```

---

### 第 3 步：合并频率最高的对

找到频率最高的对 `(l, o)`，将其合并为新 token `lo`，更新整个语料，并记录合并规则：

```
合并前: [l, o, w]       → 合并后: [lo, w]
合并前: [l, o, w, e, r] → 合并后: [lo, w, e, r]

记录合并规则: merge(l, o) → lo    ← merge rule
词表新增 token: lo
```

---

### 第 4 步：重复直到词表满

重新统计频率，找下一个最高频对，继续合并，循环直到词表达到目标大小：

```
第 1 次合并: (l, o)   → lo     词表: 257 个
第 2 次合并: (lo, w)  → low    词表: 258 个
第 3 次合并: (e, r)   → er     词表: 259 个
...
第 N 次合并:                   词表: 256 + N 个
```

nanochat 默认目标词表大小为 **32,768**，其中 9 个为特殊 token，  
因此共学习 `32768 - 256 - 9 = 32503` 条合并规则。

---

## 训练产物

训练完成后，tokenizer 的核心产物是一张**有序的合并规则列表**：

```python
merge_rules = [
    (b"l",  b"o"),    # 第 1 条：优先级最高（最先被合并）
    (b"lo", b"w"),    # 第 2 条
    (b"e",  b"r"),    # 第 3 条
    ...               # 共 32503 条
]
```

规则的**顺序即优先级**——训练时越早合并的对，推理时优先级越高。  
在代码中以 `mergeable_ranks: dict[bytes, int]` 形式存储（rank 越小优先级越高）。

---

## 推理时如何编码

拿到新文本后，按如下步骤编码：

1. 用 pre-tokenize 正则将文本切成片段
2. 每个片段转为字节序列
3. **按优先级顺序**依次尝试每条合并规则，能合并则合并
4. 最终每个片段变成一串 token id

```
编码 "lower":
初始:              [l, o, w, e, r]
应用 merge(l,o):   [lo, w, e, r]
应用 merge(lo,w):  [low, e, r]
应用 merge(e,r):   [low, er]
结果: [token_id("low"), token_id("er")]
```

---

## 与 nanochat 实现的对应关系

| 概念 | nanochat 对应代码 |
|---|---|
| Pre-tokenize 正则 | `SPLIT_PATTERN` in `nanochat/tokenizer.py` |
| BPE 训练（Rust 实现） | `rustbpe.Tokenizer.train_from_iterator()` |
| 合并规则存储 | `mergeable_ranks: dict[bytes, int]` |
| 推理编码引擎 | `tiktoken`（C 实现，高速） |
| 特殊 token 注入 | `<\|bos\|>` 等 9 个 token 直接 append，不参与 BPE 训练 |
| 训练入口脚本 | `scripts/tok_train.py` |
| 保存路径 | `$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl` |
| bits-per-byte 辅助文件 | `$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt` |

训练使用 `rustbpe`（Rust，擅长统计频率、速度快），推理使用 `tiktoken`（C 扩展，编码吞吐高），两者分工明确。

---

## 完整训练流程图

```
下载的 parquet 语料
        │
        ▼  (流式读取，最多 2B 字符，每篇文档截断至 10K 字符)
  Pre-tokenize（正则切片段）
        │
        ▼
  统计相邻字节对频率
        │
        ▼
  合并最高频对 → 更新语料 → 记录 merge rule
        │
        └─── 循环，直到词表达到 32,768
        │
        ▼
  tiktoken.Encoding 对象
        │
        ├──▶  tokenizer.pkl       （推理 / 训练时加载）
        └──▶  token_bytes.pt      （计算 bits per byte 时加载）
```
