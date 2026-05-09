# Mac 本地训练到 Serving 全流程指南

> 本文记录在 Apple Silicon Mac（MPS 设备）上，从零完成 nanochat **预训练 → SFT →（可选）RL → Web Serving** 的完整流程，并包含环境修复与注意事项。  
> **主推荐路径**：**16GB 统一内存 baseline**（`model-tag=mac-baseline-d4`，短上下文 + 小 batch）。更高配命令均为可选。

**设计溯源（superpowers brainstorming）**：教程结构与验收标准对应设计 spec，便于评审与迭代：

- [`docs/superpowers/specs/2026-05-09-mac-training-guide-design.md`](../docs/superpowers/specs/2026-05-09-mac-training-guide-design.md)

---

## 目录

0. [设计说明与 baseline 验收](#0-设计说明与-baseline-验收)
1. [环境配置](#1-环境配置)
2. [数据准备](#2-数据准备)
3. [训练 Tokenizer](#3-训练-tokenizer)
4. [预训练（Pretrain）](#4-预训练pretrain)
5. [监督微调（SFT）](#5-监督微调sft)
6. [（可选）强化学习（RL）](#6-可选强化学习rl)
7. [Web Serving](#7-web-serving)
8. [Mac 兼容性补丁说明](#8-mac-兼容性补丁说明)
9. [常见问题](#9-常见问题)
10. [训练时资源监控](#10-训练时资源监控)

---

## 0. 设计说明与 baseline 验收

### 0.1 设计目标（摘自 spec）

| 维度 | 说明 |
|------|------|
| **读者** | Apple Silicon + **约 16GB** 内存、优先跑通闭环而非追 SOTA |
| **主路径** | `gpt` + `depth=4` + `max-seq-len=256` + 小 batch；`--window-pattern=L`（MPS/SDPA 勿用滑窗） |
| **观测** | 默认 `--run=dummy` 无 W&B；loss 在终端 / `tee` 日志；详见第 4 节表格 |
| **阶段对齐** | SFT **显式**使用与预训练一致的 `max-seq-len` 与 batch，避免覆盖为 512/大 batch 再次 OOM |

### 0.2 方案取舍（概要）

- **主路径（A）**：单一文档内区分「16GB baseline」与「大内存/DSV4 可选」，命令写死整除关系 — **本教程采用**。
- **备选（B）**：仅写「OOM 则减半」— 读者仍易误用 512 上下文。
- **备选（C）**：拆多文档 — 增加导航成本；与本 repo 维护方式不符。

### 0.3 Baseline 验收清单（合格 = 流程跑通，不承诺「像 ChatGPT」）

1. **数据**：`base_data_climbmix` 下存在 parquet；tokenizer 已训练。
2. **预训练**：`base_train` 能跑完设定步数；终端中 `loss:` 或周期性 `Validation bpb:` 可见趋势（非严格数值门槛）。
3. **SFT**：`chat_sft` 能完成；`chatsft_checkpoints/{model-tag}/` 下出现新 checkpoint。
4. **Serving**：`chat_web` 可启动，浏览器或 `curl` 能完成至少一轮对话而不崩溃。
5. **稳定性**：无持续 OOM；若偶发，按第 9 节降 `device-batch-size` / `max-seq-len`。

---

## 1. 环境配置

### 使用 Miniconda（推荐，替代 uv）

```bash
# 创建独立环境
conda create -n nanochat python=3.11 -y
conda activate nanochat

# 安装 PyTorch（Mac CPU + MPS 版本，从官方 whl 源）
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# 安装其余依赖
pip install \
  datasets>=4.0.0 \
  fastapi>=0.117.1 \
  kernels>=0.11.7 \
  psutil>=7.1.0 \
  rustbpe==0.1.0 \
  tiktoken>=0.11.0 \
  "tokenizers>=0.22.0" \
  uvicorn>=0.36.0 \
  wandb>=0.21.3 \
  filelock pyarrow requests

# 以开发模式安装项目本身（改代码立即生效）
cd /path/to/nanochat
pip install -e . --no-deps
```

### 配置数据目录（可选，默认 `~/.cache/nanochat`）

```bash
# 将所有数据和 checkpoint 存到 ~/data
export NANOCHAT_BASE_DIR="$HOME/data"

# 永久生效（写入 ~/.zshrc）
echo 'export NANOCHAT_BASE_DIR="$HOME/data"' >> ~/.zshrc
source ~/.zshrc
```

目录结构说明：

| 路径 | 内容 |
|---|---|
| `$NANOCHAT_BASE_DIR/base_data_climbmix/` | 预训练数据（parquet 分片） |
| `$NANOCHAT_BASE_DIR/tokenizer/` | tokenizer.pkl + token_bytes.pt |
| `$NANOCHAT_BASE_DIR/base_checkpoints/` | 预训练 checkpoint |
| `$NANOCHAT_BASE_DIR/chatsft_checkpoints/` | SFT checkpoint（与代码中 `load_model("sft", ...)` 一致） |
| `$NANOCHAT_BASE_DIR/chatrl_checkpoints/` | RL checkpoint（若运行 `chat_rl`） |

---

## 2. 数据准备

### 方案一：从 HuggingFace 下载官方数据（推荐）

```bash
mkdir -p $NANOCHAT_BASE_DIR

# 如遇网络问题，先设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载 8 个训练分片（Mac 本地实验够用）+ 1 个验证分片
python -m nanochat.dataset -n 8
```

### 方案二：用自己的文本数据

数据格式要求：带 `text` 列的 parquet 文件，文件名格式为 `shard_XXXXX.parquet`。
至少需要 2 个文件（**最后一个**自动作为验证集）。

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

texts_train = ["训练文本1...", "训练文本2...", ...]
texts_val   = ["验证文本1...", ...]

data_dir = os.path.expanduser("~/.cache/nanochat/base_data_climbmix")
os.makedirs(data_dir, exist_ok=True)

pq.write_table(pa.table({"text": texts_train}),
               os.path.join(data_dir, "shard_00000.parquet"))
pq.write_table(pa.table({"text": texts_val}),
               os.path.join(data_dir, "shard_06542.parquet"))  # val shard（编号最大）
```

---

## 3. 训练 Tokenizer

在预训练数据上从零训练一个 GPT-4 风格的 BPE tokenizer（词表大小 32,768）。

```bash
# 默认使用 2B 字符；Mac 本地可以缩小加速
python -m scripts.tok_train --max-chars=2000000000

# Mac 快速验证（1 亿字符，约 2 分钟）
python -m scripts.tok_train --max-chars=100000000
```

产物保存在 `$NANOCHAT_BASE_DIR/tokenizer/`：
- `tokenizer.pkl`：tiktoken Encoding 对象（推理用）
- `token_bytes.pt`：每个 token 的字节数（bits-per-byte 评估用）

---

## 4. 预训练（Pretrain）

### 16GB 内存 / M1 Pro 类机器：最低可跑 baseline（推荐默认）

> 目标：在 **Apple Silicon + 16GB 统一内存** 上尽量降低 OOM 概率，跑通预训练闭环；模型能力会偏弱，适合作为后续迭代的 baseline。  
> `total_batch_size` 必须整除 `device_batch_size × max_seq_len × 进程数`（单机默认可认为进程数为 1）。

```bash
python -m scripts.base_train \
    --arch=gpt \
    --depth=4 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=256 \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --eval-every=50 \
    --eval-tokens=65536 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --num-iterations=2000 \
    --run=dummy \
    --model-tag=mac-baseline-d4
```

- **`--num-iterations=2000`**：比「冒烟 200 step」更容易在终端里看到 `loss:` 缓慢下降；若只想验证能跑通，可改小（如 `200`）。
- 若仍 OOM，依次尝试：把 `--device-batch-size` 改为 `2`，并把 `--total-batch-size` 改为 `2048`（须仍满足整除：`2×256=512`，`2048÷512=4`）。

### 训练 loss / 验证指标：为什么看不到「曲线图」？

`scripts/base_train.py` 的行为可以概括为：

| 你想看的 | 实际在哪里 |
|---------|------------|
| **训练 loss（每步）** | 终端标准输出里每一行里的 `loss: ...`（每步都会 `print`） |
| **W&B 网页上的折线图** | 只有 **`--run` 不是 `dummy`** 且本机已配置 `wandb login` 时才会上传到 Weights & Biases；**`--run=dummy` 会使用 `DummyWandb`，不会上传任何曲线** |
| **W&B 上的 `train/loss`** | 代码里大约每 **`step % 100 == 0`** 才调用一次 `wandb.log`；不是每个 step 一个点 |
| **验证集 `val_bpb`** | 每隔 `--eval-every` 步会在终端打印 `Validation bpb:`；`meta_*.json` 里保存的是**该次保存/结束时的快照**，**不是**完整历史序列 |

因此：第一次用文档里的 **`--run=dummy`** 跑时，**没有 W&B 可视化是正常的**。要看曲线可以任选其一：

1. **开 W&B**：`wandb login` 后把 `--run=dummy` 改成例如 `--run=mac_pretrain_d4`，在项目 `nanochat` 下看 `train/loss`、`val/bpb`。
2. **只看终端**：观察每步 `loss:` 与周期性 `Validation bpb:` 是否下降。
3. **自己留日志**：例如 `python -m scripts.base_train ... 2>&1 | tee pretrain.log`，事后用脚本或编辑器从 `pretrain.log` 里抽 `loss:` 画图。

### 内存较宽裕（例如 32GB+ 或 M3 Max 类）：更大 GPT 配置（可选）

`runs/runcpu.sh` 与旧版文档中的大 batch 更适合大内存机器；在 16GB 上容易 OOM。

```bash
python -m scripts.base_train \
    --arch=gpt \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=dummy \
    --model-tag=d6-mac-large
```

### DSV4 风格预训练（MLA + MoE，Mac 推荐参数）

```bash
python -m scripts.base_train \
    --arch=ds_v4 \
    --depth=4 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=16 \
    --total-batch-size=8192 \
    --eval-every=100 \
    --eval-tokens=262144 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=3000 \
    --run=dummy

```

> **注意**：DSV4 段默认 `max-seq-len=512`、`device-batch-size=16` 对 **16GB** 仍可能偏紧；若 OOM，请向上一节 **16GB baseline** 对齐（更小的 `max-seq-len` 与 `device-batch-size`），或优先用 **`--arch=gpt`** 跑通流程。

### 主要参数说明

| 参数 | 说明 |
|---|---|
| `--arch` | 模型架构：`gpt`（标准 Transformer）或 `ds_v4`（MLA+MoE） |
| `--depth` | Transformer 层数，单一旋钮控制模型大小 |
| `--window-pattern` | 注意力窗口模式；Mac 必须用 `L`（全窗口），因为 SDPA 不支持滑窗 |
| `--device-batch-size` | 单设备 batch size，OOM 时依次减半 |
| `--run=dummy` | 禁用 wandb 上传；改成 `--run=my_run` 可启用 |
| `--model-tag` | checkpoint 目录名，默认 `d{depth}` |

### Checkpoint 位置

```
$NANOCHAT_BASE_DIR/base_checkpoints/{model-tag}/
├── model_{step:06d}.pt       # 模型参数
├── optim_{step:06d}_rank0.pt # 优化器状态
└── meta_{step:06d}.json      # 训练元数据（arch、config、loss 等）
```

---

## 5. 监督微调（SFT）

SFT 在 **base** 预训练 checkpoint 上做指令微调。`scripts/chat_sft.py` 会从 `base_checkpoints` 加载；**务必**与第 4 节预训练使用 **同一 `--model-tag`**（本 baseline 为 `mac-baseline-d4`）。

### 下载身份对话数据（可选）

```bash
# NANOCHAT_BASE_DIR 未设置时默认为 ~/.cache/nanochat
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
```

### 16GB baseline：与预训练对齐的 SFT（推荐）

与第 4 节相同：`max-seq-len=256`，`device-batch-size=4`，`total-batch-size=4096`（须整除 `4×256=1024`）。`eval-tokens` 略减以降低 eval 峰值。`num-iterations` 取正整数以便观察收敛；**仅冒烟**可改小（如 `50`）。

```bash
python -m scripts.chat_sft \
    --model-tag=mac-baseline-d4 \
    --max-seq-len=256 \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --eval-every=100 \
    --eval-tokens=131072 \
    --chatcore-every=-1 \
    --num-iterations=800 \
    --run=dummy \
    2>&1 | tee sft_mac_baseline.log
```

**关于继承**：若省略 `--max-seq-len` / batch 等，脚本会尝试从 **base** 的 `meta` 继承。教程仍 **显式写出** 上述参数，避免沿用过大的历史 checkpoint 或误读行为。

**W&B**：与预训练相同，`--run=dummy` 无网页曲线；需要图表时 `wandb login` 后改用 `--run=my_sft_run`。

### 大内存机器可选（长上下文 SFT）

仅在 **不易 OOM** 时使用（例如 32GB+ 且 base 模型也在长序列上训练）：

```bash
python -m scripts.chat_sft \
    --model-tag=d6-mac-large \
    --max-seq-len=512 \
    --device-batch-size=16 \
    --total-batch-size=8192 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --chatcore-every=-1 \
    --num-iterations=1500 \
    --run=dummy
```

SFT 产物目录（与代码一致）：

```
$NANOCHAT_BASE_DIR/chatsft_checkpoints/{model-tag}/
├── model_{step:06d}.pt
├── optim_{step:06d}_rank0.pt
└── meta_{step:06d}.json
```

---

## 6. （可选）强化学习（RL）

> **非 16GB 闭环必选项。** `scripts/chat_rl.py` 默认超参偏 GPU；在 Mac 上建议先跑通 SFT 与 Web，再按需做 RL 实验。MPS 上仍可能存在 sharp edge，以下仅为 **保守起点**。

前置：已完成 SFT，且与下面 `--model-tag` 一致（与预训练 tag 相同即可，例如 `mac-baseline-d4`）。

```bash
python -m scripts.chat_rl \
    --model-tag=mac-baseline-d4 \
    --device-batch-size=2 \
    --examples-per-step=4 \
    --num-samples=4 \
    --max-new-tokens=128 \
    --eval-every=30 \
    --eval-examples=100 \
    --save-every=120 \
    --num-epochs=1 \
    --run=dummy \
    2>&1 | tee chat_rl_mac_baseline.log
```

若 OOM 或极慢：继续降低 `--device-batch-size`、`--examples-per-step`、`--num-samples`、`--max-new-tokens`，或暂时跳过 RL。

产物目录：`$NANOCHAT_BASE_DIR/chatrl_checkpoints/{model-tag}/`。

---

## 7. Web Serving

### 启动 Web UI

```bash
# 自动加载最新 SFT checkpoint，在 MPS/CPU 上运行
python -m scripts.chat_web

# 指定参数（max-tokens 建议不超过训练时常见长度太多；baseline 为短上下文模型）
python -m scripts.chat_web \
    --source=sft \
    --port=8000 \
    --temperature=0.8 \
    --top-k=50 \
    --max-tokens=256
```

浏览器访问 `http://localhost:8000` 即可看到 ChatGPT 风格的对话界面。

### 命令行对话

```bash
python -m scripts.chat_cli -p "法国的首都是哪里？"
```

### API 接口（兼容 OpenAI 格式）

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好！"}],
    "temperature": 0.8,
    "max_tokens": 200
  }'
```

---

## 8. Mac 兼容性补丁说明

在 Mac（MPS 设备）上运行时，需要对以下两个文件做修改，项目仓库中已包含这些修复：

### `scripts/base_train.py`

**问题**：`torch.compile` 的 `inductor` 后端不支持 MPS，`aten.mean.dim` 等算子会返回 `None` 导致崩溃。

**修复**：非 CUDA 设备跳过 `torch.compile`：

```python
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)
else:
    print0(f"Skipping torch.compile (device_type={device_type}): inductor backend not fully supported on MPS/CPU.")
```

### `nanochat/optim.py`

**问题一**：`adamw_step_fused` 和 `muon_step_fused` 函数级 `@torch.compile` 在 MPS 上崩溃（`Device mps not supported`）。

**修复**：用条件编译装饰器替换：

```python
_CUDA_AVAILABLE = torch.cuda.is_available()
def _maybe_compile(**kwargs):
    if _CUDA_AVAILABLE:
        return torch.compile(**kwargs)
    return lambda fn: fn  # MPS/CPU 上跳过编译
```

**问题二**：优化器函数内部的 0-D CPU 标量张量（`lr_t`、`beta1_t` 等）在 eager 模式下无法与 MPS 张量混用（`Expected all tensors to be on the same device`）。

**修复**：在 eager 模式下提前用 `.item()` 提取 Python 标量：

```python
if not torch.compiler.is_compiling():
    step_t  = step_t.item()
    lr_t    = lr_t.item()
    beta1_t = beta1_t.item()
    # ...
```

---

## 9. 常见问题

### `torch._inductor.exc.LoweringException: TypeError: 'NoneType' object is not callable`

MPS 上 `torch.compile` inductor 不支持。已在 `base_train.py` 中修复（见第 8 节）。

### `RuntimeError: Expected all tensors to be on the same device`

优化器 0-D CPU 标量张量与 MPS 参数不在同一设备。已在 `optim.py` 中修复（见第 8 节）。

### 数据目录找不到

```
AssertionError: No dataset parquet files found, did you run dataset.py?
```

确认数据已下载，且目录结构正确：

```bash
ls $NANOCHAT_BASE_DIR/base_data_climbmix/
# 应该看到 shard_XXXXX.parquet 文件
```

### OOM / 内存不足

依次尝试以下参数缩减：
1. `--device-batch-size` 减半（32 → 16 → 8 → 4）
2. `--max-seq-len` 减半（512 → 256）
3. `--depth` 降低（6 → 4）

### 训练速度很慢

Mac 上没有 FA3，使用 PyTorch SDPA fallback，这是正常现象。训练是为了学习流程，不追求速度。`bf16_mfu: 0.00` 是因为 MPS 没有 CUDA FLOPS 参考值，不代表实际利用率为 0。

### 用了 `--run=dummy` 看不到 W&B 上的 loss 曲线

这是预期行为：`dummy` 会禁用 wandb。训练 loss 在终端每步的 `loss:` 里；需要网页曲线时请 `wandb login` 并把 `--run` 改成非 `dummy` 的名称。说明见上文 **「训练 loss / 验证指标」** 小节。

---

## 10. 训练时资源监控

图形界面：`open -a "Activity Monitor"`，观察 **内存** 与 **CPU**，在进程列表中筛选 `python`。

终端快照：

```bash
top -l 1 -s 0 | head -n 20
memory_pressure
vm_stat
df -h /
```

持续刷新：`top`（按 `q` 退出）。若内存长期顶满、系统严重卡顿或大量换页，请回到第 4～5 节降低 `device-batch-size` / `max-seq-len`。
