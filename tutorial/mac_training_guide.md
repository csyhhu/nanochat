#  Mac 本地训练到 Serving 全流程指南

> 本文记录在 Apple Silicon Mac（MPS 设备）上，从零完成 nanochat **预训练 → SFT →（可选）RL → Web Serving** 的完整流程，并包含环境修复与注意事项。  
> **主推荐路径**：**16GB 统一内存 baseline**（`model-tag=mac-baseline-d4`，短上下文 + 小 batch）。更高配命令均为可选。

**设计溯源（superpowers brainstorming）**：教程结构与验收标准对应设计 spec，便于评审与迭代：

- `[docs/superpowers/specs/2026-05-09-mac-training-guide-design.md](../docs/superpowers/specs/2026-05-09-mac-training-guide-design.md)`

---

## 目录

1. [设计说明与 baseline 验收](#0-设计说明与-baseline-验收)
2. [环境配置](#1-环境配置)
3. [数据准备](#2-数据准备)
4. [训练 Tokenizer](#3-训练-tokenizer)
5. [预训练（Pretrain）](#4-预训练pretrain)
6. [监督微调（SFT）](#5-监督微调sft)
7. [（可选）强化学习（RL）](#6-可选强化学习rl)
8. [Web Serving](#7-web-serving)（含 [Qwen / Hugging Face 推理后端](#hf-qwen-backend)）
9. [Mac 兼容性补丁说明](#8-mac-兼容性补丁说明)
10. [常见问题](#9-常见问题)
11. [训练时资源监控](#10-训练时资源监控)

---

## 0. 设计说明与 baseline 验收

### 0.1 设计目标（摘自 spec）


| 维度       | 说明                                                                                  |
| -------- | ----------------------------------------------------------------------------------- |
| **读者**   | Apple Silicon + **约 16GB** 内存、优先跑通闭环而非追 SOTA                                        |
| **主路径**  | `gpt` + `depth=4` + `max-seq-len=256` + 小 batch；`--window-pattern=L`（MPS/SDPA 勿用滑窗） |
| **观测**   | 默认 `--run=dummy` 无 W&B；loss 在终端 / `tee` 日志；详见第 4 节表格                                |
| **阶段对齐** | SFT **显式**使用与预训练一致的 `max-seq-len` 与 batch，避免覆盖为 512/大 batch 再次 OOM                  |


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


| 路径                                        | 内容                                               |
| ----------------------------------------- | ------------------------------------------------ |
| `$NANOCHAT_BASE_DIR/base_data_climbmix/`  | 预训练数据（parquet 分片）                                |
| `$NANOCHAT_BASE_DIR/tokenizer/`           | tokenizer.pkl + token_bytes.pt                   |
| `$NANOCHAT_BASE_DIR/base_checkpoints/`    | 预训练 checkpoint                                   |
| `$NANOCHAT_BASE_DIR/chatsft_checkpoints/` | SFT checkpoint（与代码中 `load_model("sft", ...)` 一致） |
| `$NANOCHAT_BASE_DIR/chatrl_checkpoints/`  | RL checkpoint（若运行 `chat_rl`）                     |


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

- `**--num-iterations=2000**`：比「冒烟 200 step」更容易在终端里看到 `loss:` 缓慢下降；若只想验证能跑通，可改小（如 `200`）。
- 若仍 OOM，依次尝试：把 `--device-batch-size` 改为 `2`，并把 `--total-batch-size` 改为 `2048`（须仍满足整除：`2×256=512`，`2048÷512=4`）。

### 训练 loss / 验证指标：为什么看不到「曲线图」？

`scripts/base_train.py` 的行为可以概括为：


| 你想看的                    | 实际在哪里                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **训练 loss（每步）**         | 终端标准输出里每一行里的 `loss: ...`（每步都会 `print`）                                                                             |
| **W&B 网页上的折线图**         | 只有 `**--run` 不是 `dummy`** 且本机已配置 `wandb login` 时才会上传到 Weights & Biases；`**--run=dummy` 会使用 `DummyWandb`，不会上传任何曲线** |
| **W&B 上的 `train/loss`** | 代码里大约每 `**step % 100 == 0**` 才调用一次 `wandb.log`；不是每个 step 一个点                                                       |
| **验证集 `val_bpb`**       | 每隔 `--eval-every` 步会在终端打印 `Validation bpb:`；`meta_*.json` 里保存的是**该次保存/结束时的快照**，**不是**完整历史序列                        |


因此：第一次用文档里的 `**--run=dummy`** 跑时，**没有 W&B 可视化是正常的**。要看曲线可以任选其一：

1. **开 W&B**：`wandb login` 后把 `--run=dummy` 改成例如 `--run=mac_pretrain_d4`，在项目 `nanochat` 下看 `train/loss`、`val/bpb`。
2. **只看终端**：观察每步 `loss:` 与周期性 `Validation bpb:` 是否下降。
3. **自己留日志 + 本地画图**：例如 `python -m scripts.base_train ... 2>&1 | tee pretrain.log`，训练结束后用仓库自带脚本生成 PNG（见下一小节）。

### 从 tee 日志生成 loss / val bpb 曲线（`dev/plot_pretrain_log.py`）

适用于 `base_train` 通过 `tee` 保存的文本日志（解析其中的 `step ... | loss:` 与 `Step ... | Validation bpb:` 行）。

**依赖**：需要 `matplotlib`。若使用本仓库的 uv 环境：

```bash
uv sync --group dev
```

否则：`pip install matplotlib`。

**用法**（在仓库根目录执行）：

```bash
cd /path/to/nanochat

# 默认在日志同目录生成 <日志文件名去掉后缀>_loss.png
python dev/plot_pretrain_log.py pretrain_mac_baseline.log

# 指定输出路径
python dev/plot_pretrain_log.py pretrain_mac_baseline.log -o ~/Desktop/pretrain_loss.png
```

上图包含两条曲线：**训练 loss**（每步 debiased EMA）与 **验证 val bpb**（按 `eval-every` 打印的点）。若日志里只有其中一种行，对应子图仍会生成，缺失的一种会显示提示。

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

> **注意**：DSV4 段默认 `max-seq-len=512`、`device-batch-size=16` 对 **16GB** 仍可能偏紧；若 OOM，请向上一节 **16GB baseline** 对齐（更小的 `max-seq-len` 与 `device-batch-size`），或优先用 `**--arch=gpt`** 跑通流程。

### 主要参数说明


| 参数                    | 说明                                           |
| --------------------- | -------------------------------------------- |
| `--arch`              | 模型架构：`gpt`（标准 Transformer）或 `ds_v4`（MLA+MoE） |
| `--depth`             | Transformer 层数，单一旋钮控制模型大小                    |
| `--window-pattern`    | 注意力窗口模式；Mac 必须用 `L`（全窗口），因为 SDPA 不支持滑窗       |
| `--device-batch-size` | 单设备 batch size，OOM 时依次减半                     |
| `--run=dummy`         | 禁用 wandb 上传；改成 `--run=my_run` 可启用            |
| `--model-tag`         | checkpoint 目录名，默认 `d{depth}`                 |


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

<a id="hf-qwen-backend"></a>

### Hugging Face（Qwen）/ `transformers` 推理后端（进展记录）

同一套 Web UI 与 `/chat/completions`（SSE streaming）接口可通过 **`scripts/chat_web.py --backend=transformers`** 加载 Hugging Face 上的 Qwen 等模型。**nanochat 自带 checkpoint 路径与此无关**（仍用 `--backend=nanochat` 默认值）。

#### 独立 Python 环境（推荐）

`nanochat` 主依赖里的 **`kernels`** 通常要求 `huggingface-hub>=1.10`，而 **`transformers`/`tokenizers`** 常用版本要求 **`huggingface-hub<1.0`**，两边在同一环境里经常会 **pip 冲突**。建议 **单独建一个 conda env** 专门跑 `--backend=transformers`，只装：`torch`、`transformers`、`tokenizers`、`fastapi`、`uvicorn`，以及 **`huggingface-hub` 版本满足 transformers 要求**（不必安装完整 nanochat 依赖）。

为减轻冲突，`chat_web.py` 在 **`--backend=transformers`** 时对 nanochat 引擎做了 **延迟 import**，一般 **不需要** `rustbpe` 即可启动服务。

#### 设备选择：**`--device-type cpu` 往往才能稳定跑 Qwen2.5**

在 **MPS** 上推理 Qwen2.5 时，曾出现 Apple Metal 断言：**临时 NDArray 总字节数超过 \(2^{32}\)**（约 4GB），即使用很小的 `--hf-max-context-len` / `--max-tokens` 也可能触发。**改用 `--device-type cpu`** 后同一模型可稳定对话——这是当前实践中关键的一步。

可选缓解（在仍想用 MPS 时尝试，**不保证**在所有 PyTorch/Qwen 版本组合下有效）：

- `--hf-max-context-len`（例如 `1024`/`2048`，默认脚本里已有上限）
- `--hf-use-cache 0`（关闭 KV cache，有时减小大张量分配）
- `--hf-attn-impl eager`（强制较原始的 attention 路径）

#### 其它有用开关

| 参数 | 说明 |
|------|------|
| `--hf-model-id` | 例如 `Qwen/Qwen2.5-0.5B-Instruct` |
| `--hf-max-layers N` | 只保留前 N 层做实验（会显著削弱能力，通常需要后续训练补偿） |
| `POST /chat/completions_sync` | 非 streaming 调试接口，返回 JSON `{"text": "..."}` |

#### 示例：在本机用 CPU 稳定跑 Qwen2.5（节选）

```bash
cd /path/to/nanochat
export PYTHONPATH="$(pwd)"

python -m scripts.chat_web \
  --backend=transformers \
  --hf-model-id Qwen/Qwen2.5-0.5B \
  --device-type cpu \
  --hf-max-context-len 1024 \
  --port 8001
```

同步调试（一次性返回全文）：`POST http://localhost:8001/chat/completions_sync`。

#### 只保留前 6 层（`--hf-max-layers 6`，实验用）

用于快速观察「浅层 student」在未训练时的行为（后续会做小规模 PT + SFT 补偿）。实现上是：**先从 Hub 加载完整权重，再在内存里只保留 Transformer 的前 6 个 block**；磁盘下载体积不变，但推理时计算量变小。

**建议仍用 CPU**（`--device-type cpu`），与上文 MPS 限制一致。Base 与 Instruct 均可试；做 **PT → SFT** 主线时更常用 **Base** 看裸语言能力。

**Base（`Qwen2.5-0.5B`），仅前 6 层：**

```bash
cd /path/to/nanochat
export PYTHONPATH="$(pwd)"

python -m scripts.chat_web \
  --backend=transformers \
  --hf-model-id Qwen/Qwen2.5-0.5B \
  --hf-max-layers 6 \
  --device-type cpu \
  --hf-max-context-len 1024 \
  --port 8003
```

**Instruct（`Qwen2.5-0.5B-Instruct`），仅前 6 层**（与 full Instruct 对照用）：

```bash
python -m scripts.chat_web \
  --backend=transformers \
  --hf-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --hf-max-layers 6 \
  --device-type cpu \
  --hf-max-context-len 1024 \
  --port 8004
```

- 与 **不截层** 的全模型对照时，请换不同 **`--port`**，避免和已有服务冲突。  
- Web UI 里换模型后若仍看到旧对话，是浏览器标签页内 **`messages` 未清空**；点「新对话」或刷新页面即可。

#### Streaming

transformers 后端使用 **`TextIteratorStreamer`**，仍通过 **`text/event-stream`** 逐段推送 `token` 字段，与前端现有约定兼容。

#### 继续预训练（PT）：开源语料 + 6 层 student

仓库脚本 **`scripts/qwen_continue_pt.py`** 在 HF causal LM（默认 **Base**：`Qwen/Qwen2.5-0.5B`）上做 **因果语言建模损失** 的继续预训练，并支持与推理相同的 **`--max-layers N`**（先加载完整权重再在内存里只保留前 N 个 block）、**`--max-context-len`**（缓解 Mac 上过长上下文带来的显存/缓存问题）。

需要在带 **`torch` + `transformers` + `datasets`** 的环境里运行（可与 `--backend=transformers` 的 Web 服务共用同一 conda env）。在项目里可用：`uv sync --group dev`（若本机未装 `uv`，可用 `pip install transformers datasets torch`）。

**小型公开数据（本地试跑）：** `--preset wikitext` 会拉取 `wikitext` / `wikitext-103-raw-v1`，可按 `--max-samples` 截断行数。

**更大规模开源语料（需自行接受 HF 数据集条款 / 部分需登录）：** 例如 FineWeb-Edu 的 sample 配置 `HuggingFaceFW/fineweb-edu` + `sample-10BT`，用 `--dataset` / `--dataset-config` / `--split train[:20000]` 指定；其它常用 plain-text 列名为 `text` 的数据集也可同样接入 `--text-column`。

**示例（6 层 + WikiText 子集，CPU）：**

```bash
cd /path/to/nanochat
export PYTHONPATH="$(pwd)"

python -m scripts.qwen_continue_pt \
  --model-id Qwen/Qwen2.5-0.5B \
  --max-layers 6 \
  --max-context-len 2048 \
  --device-type cpu \
  --preset wikitext \
  --max-samples 5000 \
  --block-size 512 \
  --max-steps 200 \
  --gradient-accumulation-steps 8 \
  --output-dir ./checkpoints/qwen2.5-0.5b-layers6-pt
```

训练结束后 **`--output-dir`** 下会有 `transformers` 可用的权重与 tokenizer；可用 **`scripts/chat_web.py --backend=transformers --hf-model-id <本地目录>`** 试推理（路径填该目录）。

仅截断到 **前 6 层** 时，裸对话质量往往较差；PT 后再做 **SFT**（以及可选的教师蒸馏）更符合常用流程。

#### 训练后续方向（摘要）

浅层 student 要可用，一般需要 **PT（可选）→ SFT**，复杂场景再加 **教师蒸馏** 等步骤；上文 PT 脚本覆盖「开源语料 + 截层」路径，SFT 仍可用仓库既有流程扩展。

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

### MPS 上 Qwen（transformers）报错：`total bytes of NDArray > 2**32`

属于 Metal Performance Shaders 对临时张量大小的限制；长上下文模型在 MPS 上更容易触发。**优先改用 `--device-type cpu`** 运行 `--backend=transformers`，详见上文 **[Hugging Face（Qwen）/ transformers 推理后端](#hf-qwen-backend)**。

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