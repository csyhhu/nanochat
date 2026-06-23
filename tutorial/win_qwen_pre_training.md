# Windows：Qwen2.5-0.5B（6 层）加载与 Serving

> 延续 Mac 上 `tutorial/mac_training_guide.md` 第 7 节 **Hugging Face / transformers 后端** 实验。  
> Windows **无 MPS**，`--device-type cpu` 是默认稳妥路径，且**不会**遇到 Mac 上 Metal 临时张量 >4GB 的限制。

- [Windows：Qwen2.5-0.5B（6 层）加载与 Serving](#windowsqwen25-05b6-层加载与-serving)
  - [1. 能否在 Windows 上跑？](#1-能否在-windows-上跑)
  - [2. 环境策略：单独 conda env（推荐）](#2-环境策略单独-conda-env推荐)
    - [2.1 创建环境（PowerShell）](#21-创建环境powershell)
    - [2.2 安装依赖（推荐：国内 PyPI 镜像）](#22-安装依赖推荐国内-pypi-镜像)
    - [2.3 Hugging Face 模型下载镜像（国内必设）](#23-hugging-face-模型下载镜像国内必设)
      - [方式 A：环境变量（推荐，与 `chat_web` / 冒烟脚本通用）](#方式-a环境变量推荐与-chat_web--冒烟脚本通用)
      - [默认缓存位置（未指定 `--local-dir` 时）](#默认缓存位置未指定---local-dir-时)
      - [方式 B：先离线下好到固定目录，再指向本地](#方式-b先离线下好到固定目录再指向本地)
      - [方式 C：ModelScope（HF 仍失败时）](#方式-cmodelscopehf-仍失败时)
      - [常见报错](#常见报错)
    - [2.4 可选：缓存目录](#24-可选缓存目录)
  - [3. 激活与项目路径](#3-激活与项目路径)
  - [4. 冒烟：6 层加载（不启动 Web）](#4-冒烟6-层加载不启动-web)
  - [5. Web Serving（与 Mac 相同参数）](#5-web-serving与-mac-相同参数)
      - [浏览器访问地址（Windows 实践记录）](#浏览器访问地址windows-实践记录)
  - [8. WikiText continue Pre-Training（含周期性 eval）](#8-wikitext-continue-pre-training含周期性-eval)
    - [8.0 机器运行时间测试：**Small's Win**（吞吐与 LoRA / 全量对比）](#80-机器运行时间测试smalls-win吞吐与-lora--全量对比)
      - [全量微调 vs LoRA（实测）](#全量微调-vs-lora实测)
      - [全量扫参最佳（`run_0020`）](#全量扫参最佳run_0020)
      - [时间外推（仅 train，无 eval）](#时间外推仅-train无-eval)
    - [8.1 观测 CPU / 内存 占用](#81-观测-cpu--内存-占用)
      - [Windows 本机（PowerShell 里跑训练时）](#windows-本机powershell-里跑训练时)
      - [htop](#htop)
      - [怎么判断「是否吃满机器」（CPU 训练）](#怎么判断是否吃满机器cpu-训练)
    - [8.2 最佳训练参数（速度最快）：扫参 benchmark 与结果分析](#82-最佳训练参数速度最快扫参-benchmark-与结果分析)
      - [比速度看哪个指标？](#比速度看哪个指标)
      - [小网格（约 6 组）](#小网格约-6-组)
      - [大网格（72 组，耗时长）](#大网格72-组耗时长)
      - [分析已有 run（无需重新训练）](#分析已有-run无需重新训练)
    - [8.4 快速Eval: `--eval-max-samples` 选择](#84-快速eval---eval-max-samples-选择)
      - [运行 sweep（自动批量）](#运行-sweep自动批量)
      - [汇总结果](#汇总结果)
      - [结果](#结果)
      - [解读与建议](#解读与建议)
    - [8.5 全量微调 + 定期 Eval（`--full-finetune`）](#85-全量微调--定期-eval--full-finetune)
  - [7. 常见问题](#7-常见问题)


## 1. 能否在 Windows 上跑？

**可以。** 相关代码均为纯 PyTorch + transformers，无 Mac 专用逻辑：

| 能力 | 脚本 / 模块 |
|------|-------------|
| Hub 加载 + 截断前 N 层 | `nanochat/transformers_backend.py` → `_truncate_layers_inplace` |
| Web UI + SSE | `scripts/chat_web.py --backend=transformers` |
| 继续预训练（可选） | `scripts/qwen_continue_pt.py` |

与 Mac 相同的注意点：

- 仍会**先下载完整 0.5B 权重**（约 1GB），再在内存里只保留前 6 层。
- 6 层未做 PT/SFT 时，对话质量很差，仅用于验证「加载 + serving 链路」。

---

## 2. 环境策略：单独 conda env（推荐）

主环境 `nanochat`（`uv sync` / `kernels`）与 `transformers` 对 **`huggingface-hub` 版本** 要求常冲突。  
请使用 **独立 env** `nanochat-qwen`，只装推理 + 可选 PT 所需包（与 Mac 教程一致）。

### 2.1 创建环境（PowerShell）

```powershell
# 若 conda 未进 PATH，用完整路径（按本机安装位置改）
$Conda = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"

& $Conda create -n nanochat-qwen python=3.11 -y
& $Conda run -n nanochat-qwen python -m pip install --upgrade pip
```

### 2.2 安装依赖（推荐：国内 PyPI 镜像）

激活环境后，**优先用清华源**（避免 `No matching distribution` / 下载中断）：

```powershell
conda activate nanochat-qwen

$PyPI = "https://pypi.tuna.tsinghua.edu.cn/simple"
$Host = "pypi.tuna.tsinghua.edu.cn"

# 1) PyTorch CPU（官方 CPU 索引，体积大；若慢可多次重试）
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# 2) transformers 及 Web 服务依赖（走清华源）
python -m pip install -i $PyPI --trusted-host $Host `
  numpy "transformers>=4.57.3,<5.0" fastapi uvicorn psutil requests safetensors

# 3) 可选：继续预训练（Trainer 需要 accelerate；默认 LoRA 需要 peft）
python -m pip install -i $PyPI --trusted-host $Host "datasets>=4.0.0" "accelerate>=0.26.0" "peft>=0.13.0"

# 验收
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

也可用项目清单（同样加 `-i`）：

```powershell
python -m pip install -i $PyPI --trusted-host $Host -r D:\WorkSpace\nanochat\requirements-qwen-win.txt
```

**其它常用 PyPI 镜像**（任选其一替换 `$PyPI` / `$Host`）：

| 镜像 | `-i` URL | `--trusted-host` |
|------|----------|------------------|
| 清华 | `https://pypi.tuna.tsinghua.edu.cn/simple` | `pypi.tuna.tsinghua.edu.cn` |
| 阿里 | `https://mirrors.aliyun.com/pypi/simple/` | `mirrors.aliyun.com` |
| 豆瓣 | `https://pypi.doubanio.com/simple/` | `pypi.doubanio.com` |

> `transformers 4.57.x` 会拉 `huggingface-hub<1.0`；若曾误装 `hub 1.x`，上面命令会自动降级到兼容版本。

### 2.3 Hugging Face 模型下载镜像（国内必设）

**PyPI 镜像只加速 `pip`，不加速模型权重。** 下 Qwen 权重请用下面任一方式。

#### 方式 A：环境变量（推荐，与 `chat_web` / 冒烟脚本通用）

在**当前 PowerShell 会话**里先设（再运行任何会 `from_pretrained` 的命令）：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
# 可选：加大超时、减少并发（网络不稳时）
$env:HF_HUB_DOWNLOAD_TIMEOUT = "600"

# 确认已生效（必须在本窗口 echo 出镜像地址，否则仍会连 huggingface.co 超时）
echo $env:HF_ENDPOINT
```

若日志里仍出现 `https://huggingface.co/...`，说明**本终端未设置** `HF_ENDPOINT`（新开 Anaconda Prompt 会丢）。  
**已有完整缓存时**：新版 `nanochat` 会自动用 `%USERPROFILE%\.cache\huggingface\hub\...` 离线加载；也可手写 snapshot 路径（见方式 B / 默认缓存）。

**永久生效（Windows）：**

1. `Win + R` → `sysdm.cpl` → **高级** → **环境变量**
2. 用户变量 → **新建**：变量名 `HF_ENDPOINT`，值 `https://hf-mirror.com`
3. 新开一个终端，执行 `echo $env:HF_ENDPOINT` 应显示镜像地址

验证能否访问镜像（可选）：

```powershell
curl.exe -I https://hf-mirror.com
```

#### 默认缓存位置（未指定 `--local-dir` 时）

用 **`Qwen/Qwen2.5-0.5B`** 或 `from_pretrained` 拉模型时，权重在**用户缓存**，不在 `D:\hf_models`：

```text
%USERPROFILE%\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots\<一串 hash>\
```

Windows 上可直接把 **`--hf-model-id`** 设为该 **snapshots 下含 `config.json` 的目录**（把 `<hash>` 换成你机器上的文件夹名）：

```powershell
# 查 snapshot 路径（PowerShell）
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B\snapshots" -Directory
```

或继续用 Hub 名（推荐，会自动用缓存、无需手写路径）：

```powershell
--hf-model-id Qwen/Qwen2.5-0.5B
```

#### 方式 B：先离线下好到固定目录，再指向本地

只有执行了带 **`--local-dir`** 的下载时，`D:/hf_models/...` 才会存在；否则不要填该路径。

```powershell
conda activate nanochat-qwen
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 预下载到指定目录（约 1GB；完成后该目录下应有 config.json）
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir D:/hf_models/Qwen2.5-0.5B
```

Serving 时把 Hub id 换成本地路径：

```powershell
python -m scripts.chat_web `
  --backend=transformers `
  --hf-model-id D:/hf_models/Qwen2.5-0.5B `
  --hf-max-layers 6 `
  --device-type cpu `
  --hf-max-context-len 1024 `
  --port 8003
```

**本地路径请用正斜杠 `D:/...`**，或确认目录存在且含 `config.json`。若报 `HFValidationError: Repo id must use alphanumeric chars...`，说明 Hub 把路径当成了仓库名——通常是**目录不存在**或路径写错；可先检查：

```powershell
Test-Path D:\hf_models\Qwen2.5-0.5B\config.json
```

#### 方式 C：ModelScope（HF 仍失败时）

```powershell
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn modelscope

python -c @"
from modelscope import snapshot_download
path = snapshot_download('Qwen/Qwen2.5-0.5B', cache_dir=r'D:\hf_models')
print(path)
"@
```

将打印的目录传给 `--hf-model-id <该目录>`（与方式 B 相同）。

#### 常见报错

| 现象 | 处理 |
|------|------|
| `Connection reset` / `timed out` | 确认已设 `HF_ENDPOINT`；用方式 B 先 `huggingface-cli download` |
| 仍访问 `huggingface.co` | 旧终端未加载环境变量 → **新开** PowerShell 或设用户级环境变量 |
| SSL / 代理 | 公司代理需配置 `HTTP_PROXY`/`HTTPS_PROXY`；或在家用网络 + 镜像 |

### 2.4 可选：缓存目录

```powershell
$env:HF_HOME = "$env:USERPROFILE\.cache\huggingface"
$env:NANOCHAT_BASE_DIR = "$env:USERPROFILE\.cache\nanochat"
```

---

## 3. 激活与项目路径

每次开新终端：

```powershell
conda activate nanochat-qwen
cd D:\WorkSpace\nanochat
$env:PYTHONPATH = (Get-Location).Path
```

验证 import：

```powershell
python -c "import torch; from nanochat.transformers_backend import TransformersChatBackend; print('ok', torch.__version__)"
```

---

## 4. 冒烟：6 层加载（不启动 Web）

```powershell
python -c @"
import torch
from nanochat.transformers_backend import TransformersChatBackend
b = TransformersChatBackend(
    'Qwen/Qwen2.5-0.5B',
    device=torch.device('cpu'),
    max_layers=6,
    max_context_len=1024,
)
n = len(b.model.model.layers)
print('layers', n)
out = b.generate_text([{'role':'user','content':'你好'}], max_new_tokens=32)
print('reply:', out[:200])
"@
```

首次运行会从 Hugging Face 下载模型，需联网。

---

## 5. Web Serving（与 Mac 相同参数）

```powershell
python -m scripts.chat_web `
  --backend=transformers `
  --hf-model-id Qwen/Qwen2.5-0.5B `
  --hf-max-layers 6 `
  --device-type cpu `
  --hf-max-context-len 1024 `
  --port 8003
```

#### 浏览器访问地址（Windows 实践记录）

终端出现 **`Server ready at http://localhost:8003`** 后再打开页面。

| 推荐 | 不推荐 / 易失败 |
|------|------------------|
| **`http://127.0.0.1:8003`** | `http://localhost:8003`（部分 Windows 会解析到 IPv6 `::1`，与服务监听不一致，表现为「无法访问网站」） |
| 端口与 `--port` 一致 | `https://...`（本地 uvicorn 默认仅 **http**） |

本机记录：**请用 `127.0.0.1`，不要只靠 `localhost`。**

自检（另开一个 PowerShell）：

```powershell
netstat -ano | findstr :8003
curl.exe http://127.0.0.1:8003/health
```

同步 API 调试：

```powershell
curl.exe -X POST http://127.0.0.1:8003/chat/completions_sync `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":64}"
```

---

## 8. WikiText continue Pre-Training（含周期性 eval）

`scripts/qwen_continue_pt.py` 支持 **train / eval 分 split**（`--preset wikitext` 时默认 `train` + **`validation`**）。

**依赖：** `datasets`、`accelerate>=0.26.0`、`peft`（见 §2.2）。

**训练方式（默认）：** **LoRA（PEFT）** —— 冻结 `Qwen2.5-0.5B` 底座，只训低秩适配器，在较好 Base 上**轻量走一遍**因果 LM（WikiText 等），不是全量微调。全量微调需显式加 **`--full-finetune`**。

| 输出目录内容 | 含义 |
|--------------|------|
| `adapter_config.json`、`adapter_model.safetensors` 等 | LoRA 权重（底座仍用 `--model-id`） |
| `tokenizer` 文件 | 与 Base 相同，便于复现 |
| `pt_run_summary.json` | 含 `peft: lora`、`trainable_params` 等 |

推理加载 LoRA 需 Base + adapter（`chat_web` 若未接 PEFT，可先用 `peft` 合并权重或后续再接 adapter 路径）。

### 8.0 机器运行时间测试：**Small's Win**（吞吐与 LoRA / 全量对比）

结构化备份：[`benchmarks/pt-grid-full/LuoYu_Win_pt_benchmark.json`](../benchmarks/pt-grid-full/LuoYu_Win_pt_benchmark.json)。

| 项目 | 数值 |
|------|------|
| CPU / 内存 | 12 逻辑核，约 15.7 GiB RAM |
| 模型 | `Qwen/Qwen2.5-0.5B`，截断 **6 层**，`device=cpu` |
| 数据集 | `wikitext-103-raw-v1` **train 共 1,801,350 行** |
| 短跑子集 | `--max-samples 800` → **197 packed blocks** → **50,432 token**（197×256） |

**对齐配置（两次短跑相同）：** `block=256`，`batch=4`，`grad_accum=8`，800 行，无 eval。

#### 全量微调 vs LoRA（实测）

| 指标 | 全量 `--full-finetune`（`run_0020`，30 step） | LoRA 默认（`lora-speed-5step`，5 step） | LoRA / 全量 |
|------|---------------------------------------------|----------------------------------------|-------------|
| `train_samples_per_second` | **0.335** | **0.281** | **≈ 0.84×（慢约 16%）** |
| **tokens/s**（×256） | **85.8** | **71.9** | **≈ 0.84×** |
| `train_steps_per_second` | 0.010 | 0.009 | ≈ 0.9× |
| **秒 / optimizer step** | **≈ 95.5**（2865÷30） | **≈ 113.7**（568.6÷5） | **≈ 1.19×（更慢）** |
| 短跑墙钟 | **2865 s（≈ 47.8 min）** | **568.6 s（≈ 9.5 min）** | 步数不同 |
| 外推 30 step 墙钟 | 2865 s | **≈ 3412 s（≈ 57 min）** | ≈ 1.19× |
| **可训练参数** | ~228M（≈ 100%） | **2.2M（≈ 0.97%）** | LoRA 省显存/磁盘 |
| checkpoint | 整模 ~0.7GB 级 | adapter 很小 | LoRA 省磁盘 |
| 学习率（本次） | `2e-5`（扫参默认） | `1e-4`（LoRA 常用） | 不影响墙钟对比 |

LoRA 汇总行（终端末尾）：

```text
{'train_runtime': 568.6003, 'train_samples_per_second': 0.281, 'train_steps_per_second': 0.009,
 'train_loss': 14.944007873535156, 'epoch': 0.8}
```

**为何本机 CPU 上 LoRA 更慢？** 瓶颈多在 **完整 6 层前向**；LoRA 在底座之外还要算 adapter 支路，**forward 更重**，而「只训 1% 参数」省下的 optimizer 时间占比很小。GPU + 大模型上 LoRA 更常因省显存/反向而划算；**LuoYu's Win 上若只比 tokens/s，全量反而略快**。

**选哪种：**

| 目标 | 建议 |
|------|------|
| 单位时间多过 token（CPU） | 全量 + 上表 batch 配置，或子集 + `max-steps` |
| 轻量走 PT、省内存/磁盘、只改 adapter | **默认 LoRA**（接受略慢） |
| 正式 WikiText 子集实验 | §8 下方 LoRA 命令（`batch=2, grad_accum=4`）；扫参最优 `4/8` 见全量列 |

#### 全量扫参最佳（`run_0020`）

| 参数 | 值 |
|------|-----|
| `--block-size` | `256` |
| `--per-device-train-batch-size` | `4` |
| `--gradient-accumulation-steps` | `8` |
| 需加 | **`--full-finetune`** |

#### 时间外推（仅 train，无 eval）

| 数据范围 | 全量 @ 85.8 tokens/s | LoRA @ 71.9 tokens/s |
|----------|----------------------|----------------------|
| 800 行（50,432 token） | **≈ 9.8 min** | **≈ 11.7 min** |
| 全库 ~10⁸ token（1 epoch） | **≈ 333 h（~14 天）** | **≈ 386 h（~16 天）** |

说明：800 行 ≠ 全库；全库以正式跑时 `Train packed blocks: N` 为准，\(T \approx N \times 256 / \text{tokens/s}\)。

**正式训练（默认 LoRA，子集 + 固定 step）：**

```powershell
cd D:\WorkSpace\nanochat
$env:PYTHONPATH = (Get-Location).Path
conda activate nanochat-qwen

python -m scripts.qwen_continue_pt `
  --model-id Qwen/Qwen2.5-0.5B `
  --max-layers 6 `
  --preset wikitext `
  --max-samples 5000 `
  --block-size 256 `
  --per-device-train-batch-size 2 `
  --gradient-accumulation-steps 4 `
  --lora-rank 16 `
  --lora-alpha 32 `
  --learning-rate 1e-4 `
  --max-steps 500 `
  --logging-steps 10 `
  --eval-steps 50 `
  --eval-max-samples 500 `
  --per-device-eval-batch-size 4 `
  --device-type cpu `
  --output-dir D:/WorkSpace/nanochat/checkpoints/qwen6-lora-wiki
```

全量微调：在同上命令末尾加 **`--full-finetune`**、`--learning-rate 2e-5`；若要扫参最优吞吐可试 `--per-device-train-batch-size 4 --gradient-accumulation-steps 8`（见 §8.0 对比表；**CPU 上略快于 LoRA，但更吃内存/磁盘**）。

输出：

| 文件 | 内容 |
|------|------|
| `trainer_state.json` | `log_history` 里的 **`loss`**（train）、**`eval_loss`**（eval） |
| `pt_run_summary.json` | 初始/最终 eval loss、train packed 块数等摘要 |

关闭 eval：`--no-eval`。自定义 eval split：`--eval-split test`。

**加速 eval：** 全量 validation 在 CPU 上可能 **单次 10–20 分钟**；务必加 `--eval-max-samples 500`、`--per-device-eval-batch-size 4`，并适当加大 `--eval-steps`（见上例）。

**独立 eval（只跑 eval，不训练）：** 用 `--eval-only` 开关，跳过 train 数据准备和 optimizer 创建，只加载模型 + eval 集跑一次 eval 并打印 wall-clock 时间。

```powershell

# 全量 validation eval
python -m scripts.qwen_continue_pt `
  --model-id Qwen/Qwen2.5-0.5B `
  --max-layers 6 `
  --preset wikitext `
  --eval-split validation `
  --block-size 256 `
  --per-device-eval-batch-size 4 `
  --device-type cpu `
  --eval-only
```

输出示例：
```text
Eval-only mode: 1125 blocks (288,000 tokens) | batch=4 | device=cpu | layers=6
Running eval (this may take a while on CPU) …
============================================================
  Eval loss          : 9.234567
  Perplexity         : 10246.78
  Wall-clock         : 425.3s (7.1 min)
  Samples/second     : 2.645
  Tokens/second      : 677.1
============================================================
```

**保存结果到 JSON 文件：**

```powershell
python -m scripts.qwen_continue_pt `
  --model-id Qwen/Qwen2.5-0.5B --max-layers 6 `
  --preset wikitext --eval-split validation `
  --eval-max-samples 500 --block-size 256 `
  --per-device-eval-batch-size 4 --device-type cpu `
  --eval-only `
  --eval-max-samples 200`
  --output-json ./eval_results/test_200.json
```

JSON 输出示例：
```json
{
  "model_id": "Qwen/Qwen2.5-0.5B",
  "max_layers": 6,
  "dataset": "Salesforce/wikitext",
  "dataset_config": "wikitext-103-raw-v1",
  "eval_split": "validation",
  "block_size": 256,
  "per_device_eval_batch_size": 4,
  "device_type": "cpu",
  "eval_blocks": 1125,
  "eval_tokens": 288000,
  "eval_loss": 9.234567,
  "perplexity": 10246.78,
  "wall_clock_sec": 425.3,
  "samples_per_second": 2.645,
  "tokens_per_second": 677.1
}
```

### 8.1 观测 CPU / 内存 占用

在**另一个终端或图形界面**观察；`qwen_continue_pt` 不内置资源日志。

#### Windows 本机（PowerShell 里跑训练时）

| 工具 | 用法 |
|------|------|
| **任务管理器** | `Ctrl+Shift+Esc` →「性能」看总 CPU/内存；「进程」找 `python.exe`，右键「将视图设置为」→「按逻辑处理器」可看是否**个别核打满** |
| **资源监视器** | 运行 `resmon` → CPU 页看每核曲线；内存页看提交量是否顶满 |
| **Process Explorer**（Sysinternals，可选） | 选中 `python.exe` 看 per-CPU 与工作集 |

另开 PowerShell 轮询（训练已启动、存在 `python` 进程时）：

```powershell
while ($true) {
  Get-Process python -ErrorAction SilentlyContinue |
    Select-Object Id, CPU,
      @{n='WS_GiB';e={[math]::Round($_.WorkingSet64/1GB,2)}}
  Start-Sleep -Seconds 5
}
```

#### htop

**htop 面向 Linux / macOS 终端，Windows 无官方原生版。**

| 环境 | 做法 |
|------|------|
| **WSL2** 或 Linux 服务器 | 训练在 Linux 里跑时，另开终端执行 `htop`，`F4` 过滤 `python`，`t` 切换每核条形图 |
| **Mac** | `brew install htop` 后 `htop`（见 `tutorial/mac_training_guide.md`） |
| **Windows 本机、想要类似 TUI** | 可用 **btop4win** 等第三方工具，或继续用任务管理器 / Process Explorer |

#### 怎么判断「是否吃满机器」（CPU 训练）

| 现象 | 含义 | 可尝试 |
|------|------|--------|
| 总 CPU 10–20%，但**个别逻辑 CPU 长期 100%** | 正常：PyTorch CPU 常只跑满少数核 | 算力已较满；要更快需 GPU 或更大 batch |
| 各核都很低且 step 很慢 | 可能在 eval、预处理，或 batch 太小 | 加大 train/eval batch；缩小 eval 集 |
| 内存接近物理上限或 python 工作集持续上涨 | 内存吃紧 | 减小 `block-size` / 样本数，或 `--gradient-checkpointing` |
| CPU、内存都明显有余量 | 还可加压 | 试更大 `per-device-train-batch-size`（注意 OOM） |

> 任务管理器里的 **CPU % 多为全核平均**；16 核上单核 100% 可能只显示约 6%。是否在算，要看**单核曲线**或 htop 里是否有列顶满。

**判断 batch 是否还能加大：** 在内存不 OOM 的前提下，对比日志里的 `train_samples_per_second`（越大越好），不要只看总 CPU % 或 `train_steps_per_second`。

### 8.2 最佳训练参数（速度最快）：扫参 benchmark 与结果分析

两套脚本分工：

| 脚本 | 作用 |
|------|------|
| `scripts/qwen_pt_benchmark.py` | **跑**网格：每组短训练（`--no-eval`、`--max-steps 30`），子进程带 **`--benchmark-no-save`**，不写 `model.safetensors` |
| `scripts/analyze_pt_benchmark.py` | **读**已有 `run_*/benchmark_stdout.log` + `pt_run_summary.json`，按 **`train_samples_per_second`** 排序，写出 `benchmark_analysis.json` |

扫参结束后目录里只需保留 **`benchmark_stdout.log`** 与 **`pt_run_summary.json`** 即可复现分析；checkpoint / tokenizer 等可删以省磁盘（每个 `model.safetensors` 约 0.7GB）。

#### 比速度看哪个指标？

一个 **optimizer step** 的工作量 ≈ `per_device_train_batch_size × gradient_accumulation_steps` 个 packed 样本（每条长 `block_size`）。因此：

| 指标（日志字段） | 回答的问题 |
|------------------|------------|
| **`train_samples_per_second`**（分析脚本里的 `samples_per_sec`） | **单位时间处理多少样本** → 扫参排序的**主指标** |
| **`train_steps_per_second`**（`steps_per_sec`） | **单位时间多少步更新** → 仅当你固定 `--max-steps`、要尽快跑满 step 数时优先 |
| `sec_per_10_steps` | 与 `--logging-steps 10` 对齐的墙钟时间，由 `steps_per_sec` 换算 |

跨 `block_size` 比吞吐时，可近似 **`tokens/s ≈ train_samples_per_second × block_size`**。

#### 小网格（约 6 组）

```powershell
cd D:\WorkSpace\nanochat
$env:PYTHONPATH = (Get-Location).Path
conda activate nanochat-qwen

python -m scripts.qwen_pt_benchmark `
  --output-dir D:/WorkSpace/nanochat/benchmarks/pt-grid `
  --grid quick `
  --max-layers 6 `
  --device-type cpu
```

#### 大网格（72 组，耗时长）

```powershell
# 先看组合列表（不训练）
python -m scripts.qwen_pt_benchmark `
  --output-dir D:/WorkSpace/nanochat/benchmarks/pt-grid-full `
  --grid full `
  --dry-run

# 试跑前 3 组
python -m scripts.qwen_pt_benchmark `
  --output-dir D:/WorkSpace/nanochat/benchmarks/pt-grid-full `
  --grid full `
  --max-layers 6 `
  --device-type cpu `
  --max-runs 3

# 完整大网格
python -m scripts.qwen_pt_benchmark `
  --output-dir D:/WorkSpace/nanochat/benchmarks/pt-grid-full `
  --grid full `
  --max-layers 6 `
  --device-type cpu
```

#### 分析已有 run（无需重新训练）

```powershell
python -m scripts.analyze_pt_benchmark --benchmark-dir benchmarks/pt-grid-full
```

输出：`benchmarks/pt-grid-full/benchmark_analysis.json`（含 `best`、`top10`、`all_runs`）。终端会打印按 **`samples_per_sec`** 选出的推荐 flags。

自定义网格：`--grid-json path/to/grid.json`，例如  
`{"per_device_train_batch_size":[1,2,4],"gradient_accumulation_steps":[4,8],"block_size":[256,512],"omp_num_threads":[null,8],"gradient_checkpointing":[false]}`。

### 8.4 快速Eval: `--eval-max-samples` 选择

**问题：** `--eval-max-samples` 设多少才够？设太小 loss 波动大不可信，设太大浪费时间。

**方法：** 固定 `--eval-seed` 跑多轮，对每个 `max_samples` 计算 **多次 eval loss 的 std**。若 std 足够小，则以此 `max_samples` 跑出的 loss 差值（如对比两个 checkpoint 或不同配置）在统计上是置信的。

**环境：** `Qwen/Qwen2.5-0.5B`，截断 6 层，`block_size=256`，`device=cpu`，数据集 `wikitext-103-raw-v1` validation split。

#### 运行 sweep（自动批量）

```powershell
cd D:\WorkSpace\nanochat
$env:PYTHONPATH = (Get-Location).Path
conda activate nanochat-qwen

python scripts/eval_samples_sweep.py `
  --samples 10,20,30,50,75 `
  --num-trials 5 `
  --base-seed 100 `
  --block-size 256 `
  --max-layers 6
```

每轮输出 JSON 到 `eval_results/sweep_n<max_samples>_s<seed>.json`。

#### 汇总结果

```powershell
python scripts/summary_for_user.py
```

#### 结果

| `--eval-max-samples` | n_trials | avg_eval_loss | std | avg_wall(s) |
|---|---|---|---|---|
| 10 | 10 | 15.5805 | 0.4822 | 12.3 |
| 20 | 10 | 15.5255 | 0.4841 | 23.1 |
| 30 | 10 | 15.4352 | 0.3291 | 38.6 |
| 50 | 10 | 15.4481 | 0.2322 | 73.2 |
| 75 | 10 | 15.4691 | 0.1496 | 116.2 |
| 100 | 7 | 15.4187 | 0.0925 | 333.2 |
| 150 | 3 | 15.4623 | 0.0629 | 279.5 |
| 750 | 1 | 15.5279 | — | 785.1 |
| 1000 | 1 | 15.5555 | — | 1143.0 |

> **注：** n=100 有 7 trial，n=150 有 3 trial（部分未完成）；n=750、1000 仅 1 trial 无 std。`avg_wall(s)` 为各 trial 的平均墙钟时间。

#### 解读与建议

- **std 即该 `max_samples` 下 eval loss 的噪声水平。** 若某方法让 eval loss 下降幅度 **> std**，则结果是置信的。
- **`--eval-max-samples 50`（std≈0.23，约 73 秒）推荐日常使用**：灵敏度足够（loss 下降 0.23 即可判定显著），耗时可接受。
- **`--eval-max-samples 75`（std≈0.15，约 116 秒）**：更稳定，适合最终对比。
- `max_samples=10~20`（std≈0.48，约 12~23 秒）：噪声太大，不推荐。
- `max_samples=30`（std≈0.33，约 39 秒）：勉强可用，快速冒烟可接受。
- `max_samples≥100`：std 很小但耗时陡增（n=100 反而 333s 是因为 block_size 变大导致 block 数暴增），性价比低。

**结论：** 日常开发 / 消融实验用 `--eval-max-samples 50`；最终报告 / 精密对比用 `--eval-max-samples 75`。

### 8.5 全量微调 + 定期 Eval（`--full-finetune`）

默认 `qwen_continue_pt` 使用 LoRA（只训练 ~2.2M 参数）。加上 `--full-finetune` 后训练全部 ~228M 参数，模型质量更高，但吞吐略慢（见 8.0 节 benchmark：LoRA ≈ 0.84× 全量吞吐）。

以下命令使用 `--eval-max-samples 100`（std≈0.09），每 50 步 eval 一次，eval loss 变化 ≥0.1 即为置信。

```powershell
cd D:\WorkSpace\nanochat
$env:PYTHONPATH = (Get-Location).Path
conda activate nanochat-qwen

python -m scripts.qwen_continue_pt `
  --model-id Qwen/Qwen2.5-0.5B `
  --max-layers 6 `
  --preset wikitext `
  --eval-max-samples 100 `
  --eval-steps 2 `
  --block-size 256 `
  --max-steps 10 `
  --learning-rate 2e-5 `
  --warmup-steps 50 `
  --per-device-train-batch-size 4 `
  --per-device-eval-batch-size 4 `
  --gradient-accumulation-steps 8 `
  --device-type cpu `
  --full-finetune `
  --output-dir D:/WorkSpace/nanochat/checkpoints/qwen6-ft-wiki-quick `
  --benchmark-no-save `
  --logging-steps 1 `
  > logs/qwen6-ft-wiki-quick.log 2>&1
```

**参数说明：**

| 参数 | 值 | 说明 |
|---|---|---|
| `--full-finetune` | ✓ | 训练全部参数（去掉即回退到 LoRA） |
| `--max-samples` | 5000 | 训练样本数 |
| `--eval-max-samples` | 100 | 每轮 eval 用 100 条（std≈0.09） |
| `--eval-steps` | 50 | 每 50 个 optimizer step 跑一次 eval |
| `--max-steps` | 500 | 总共 500 步，即 10 次 eval |
| `--per-device-train-batch-size` | 4 | 比默认 1 大，充分利用内存 |
| `--gradient-accumulation-steps` | 8 | 等效 batch=4×8=32 |

**输出：** `checkpoints/qwen6-ft-wiki/` 下：
- `trainer_state.json` — 每步的 `loss` 和每 50 步的 `eval_loss`
- `pt_run_summary.json` — 训练摘要（`peft: none`）

**想改回 LoRA：** 去掉 `--full-finetune` 并改 `--output-dir` 即可。


## 7. 常见问题

**浏览器「无法访问网站」**  
1. 是否已打印 **`Server ready at http://localhost:...`**（模型未加载完时访问会失败）。  
2. 是否使用 **`http://127.0.0.1:<端口>`** 而非 `localhost`（见第 5 节）。  
3. `netstat -ano | findstr :<端口>` 是否有 `LISTENING`。  
4. 是否在 `D:\WorkSpace\nanochat` 下启动（否则 UI 静态页可能 500）。

**`conda` 找不到**  
将 `C:\Users\<你>\miniconda3\Scripts` 加入系统 PATH，或始终用 `$env:USERPROFILE\miniconda3\Scripts\conda.exe` 全路径。

**下载慢 / 超时**  
设置 `HF_ENDPOINT` 镜像，或提前用 `huggingface-cli download Qwen/Qwen2.5-0.5B`。

**内存不足**  
保持 `--hf-max-context-len 1024`，关闭其它占内存程序；6 层 0.5B 在 CPU 上通常需约 2–4GB 量级 RAM（视是否缓存全量下载而定）。

**主 nanochat 环境与 qwen env 混用**  
训练自研 GPT 用主 env；Qwen 实验只用 `nanochat-qwen`，避免 `kernels` 与 `transformers` 抢 `huggingface-hub` 版本。

**continue PT：`Trainer` 要求 `accelerate>=0.26.0`；默认 LoRA 要求 `peft`**  
`pip install "accelerate>=0.26.0" "peft>=0.13.0"` 或按 §2.2 安装；见 §8。

**LoRA 与全量微调**  
默认只保存 **adapter**；`chat_web` 需能加载 adapter 或先合并权重。全量 checkpoint 用 `--full-finetune`。

**扫参占满磁盘**  
旧版 benchmark 会写 `model.safetensors`；请用当前脚本（自带 `--benchmark-no-save`），或只保留 `benchmark_stdout.log` / `pt_run_summary.json` 后删掉 `run_*/checkpoint-*` 与权重。用 §8.2 的 `analyze_pt_benchmark` 离线分析。

**Windows 与 Mac 差异**

| 项目 | Mac | Windows |
|------|-----|---------|
| 推荐设备 | `--device-type cpu`（MPS 易触发 4GB 限制） | `--device-type cpu`（无 MPS） |
| 环境变量 | `export PYTHONPATH=...` | `$env:PYTHONPATH = ...` |
| 多 GPU worker | 不适用 | 不适用（`num-gpus` 仅 CUDA） |
| 浏览器 URL | `http://localhost:...` 通常可用 | **建议 `http://127.0.0.1:<port>`**（避免 localhost→IPv6 问题） |
| `execution.py` 沙箱 | Darwin 有额外逻辑 | 非 Darwin 路径不同，与 transformers 推理无关 |