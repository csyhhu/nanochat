# Windows：Qwen2.5-0.5B（6 层）加载与 Serving

> 延续 Mac 上 `tutorial/mac_training_guide.md` 第 7 节 **Hugging Face / transformers 后端** 实验。  
> Windows **无 MPS**，`--device-type cpu` 是默认稳妥路径，且**不会**遇到 Mac 上 Metal 临时张量 >4GB 的限制。

---

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

# 3) 可选：继续预训练
python -m pip install -i $PyPI --trusted-host $Host "datasets>=4.0.0"

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

## 6. Windows 与 Mac 差异

| 项目 | Mac | Windows |
|------|-----|---------|
| 推荐设备 | `--device-type cpu`（MPS 易触发 4GB 限制） | `--device-type cpu`（无 MPS） |
| 环境变量 | `export PYTHONPATH=...` | `$env:PYTHONPATH = ...` |
| 多 GPU worker | 不适用 | 不适用（`num-gpus` 仅 CUDA） |
| 浏览器 URL | `http://localhost:...` 通常可用 | **建议 `http://127.0.0.1:<port>`**（避免 localhost→IPv6 问题） |
| `execution.py` 沙箱 | Darwin 有额外逻辑 | 非 Darwin 路径不同，与 transformers 推理无关 |

---

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

---

## 8. WikiText continue PT（含周期性 eval）

`scripts/qwen_continue_pt.py` 支持 **train / eval 分 split**（`--preset wikitext` 时默认 `train` + **`validation`**）。

```powershell
cd D:\WorkSpace\nanochat
$env:PYTHONPATH = (Get-Location).Path
conda activate nanochat-qwen

python -m scripts.qwen_continue_pt `
  --model-id Qwen/Qwen2.5-0.5B `
  --max-layers 6 `
  --preset wikitext `
  --max-samples 5000 `
  --block-size 512 `
  --max-steps 500 `
  --logging-steps 10 `
  --eval-steps 50 `
  --device-type cpu `
  --output-dir D:/WorkSpace/nanochat/checkpoints/qwen6-pt-wiki
```

输出：

| 文件 | 内容 |
|------|------|
| `trainer_state.json` | `log_history` 里的 **`loss`**（train）、**`eval_loss`**（eval） |
| `pt_run_summary.json` | 初始/最终 eval loss、train packed 块数等摘要 |

关闭 eval：`--no-eval`。自定义 eval split：`--eval-split test`。

---

## 9. 下一步（线索 B）

1. 本页 4–5 节验收通过 → 2. `qwen_continue_pt`（6 层 + WikiText + eval）→ 3. 24L Base 同 eval 集 loss（待 `qwen_eval_lm_loss`）→ 4. SFT → 5. serving 加速。
