# 小模型

## 2. 开源小模型 GitHub 地图

按「能自己训 / 改」整理，**不是完整榜单**，而是社区常 fork、复现的一类。规模约 **几十 M ～ 3B**。

### 2.1 Karpathy 系（与 nanochat 最直接）

| 项目 | 链接 | 特点 |
|------|------|------|
| **nanochat** | https://github.com/karpathy/nanochat | 端到端：分词 → 预训练 → SFT → 评测；低成本追 GPT-2；miniseries / scaling laws |
| **nanoGPT** | https://github.com/karpathy/nanoGPT | 极简 GPT 预训练；作者建议新项目用 nanochat |
| **minGPT** | https://github.com/karpathy/minGPT | ~300 行教学实现；半归档 |
| **modded-nanogpt** | https://github.com/KellerJordan/modded-nanogpt | 124M GPT-2 规模「速通」优化（Muon、架构改动等） |
| **llm.c** | https://github.com/karpathy/llm.c | C/CUDA 训 GPT-2 124M；偏性能 / 系统 |
| **autoresearch** | https://github.com/karpathy/autoresearch | 自动搜索训练改动（nanochat 文档有提及） |

### 2.2 超小参数：从零训（几十 M，单卡友好）

| 项目 | 链接 | 规模 / 卖点 |
|------|------|-------------|
| **MiniMind** | https://github.com/jingyaogong/minimind | 26M～64M+；Tokenizer + PT + SFT + DPO/GRPO 等；中文社区活跃 |
| **LMFast** | https://github.com/2796gaurav/lmfast | Colab 上子 500M、QLoRA 等快速实验向 |

### 2.3 小但「正经」：约 100M～3B 全栈开源

| 项目 | 链接 | 特点 |
|------|------|------|
| **Lit-GPT** | https://github.com/Lightning-AI/litgpt | 多模型预训练 / 微调；TinyLlama 训练教程在此生态 |
| **TinyLlama** | https://github.com/jzhang38/TinyLlama | 1.1B、大规模预训练实践 |
| **SmolLM** | https://github.com/huggingface/smollm | HF 官方小模型训练与发布配套 |
| **SmolLM3** | https://huggingface.co/blog/smollm3 | |
| **OLMo / OLMo-core** | https://github.com/allenai/OLMo · https://github.com/allenai/OLMo-core | 1B 级全开放（数据、配置、步进 checkpoint） |
| **Pythia** | https://github.com/EleutherAI/pythia | 70M～12B；强调训练过程可解释 |
| **MiniPLM** | https://github.com/thu-coai/MiniPLM | 预训练阶段蒸馏 + 数据筛选；200M/500M/1.2B 等 |

### 2.4 从大模型「做小」：剪枝 / 蒸馏

| 项目 | 链接 | 特点 |
|------|------|------|
| **TiniLLM / prune_distill** | https://github.com/TiniLLM/prune_distill | Minitron 式剪枝 + 蒸馏；支持 Qwen/Llama/Phi 等教师 |

**HF 上的小模型权重**（Qwen2.5-0.5B、Phi、Gemma、Llama-3.2-1B 等）通常 **权重开放、完整预训练代码不完整**；训练尝试多落在上述第三方框架或官方微调示例。

### 2.5 推理 / 边缘部署（训练代码另找）

| 项目 | 链接 | 特点 |
|------|------|------|
| **gpt-fast** | https://github.com/meta-pytorch/gpt-fast | PyTorch 原生极速推理、量化、投机解码 |
| **llama.cpp** | https://github.com/ggerganov/llama.cpp | CPU/边缘量化推理事实标准 |

### 2.6 三类资源的区别（回顾用）

| 类型 | 典型来源 | 你能拿到什么 |
|------|----------|--------------|
| **训练框架** | nanochat、MiniMind、litgpt、OLMo | 完整或接近完整的 PT/SFT 代码 |
| **HF 权重** | Qwen、SmolLM、Gemma、Phi | 模型卡 + 推理/微调示例；完整 PT 少见 |
| **蒸馏/剪枝工具** | prune_distill、MiniPLM | 从大模型得到小学生的流程 |

---

## 3. 按目标选型

```text
你的目标
    │
    ├─ 从零训，追 GPT-2 成本/质量     → nanochat / modded-nanogpt / llm.c
    ├─ 从零训，几小时玩通全流程       → MiniMind（26M～64M）
    ├─ 1B 级论文式开放训练            → OLMo-core、smollm
    ├─ 不从头训，用现成小 HF 模型     → Qwen 截层（本仓库教程）、SmolLM、OLMo-1B
    ├─ 从大模型蒸馏/剪枝              → prune_distill、MiniPLM
    └─ 只优化推理延迟                 → gpt-fast、llama.cpp
```

| 你想做的事 | 建议先看 |
|------------|----------|
| 与 nanochat 一致：便宜训到 GPT-2 水平 | **nanochat**、**modded-nanogpt** |
| Windows 先跑通 serving（CPU） | [windows_qwen_setup.md](./windows_qwen_setup.md) + `transformers_backend` |
| Mac 上训练 / MPS 注意点 | [mac_training_guide.md](./mac_training_guide.md) |
| 中文、极小、全流程教学代码 | **MiniMind** |
| 1B 开放数据 + 训练配置 | **OLMo-core**、**smollm** |

---

## 4. 与本仓库 nanochat 的关系

### 4.1 仓库内关键路径

| 能力 | 位置 |
|------|------|
| 自研 GPT 训练 / SFT | 主环境 `uv sync`，见 README |
| HF Qwen 加载 + 截断层 | `nanochat/transformers_backend.py` |
| Web 对话 | `scripts/chat_web.py --backend=transformers` |
| Qwen 继续预训练（可选） | `scripts/qwen_continue_pt.py` |

### 4.2 环境隔离（Windows / Mac 共通思路）

- **主环境 `nanochat`**：`kernels` 等要求 `huggingface-hub>=1.10`
- **独立 env `nanochat-qwen`**：`transformers` 常要求 `huggingface-hub<1.0`  
- 详见 [windows_qwen_setup.md](./windows_qwen_setup.md) 第 2 节

### 4.3 Qwen 6 层实验要点

- 仍会**先下载完整 0.5B 权重**（约 1GB），再在内存保留前 N 层
- 未做 PT/SFT 时对话质量很差，仅用于验证「加载 + serving 链路」
- Windows 默认 `--device-type cpu` 最稳妥（无 MPS，也无 Mac Metal 4GB 限制）

### 4.4 README 中提到的延伸

- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420)
- [Beating GPT-2 for <<$100](https://github.com/karpathy/nanochat/discussions/481)
- 快速实验尺度：12 层（GPT-1 大小），见 `runs/miniseries.sh`

---

## 5. 后续可补充

讨论中未展开、可按需自行追加到本文档：

- [ ] 各仓库 **最低显卡 / 显存** 对照表
- [ ] **Star 数 / 最近提交** 快照（会过期，需定期更新）
- [ ] 是否含 **DPO / GRPO / 工具调用** 训练脚本
- [ ] 国内 **HF 镜像**（`HF_ENDPOINT=https://hf-mirror.com`）与缓存目录习惯
- [ ] 你个人 fork 或实验记录（checkpoint 路径、CORE 分数等）

---

## 修订记录

| 日期 | 内容 |
|------|------|
| 2026-05-20 | 初版：Conda Windows 速查 + 开源小模型地图 + 选型表 + nanochat 关联 |
