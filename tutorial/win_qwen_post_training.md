# Windows Qwen Post-Training 项目

## Goal

在 Windows CPU 环境下，以**最小成本**走完 LLM post-training 全流程（SFT → RL），基于 Qwen2.5-0.5B 模型进行快速实验。

核心理念：
- **小模型、快迭代**：使用 Qwen2.5-0.5B 并截断至 6 层，大幅降低计算需求
- **先评测、后训练**：每一步都有量化指标支撑，避免盲目训练
- **复用 nanochat 生态**：充分利用项目已有的 tasks/ 数据管线、chat_eval 评测框架、qwen_continue_pt.py 的 Trainer 骨架

## Background

### Qwen2.5-0.5B 官方模型

| 模型 | 说明 | 后训练流程 |
|------|------|-----------|
| `Qwen/Qwen2.5-0.5B` | Base 模型（18T tokens 预训练） | 无 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 指令微调模型 | SFT(100万+) → DPO(15万对) → GRPO |

官方 Instruct 模型已经经历了完整的 SFT+DPO+GRPO 三阶段后训练，是我们实验的参考上限。

### 硬件约束

- Windows CPU only（无 GPU）
- Qwen2.5-0.5B 全量（24层）在 CPU 上约 85 tokens/s
- 截断至 6 层可显著加速，适合快速实验

### 为什么从 Base 开始？

虽然官方 Instruct 已经很强大，但从 Base 开始做自己的 post-training 可以：
1. 完整理解 SFT/RL 每个阶段的实际效果
2. 实验不同的数据配方和训练策略
3. 形成可复现的低成本 post-training 方法论

## 项目阶段

### Phase 1: 评测体系搭建 ✅ 已完成

**目标**：建立 Qwen 模型的统一评测框架，获得 Base/Instruct 基线。

**产出**：
- `scripts/qwen_eval.py` — Qwen 专用评测脚本 ✅
  - 支持 base / sft / rl 三种模型类型（控制对话渲染方式）
  - 支持 6 个 Chat 评测任务（MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SpellingBee）
  - 支持任意层数截断（`--max-layers N`）
  - 适配 Qwen tokenizer 的 chat_template 到 nanochat task 接口
  - 关键组件：
    - `QwenTokenizerAdapter`: 包装 HF tokenizer，提供 `render_for_completion` / `encode` / `decode`
    - `QwenModelAdapter`: 包装 HF model，提供 `__call__` / `get_device`
    - `QwenEngine`: 轻量生成引擎，提供 `generate_batch`
    - `run_categorical_eval`: 多项选择评测（无需采样，CPU 友好）
    - `run_generative_eval`: 开放式生成评测（需要采样）
- `tutorial/win_qwen_post_training.md` — 项目规划文档 ✅
- 基线数据（待运行）：
  - `Qwen2.5-0.5B` (6层) 在 6 个 Chat 任务上的表现
  - `Qwen2.5-0.5B-Instruct` (6层) 在 6 个 Chat 任务上的表现

**使用方式**:

```shell
# 快速测试：Base 模型 6 层，只跑 MMLU 100 题（验证流程能跑通）
python -m scripts.qwen_eval `
  --model-id Qwen/Qwen2.5-0.5B `
  --mode base `
  --tasks MMLU `
  --max-problems 100 `
  --output-json eval_results/post-training/quick

python -m scripts.qwen_eval `
  --model-id Qwen/Qwen2.5-0.5B-Instruct `
  --mode sft `
  --tasks MMLU `
  --max-problems 100 `
  --output-json eval_results/post-training/instruct-quick.json

python -m scripts.qwen_eval `
--model-id Qwen/Qwen2.5-1.5B-Instruct `
--mode sft `
--tasks MMLU `
--max-problems 100

# 结果检查

## 检查 Instruct 模型在 MMLU 上的输出（sft mode），3 条样本
python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --tasks MMLU --num-examples 3

## 检查所有 6 个任务，2 条样本
python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --num-examples 2

## 检查 base 模型（base mode）
python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B --mode base --tasks MMLU --num-examples 3

## 检查生成任务（带采样）
python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --tasks GSM8K --num-examples 2 --temperature 0.8

## 6 层截断
python -m scripts.qwen_inspect_eval --model-id Qwen/Qwen2.5-0.5B-Instruct --mode sft --max-layers 6 --tasks MMLU --num-examples 3

# Phase 1: 评测 Base 模型（6层，categorical 任务）
python -m scripts.qwen_eval `
  --model-id Qwen/Qwen2.5-0.5B `
  --mode base `
  --max-layers 6 `
  --tasks MMLU,ARC-Easy,ARC-Challenge

  python -m scripts.qwen_eval `
  --model-id Qwen/Qwen2.5-0.5B `
  --mode base `
  --tasks MMLU,ARC-Easy,ARC-Challenge

# Phase 1: 评测 Instruct 模型（6层，全任务）
python -m scripts.qwen_eval `
  --model-id Qwen/Qwen2.5-0.5B-Instruct `
  --mode sft `
  --max-layers 6 `
  --tasks MMLU,ARC-Easy,ARC-Challenge,GSM8K,HumanEval,SpellingBee

python -m scripts.qwen_eval `
  --model-id Qwen/Qwen2.5-0.5B-Instruct `
  --mode sft `
  --tasks MMLU,ARC-Easy,ARC-Challenge,GSM8K,HumanEval,SpellingBee

# Phase 1: 评测 Base/Instruct 模型（全任务）

python -m scripts.qwen_eval `
  --model-id Qwen/Qwen2.5-0.5B `
  --mode sft `
  --tasks MMLU,ARC-Easy,ARC-Challenge,GSM8K,HumanEval,SpellingBee `
  --output-json eval_results/post-training/base-in-post-training

python -m scripts.qwen_eval `
  --model-id Qwen/Qwen2.5-0.5B-Instruct `
  --mode sft `
  --tasks MMLU,ARC-Easy,ARC-Challenge,GSM8K,HumanEval,SpellingBee `
  --output-json eval_results/post-training/quick
```
结论：在MMLU前100个任务重，Qwen2.5-0.5B-Instruct与Qwen2.5-0.5B-base相差较小（准确率为0.27/0.26），Qwen2.5-1.5B-Instruct效果为0.38.

### Phase 2: Qwen SFT

**目标**：从 Qwen2.5-0.5B Base 出发，用自己的数据做监督微调。

**数据划分原则**：训练数据和评测数据分离，避免"用考题当教材"。
- SFT 训练数据：SmolTalk（对话）、GSM8K（数学推理）、SpellingBee/SimpleSpelling（拼写）
- 留出评测集：MMLU、ARC-Easy、ARC-Challenge（纯知识/阅读理解，SFT 数据不包含它们）

**产出**：
- `scripts/qwen_sft.py` — Qwen SFT 训练脚本 ✅
  - 数据：通过 `--train-tasks` 指定，默认 SmolTalk,GSM8K,SpellingBee,SimpleSpelling
  - 支持 `--custom-json` 注入自定义 JSONL 对话数据
  - 数据格式：通过 Qwen `apply_chat_template` 渲染，prompt 部分 mask（labels=-1）
  - 训练方式：LoRA（默认）/ 全量微调（`--full-finetune`）
  - 复用 qwen_continue_pt.py 的 Trainer 骨架
  - 训练过程中通过 `--eval-tasks` 指定 ChatCORE 评测任务（默认 MMLU）
  - 支持 `--eval-only` 模式仅评测已训练的模型
- 对比分析：自己的 SFT vs 官方 Instruct 的差距

**使用方式**：

```powershell
$env:PYTHONPATH = (Get-Location).Path

# 默认 SFT（LoRA, SmolTalk+GSM8K+SpellingBee+SimpleSpelling, MMLU 作为 eval）
python -m scripts.qwen_sft `
  --model-id Qwen/Qwen2.5-0.5B `
  --max-layers 6 `
  --max-steps 500 `
  --output-dir ./out/qwen6-sft-lora-quick

# 自定义训练任务 + 更多 eval 任务
python -m scripts.qwen_sft `
  --model-id Qwen/Qwen2.5-0.5B `
  --train-tasks GSM8K `
  --max-steps 10 `
  --eval-tasks MMLU `
  --eval-max-problems 10 `
  --benchmark-no-save `
  --logging-steps 1 `
  --output-dir ./checkpoints/qwen6-sft-lora-quick

# 全量微调
python -m scripts.qwen_sft `
  --model-id Qwen/Qwen2.5-0.5B `
  --max-layers 6 `
  --train-tasks SmolTalk,GSM8K,SpellingBee,SimpleSpelling `
  --full-finetune `
  --max-steps 500 `
  --output-dir ./out/qwen6-sft-full

# Eval-only：仅评测已训练的模型
python -m scripts.qwen_sft `
  --model-id ./out/qwen6-sft-lora `
  --eval-only `
  --eval-tasks MMLU,ARC-Easy,ARC-Challenge `
  --eval-max-problems 200 `
  --output-json eval_results/post-training/sft-lora.json
```

### Phase 3: Qwen RL

**目标**：在 SFT 模型基础上做强化学习对齐。

**产出**：
- `scripts/qwen_rl.py` — Qwen GRPO 训练脚本
  - 任务：GSM8K + HumanEval + SpellingBee
  - 复用 chat_rl.py 的 GRPO 逻辑
  - 适配 Qwen tokenizer/模型接口
- 对比分析：SFT vs SFT+RL 的各指标变化

```shell
# 快速测试（10步，不保存，每步记录）
python -m scripts.qwen_rl `
  --model-id Qwen/Qwen2.5-0.5B-Instruct `
  --train-tasks GSM8K `
  --max-steps 10 `
  --logging-steps 1 `
  --num-samples 4 `
  --examples-per-step 2 `
  --device-batch-size 1 `
  --benchmark-no-save `
  --output-dir ./checkpoints/qwen6-rl-quick  > logs/qwen6-rl-lora-quick.log 2>&1

# 完整训练
python -m scripts.qwen_rl `
  --model-id ./out/qwen6-sft-lora `
  --train-tasks GSM8K,SpellingBee `
  --max-layers 6 `
  --max-steps 500 `
  --eval-tasks GSM8K,MMLU,ARC-Easy `
  --output-dir ./out/qwen6-rl

# Eval-only
python -m scripts.qwen_rl `
  --model-id ./out/qwen6-rl/final `
  --eval-only `
  --eval-tasks GSM8K,MMLU,ARC-Easy,ARC-Challenge,HumanEval,SpellingBee `
  --output-json eval_results/rl-final.json
```

### Phase 4: 总结

**目标**：形成完整的方法论文档。

**产出**：
- 数据消融分析（不同数据配比的效果）
- 训练成本记录（时间、内存、数据量）
- 小模型 post-training 最佳实践总结


## 关键设计决策

1. **Qwen 专用适配层**：不修改 chat_eval.py 核心逻辑，而是写 QwenTokenizerAdapter + QwenModelAdapter 包装层，将 Qwen 的 HF tokenizer/model 适配到 nanochat 接口
2. **chat_template 复用**：使用 Qwen 原生的 `apply_chat_template` 渲染对话，确保与 Qwen 训练时的格式一致
3. **渐进式实现**：先支持 categorical 任务（MMLU, ARC），因为不需要采样生成，CPU 上也能快速跑；再扩展 generative 任务

## Evaluation Metric

本项目涉及两类评测指标，它们测的是不同维度的能力，互补而非替代。

### WikiText eval_loss / Perplexity（`qwen_continue_pt.py`）

用于**继续预训练（CPT）阶段**，衡量模型的语言建模能力。

| 维度 | 说明 |
|------|------|
| **测什么** | 模型对 WikiText 文本的 next-token prediction loss |
| **指标** | `eval_loss` → `perplexity`（越低越好） |
| **数据** | WikiText-103 validation split（维基百科文章） |
| **方式** | 前向传播计算交叉熵 loss，不涉及生成 |
| **特点** | **平滑、低噪声**，适合追踪训练过程中模型的渐进变化 |
| **局限** | 只反映"预测维基文本"的能力，和下游任务表现可能不一致 |
| **适用场景** | CPT 训练过程中监控知识注入效果 |

### ChatCORE（`qwen_eval.py`）

用于 **SFT 和 RL 阶段**，衡量模型在对话格式下的下游任务能力。

| 维度 | 说明 |
|------|------|
| **测什么** | 模型在 6 个 Chat 基准任务上的表现 |
| **指标** | 各任务 accuracy + ChatCORE（综合分数，0~1，越高越好） |
| **任务** | MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SpellingBee |
| **方式** | categorical 任务用 logits argmax（无需采样）；generative 任务用采样解码 |
| **特点** | **直接反映下游能力**，用 Qwen chat_template 渲染，贴近实际使用场景 |
| **局限** | 比 loss 噪声大；需要对话格式，base 模型效果会明显差于 Instruct |
| **适用场景** | SFT/RL 阶段评估指令遵循和任务解决能力 |

### CORE metric（`base_eval.py --eval core`）

来自 [DCLM 论文](https://arxiv.org/abs/2406.11794) 的 22 任务 ICL 评测，用于评估 base 模型的通用知识/推理能力。

| 维度 | 说明 |
|------|------|
| **测什么** | 模型在 22 个下游任务上的 few-shot ICL 表现 |
| **指标** | `core_metric`（0~1，越高越好，GPT-2 1.6B 为 0.2565） |
| **方式** | few-shot In-Context Learning（多项选择 / Schema / 语言建模） |
| **特点** | 综合性强，nanochat 项目的主要 base 模型评价指标 |
| **局限** | 对 Instruct 模型不适用（Instruct 被训练为对话模型，不擅长 ICL 续写）；噪声比 loss 大 |
| **适用场景** | Base 模型的最终能力评估 |

### 各阶段推荐指标

| 阶段 | 推荐指标 | 原因 |
|------|---------|------|
| **CPT（继续预训练）** | WikiText eval_loss / perplexity | 平滑、低噪声，适合追踪知识注入效果 |
| **SFT** | ChatCORE（qwen_eval.py） | SFT 后 loss 会上升（正常现象），下游任务 accuracy 才是真正关心的 |
| **RL** | ChatCORE（qwen_eval.py） | RL 改变偏好/推理模式，loss 完全无意义 |

> 参考 nanochat 项目的观察："The `val_bpb` number is a great, smooth metric to track relative performance and has less noise than CORE."

## Reference

- [Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115)
- [DCLM 论文 (CORE metric)](https://arxiv.org/abs/2406.11794)