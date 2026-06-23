# NanoLLM RL 训练教程

## 概述

本教程将指导你使用 NanoLLM 框架运行强化学习（RL）训练。我们将使用 GRPO（Group Relative Policy Optimization）算法在推理任务上微调语言模型。

## 前置要求

- Python 3.8+ 已安装，且安装了 PyTorch
- 已安装 NanoLLM 包（在项目根目录运行 `pip install -e .`）
- 至少 8GB RAM（CPU 训练）或 4GB+ 显存的 GPU
- HuggingFace 账号和 token（用于下载模型）

## 理解 RL 训练

### 什么是 RL 训练？

与监督微调（SFT）不同，SFT 有真实标签（ground truth），RL 训练使用 **reward（奖励）** 来指导学习：

```
SFT: Prompt → [标签: "答案是42"] → 交叉熵损失
RL:  Prompt → [生成回复] → Reward (0.0 或 1.0) → Policy Gradient 损失
```

### GRPO 算法

GRPO（Group Relative Policy Optimization）的工作流程如下：

1. 对每个 prompt 生成 **G 个 rollout**（例如 G=8）
2. 计算每个 rollout 的 **reward**
3. 计算 **group relative advantage**：`A_i = reward_i - mean(rewards)`
4. 使用 **policy gradient** 更新模型：`loss = -mean(log_prob * advantage)`

### 核心概念

| 概念 | 解释 |
|------|------|
| **Rollout** | 模型根据给定的 prompt 生成完整回复 |
| **Reward** | 评分（0.0-1.0），表示回复质量 |
| **Advantage** | 相对于组平均的改进程度 |
| **Policy Gradient** | 增加高 reward 回复的概率的更新 |

## 快速开始：运行简单的 RL 训练

完整的 RL 训练流程包括：**训练前评估（基线） → 训练（含自动评估） → 训练后评估（对比）**。

本教程将分为以下步骤：
1. **步骤 1：准备环境**
2. **步骤 2：运行 RL 训练（含训练前后评估）**
   - 2.1 训练前评估（获取基线）
   - 2.2 运行 RL 训练（含自动评估）
   - 2.3 训练后手动评估（详细对比）
3. **步骤 3：监控训练**

### 步骤 1：准备环境

```powershell
# 进入项目根目录
cd d:\WorkSpace\nanochat

# 设置 PYTHONPATH
$env:PYTHONPATH = (Get-Location).Path

# （可选）设置 HuggingFace 缓存目录
$env:HF_HOME = "D:\huggingface_cache"
```

### 步骤 2：运行 RL 训练（含训练前后评估）

完整的 RL 训练流程包括：**训练前评估（基线） → 训练（含自动评估） → 训练后评估（对比）**。

#### 2.1 训练前评估（获取基线）

在训练之前，先评估模型在任务上的表现，作为基线：

```powershell
# 评估 Base 模型（训练前）
python -m nanollm.main `
  --stage rl `
  --eval-only `
  --model-id Qwen/Qwen2.5-0.5B `
  --eval-tasks GSM8K,SpellingBee `
  --eval-max-problems 100 `
  --output-json ./eval_results/rl_baseline.json
```

**关键点**：
- 使用 `--eval-only` 只运行评估，不训练
- 保存结果到 JSON 文件，方便后续对比
- Base 模型在 GSM8K 上表现很差（pass@1 ≈ 5%），这正常

#### 2.2 运行 RL 训练（含自动评估）

现在运行 RL 训练。训练过程中会自动运行评估：

```powershell
python -m nanollm.main `
  --stage rl `
  --model-id Qwen/Qwen2.5-0.5B `
  --train-tasks GSM8K `
  --max-steps 2 `
  --rl-num-samples 4 `
  --rl-examples-per-step 2 `
  --rl-temperature 1.0 `
  --eval-tasks MMLU `
  --eval-max-problems 1 `
  --eval-steps 2 `
  --logging-steps 1 `
  --output-dir ./checkpoints/nanollm/qwen6-rl-gsm8k-quick 2>&1 | Tee-Object -FilePath logs/qwen6-rl-gsm8k-quick.log
```

**关键参数**：
- `--eval-steps 50`：每 50 步运行一次评估
- `--eval-tasks`：指定评估任务
- `--eval-max-problems`：每个任务评估多少样本

#### 2.3 训练后手动评估（详细对比）

训练完成后，可以运行更详细的评估，对比训练前后的效果：

```powershell
# 评估训练后的模型
python -m nanollm.main `
  --stage rl `
  --eval-only `
  --model-id ./out/qwen6-rl-gsm8k/final `
  --eval-tasks GSM8K,SpellingBee,MMLU,ARC-Easy `
  --eval-max-problems 200 `
  --output-json ./eval_results/rl_trained.json

# 对比基线
echo "=== Baseline (Before Training) ==="
Get-Content ./eval_results/rl_baseline.json | ConvertFrom-Json | Format-Custom

echo "=== Trained (After Training) ==="
Get-Content ./eval_results/rl_trained.json | ConvertFrom-Json | Format-Custom
```

**关键观察**：
- ✅ GSM8K 显著提升（训练任务）
- ⚠️ SpellingBee 提升不大（未训练的任务）
- 💡 如果想提升多个任务，使用多任务训练（`--train-tasks GSM8K,SpellingBee`）

### 步骤 3：监控训练

打开另一个终端来监控训练：

```powershell
# 查看训练日志（实时更新）
Get-Content ./logs/qwen6-rl-gsm8k.log -Wait

# 或者检查输出目录
ls ./out/qwen6-rl-gsm8k
```

## 详细配置

### 训练参数

| 参数 | 默认值 | 解释 |
|------|--------|------|
| `--rl-num-samples` | 8 | 每个 prompt 生成的 rollout 数（GRPO 中的 G） |
| `--rl-examples-per-step` | 4 | 每步处理的 prompt 数 |
| `--rl-temperature` | 1.0 | 采样温度（越高 = 越多样） |
| `--rl-device-batch-size` | 2 | 每次前向传播的最大序列数（避免 OOM） |
| `--rl-reward-eval-every` | 50 | 每 N 步运行一次基于 reward 的评估 |
| `--rl-reward-eval-examples` | 100 | reward 评估的样本数 |

### 模型配置

| 参数 | 默认值 | 解释 |
|------|--------|------|
| `--model-id` | Qwen/Qwen2.5-0.5B | 要训练的模型（Hub ID 或本地路径） |
| `--max-layers` | None | 截断模型为 N 层（加速训练） |
| `--max-seq-len` | 512 | 最大序列长度 |
| `--torch-dtype` | float32 | 模型精度（float32/float16/bfloat16） |
| `--device-type` | cpu | 训练设备（cpu/cuda/mps） |

### LoRA 配置

RL 训练默认使用 LoRA（参数高效微调）：

| 参数 | 默认值 | 解释 |
|------|--------|------|
| `--no-lora` | False | 禁用 LoRA（全量微调） |
| `--lora-rank` | 8 | LoRA 秩（越高 = 更多参数） |
| `--lora-alpha` | 16 | LoRA 缩放因子 |
| `--lora-dropout` | 0.05 | LoRA 层的 dropout |

## 在不同任务上训练

### 任务 1：GSM8K（数学应用题）

**Reward 函数**：0.0（答案错误）或 1.0（答案正确）

```powershell
python -m nanollm.main `
  --stage rl `
  --model-id Qwen/Qwen2.5-0.5B `
  --train-tasks GSM8K `
  --max-steps 500 `
  --rl-num-samples 8 `
  --eval-tasks GSM8K `
  --eval-max-problems 200 `
  --output-dir ./out/qwen6-rl-gsm8k
```

### 任务 2：SpellingBee（字母计数）

**Reward 函数**：0.0（计数错误）或 1.0（计数正确）

```powershell
python -m nanollm.main `
  --stage rl `
  --model-id Qwen/Qwen2.5-0.5B `
  --train-tasks SpellingBee `
  --max-steps 500 `
  --rl-num-samples 8 `
  --eval-tasks SpellingBee `
  --eval-max-problems 200 `
  --output-dir ./out/qwen6-rl-spelling
```

### 任务 3：多任务训练

同时在多个任务上训练：

```powershell
python -m nanollm.main `
  --stage rl `
  --model-id Qwen/Qwen2.5-0.5B `
  --train-tasks GSM8K,SpellingBee `
  --max-steps 1000 `
  --rl-num-samples 8 `
  --eval-tasks GSM8K,SpellingBee,MMLU `
  --eval-max-problems 200 `
  --output-dir ./out/qwen6-rl-multi
```

## 理解训练输出

### 训练日志

训练期间，你会看到如下日志：

```
Step 10/500 | Epoch 1
  Rewards: [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
  Mean Reward: 0.375
  Advantages: [0.625, -0.375, 0.625, -0.375, -0.375, 0.625, -0.375, -0.375]
  Mean Advantage: 0.000
  Loss: -0.125
  Perplexity: 15.2
```

**关键指标**：

| 指标 | 解释 |
|------|------|
| `Rewards` | 每个 rollout 的 reward（0.0 或 1.0） |
| `Mean Reward` | rollout 的平均 reward |
| `Advantages` | 相对 advantage（reward - mean） |
| `Loss` | Policy gradient 损失（负值因为我们要最大化） |
| `Perplexity` | 模型置信度（越低 = 越自信） |

### 评估结果

每 `--eval-steps` 步，会运行评估：

```
[Eval] Step 50
  GSM8K_pass@1: 0.250
  GSM8K_pass@4: 0.500
  SpellingBee_pass@1: 0.300
  SpellingBee_pass@4: 0.550
```

**pass@k 指标**：k 个样本中至少 1 个正确的概率。

#### 🚀 CPU 优化（无 GPU 用户专用）

如果你没有 GPU，NanoLLM 提供了多种 CPU 优化选项，可以显著提升训练速度（最高 5-10x）：

**优化 1：批量生成（已默认启用）**

NanoLLM 现在使用批量生成替代逐个生成，速度提升 3-10x：

```powershell
# 无需额外配置，已默认启用
python -m nanollm.main `
  --stage rl `
  --model-id Qwen/Qwen2.5-0.5B
```

**优化 2：动态量化（2-4x 加速）**

将模型权重量化为 int8，减少内存带宽压力：

```powershell
python -m nanollm.main `
  --stage rl `
  --model-id Qwen/Qwen2.5-0.5B `
  --rl-quantize-cpu  # 启用动态量化
```

**优化 3：torch.compile()（PyTorch 2.0+）**

编译模型以优化计算图：

```powershell
python -m nanollm.main `
  --stage rl `
  --model-id Qwen/Qwen2.5-0.5B `
  --rl-compile-model  # 启用 torch.compile
```

**优化 4：组合使用（推荐）**

```powershell
# 全部启用（最快速度）
python -m nanollm.main `
  --stage rl `
  --model-id Qwen/Qwen2.5-0.5B `
  --rl-quantize-cpu `
  --rl-compile-model `
  --rl-num-samples 4 `  # 减少 rollout 数量
  --rl-examples-per-step 2  # 减少每步样本数
```

**预期加速效果**：

| 优化方案 | 预期加速 | 说明 |
|---------|---------|------|
| 批量生成（默认） | 3-10x | 已默认启用，无需配置 |
| + 动态量化 | 2-4x | 模型大小减半，推理更快 |
| + torch.compile | 1.5-2x | 首次运行慢，后续快 |
| **组合使用** | **5-20x** | 推荐无 GPU 用户使用 |

**注意**：
- 动态量化会稍微降低模型精度（但通常影响不大）
- torch.compile() 需要 PyTorch 2.0+，且首次运行会慢（编译开销）
- 如果使用 CPU 优化后仍然太慢，建议换用更小的模型（Qwen2.5-0.5B）或减少 `rl_num_samples`

### 问题 3：Reward 总是 0（模型没学到）

**可能原因**：
1. 模型太小，无法解决任务
2. Reward 函数太严格
3. 温度太高（太随机）

**解决方案**：

```powershell
# 1. 使用更大的模型或更多层
--model-id Qwen/Qwen2.5-1.5B
--max-layers 12

# 2. 使用 soft reward（部分得分）
# 编辑 tasks/gsm8k.py 给数字答案 0.2 分

# 3. 降低温度
--rl-temperature 0.6
```

### 问题 4：训练损失为 NaN

**可能原因**：
1. 学习率太高
2. Advantage 没有归一化

**解决方案**：

```powershell
# 1. 降低学习率
--learning-rate 1e-5

# 2. 使用 advantage 归一化（编辑 rl_trainer.py）
advantages = self.rollout_generator.compute_advantages(
    rewards, method="grpo", normalize=True
)
```

## 后续步骤

完成本教程后，你可以：

1. **在不同任务上实验**：尝试 HumanEval（代码生成）
2. **实现自定义 reward 函数**：为你的特定用例设计 reward
3. **分析训练动态**：绘制 reward 曲线、损失曲线
4. **结合 SFT 和 RL**：先在 GSM8K 上 SFT，然后 RL 微调
5. **扩展规模**：使用 GPU 在更大的模型（Qwen2.5-1.5B）上训练

## 其他资源

- GRPO 论文：[arXiv:2402.01652](https://arxiv.org/abs/2402.01652)
- NanoLLM 文档：`nanollm/README.md`
- 任务实现：`tasks/gsm8k.py`、`tasks/spellingbee.py`
- RL 训练器代码：`nanollm/trainers/rl_trainer.py`
