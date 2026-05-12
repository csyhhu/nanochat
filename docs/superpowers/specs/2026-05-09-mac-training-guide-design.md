# Design: Mac 训练指南（tutorial/mac_training_guide.md）结构化更新

**日期**: 2026-05-09  
**状态**: 已采纳并落入教程正文  
**关联文档**: `tutorial/mac_training_guide.md`

---

## 1. 背景与问题

用户在 Apple Silicon Mac（典型约束：**16GB 统一内存、MPS、无 Flash Attention 3**）上运行 nanochat 的 **预训练 → SFT →（可选）RL → Serving** 闭环时，容易遇到：

- **OOM**：沿用 GPU/大内存机器的 `device-batch-size` / `max-seq-len` 会爆内存。
- **观测缺失**：`--run=dummy` 禁用 W&B，误以为没有 loss；终端与 `tee` 日志才是默认观测手段。
- **阶段不一致**：预训练已缩小，SFT 仍使用大 batch / 长上下文，覆盖继承自 checkpoint 的安全默认值，导致后续阶段再次 OOM 或行为不一致。
- **验收模糊**：缺少与硬件约束匹配的「baseline 合格线」，难以判断闭环是否成功。

本 spec 定义教程应如何组织，使读者按 **单一主线（16GB baseline）** 可走通全流程，并保留 **可选高配** 与 **设计溯源**。

---

## 2. 目标与非目标

### 2.1 目标

- 在教程中明确 **主推荐路径**：16GB 可用的预训练超参（已与 `scripts/base_train.py` 约束一致：`window-pattern=L`、`total_batch_size` 整除等）。
- **SFT 与预训练对齐**：显式给出与 `mac-baseline-d4` 一致的 `max-seq-len` / batch，或说明「从 base checkpoint 继承」时的行为与必填 `--model-tag`。
- 记录 **loss / 指标** 的可观测位置（终端、W&B、`tee`、非时序的 `meta_*.json`）。
- 增加 **可选 RL** 小节：保守超参 + 「易 OOM 可跳过」说明，避免读者默认跑满 `chat_rl.py` 的 GPU 向默认值。
- 增加 **baseline 验收清单**（流程 + 最低现象标准，不承诺模型质量对标云端大模型）。
- 在教程顶部 **链接本 spec**，满足 superpowers brainstorming「设计落地可追溯」。

### 2.2 非目标

- 不在此 spec 中规定 Qwen tokenizer/权重迁移（后续独立 spec）。
- 不规定 GGUF/llama.cpp/MLX 导出（后续独立 spec）。
- 不修改 `scripts/*.py` 训练逻辑；仅文档与命令组合层面约束。

---

## 3. 方案比较


| 方案        | 做法                                                                          | 优点                 | 缺点                                       |
| --------- | --------------------------------------------------------------------------- | ------------------ | ---------------------------------------- |
| **A（推荐）** | 教程分 **主路径（16GB baseline）** + **可选大内存/DSV4**；SFT/RL 给 **对齐命令**；增加验收与 spec 链接 | 读者最少决策、OOM 最少、闭环清晰 | 正文变长，需维护多档命令                             |
| **B**     | 仅保留预训练 16GB 段，SFT/RL 写「自行按 OOM 减半」                                          | 篇幅短                | 读者仍易误用 512/大 batch，继承行为不直观               |
| **C**     | 拆成两篇文档（mac-baseline.md / mac-advanced.md）                                   | 单篇更短               | 增加导航成本；用户明确要求改单一 `mac_training_guide.md` |


**决议**: 采用 **方案 A**，在单一 `mac_training_guide.md` 内用层级标题区分主路径与可选内容。

---

## 4. 设计决议（教程信息架构）

### 4.1 文档开头

- 增加简短 **「设计说明」**：目标读者、主路径名称（16GB baseline）、指向本 spec 的相对路径 `docs/superpowers/specs/2026-05-09-mac-training-guide-design.md`。

### 4.2 预训练（已有）

- 保持现有 **16GB baseline** 命令块为首要预训练示例。
- 保留 loss/W&B/`tee` 说明表。

### 4.3 SFT

- **主路径**：`--model-tag=mac-baseline-d4`（与预训练一致）；**显式** `--max-seq-len=256 --device-batch-size=4 --total-batch-size=4096`，避免无意中继承错误 tag 或用户误用旧 checkpoint。
- 说明：若 `max_seq_len` 等与 pretrain 一致，亦可依赖 meta 继承，但教程主路径以 **显式写出** 降低歧义。
- `num-iterations`：使用 **数百～一千量级的正整数** 作为「可观察 val loss / 对话改善」的下限；**禁止**再将 `1` 作为主示例（仅可作为冒烟脚注）。
- `eval-tokens`：16GB 上适当减小（如 `131072`），降低 eval 峰值。
- 提供 `tee` 示例与 W&B 说明（与 pretrain 同源）。

### 4.4 RL（可选）

- 明确 **RL 非 16GB 闭环必选项**；默认 `chat_rl` 的 batch/采样对内存不友好。
- 给出 **保守示例**：降低 `device-batch-size`、`examples-per-step`、`num-samples`、`max-new-tokens`；`--run=dummy` 与 `tee` 可选。
- 若脚本或依赖在 MPS 上仍有 sharp edge，用语上保持 **「实验性」**，并指向 issues/上游。

### 4.5 Serving

- 保持现有 `chat_web` / `chat_cli` / curl；补充 **与 SFT `max-seq-len` 的关系**（推理侧过长上下文仍受模型训练长度影响，非自动魔法）。

### 4.6 验收标准（Baseline Definition）

- **流程**：数据 → tokenizer → base_train → chat_sft → chat_web 可启动。
- **产物路径**：SFT checkpoint 位于 `chatsft_checkpoints/`（与 `load_model("sft", ...)` 一致），**不是** `sft_checkpoints/`。
- **现象**：预训练 `loss` 或 `val_bpb` 在短程训练中可见趋势（非严格数值）；SFT 后 CLI/Web 可完成多轮对话不立即崩溃；无持续 OOM。
- **不承诺**：具体 CORE/ChatCORE 分数或「像 ChatGPT」。

### 4.7 资源监控

- 增加简短命令：`Activity Monitor`、`top`、`memory_pressure`、`vm_stat`、`df -h`（与工程相关、用户曾问）。

---

## 5. 测试与自检（文档层面）

- 命令块中 **batch 整除关系** 人工核对：`total_batch_size % (device_batch_size * max_seq_len) == 0`（单机单卡）。
- 内部链接：设计说明 ↔ spec 路径、目录锚点更新。
- **Spec 自检**（brainstorming 要求）：
  - 无未解析的 TBD/TODO。
  - 与教程中 DSV4「16GB 可能 OOM」表述一致。
  - 范围仅限文档；代码行为以仓库现状为准。

---

## 6. 维护说明

- 若 `scripts/chat_sft.py` / `chat_rl.py` CLI 变更，优先更新本 spec 的「设计决议」与教程命令块。
- 若新增「官方 Mac 最小 smoke」脚本（未来），可在 spec 中增加「单一入口脚本」一节并 deprecate 长命令块。

