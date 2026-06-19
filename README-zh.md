# ImportanceGradient

一个面向大语言模型后训练的 importance-aware 稀疏梯度同步引擎。

ImportanceGradient 来源于一个后训练过程中的经验观察：即使已经通过 Top-k 或 ADTopk 等常规稀疏梯度选择方法过滤掉大部分 dense gradients，剩余 sparse gradients 中仍有相当一部分对最终模型质量贡献很小。这些低重要性梯度不必在每一个训练迭代中都同步。基于这一观察，ImportanceGradient 设计了一套 **重要性感知的梯度同步系统**，在保持模型精度的同时减少稀疏梯度通信开销。

## 核心思想

ImportanceGradient 采用离线-在线两阶段设计：

### 1. 离线 profiling 阶段

在离线阶段，ImportanceGradient 对保留下来的 sparse gradients 进行 profiling，并构建组件级同步知识，包括：

- 组件重要性分数
- 自适应重要性阈值
- 重要组件集合和低重要性组件集合

该阶段会将 retained sparse gradients 映射到 Transformer 组件，并结合梯度幅值和跨迭代一致性，识别哪些组件应保留在每轮同步路径上。

### 2. 在线同步阶段

在在线阶段，ImportanceGradient 应用重要性感知同步策略：

- 重要组件梯度每轮同步
- 低重要性组件梯度延迟同步，并累积到 residual buffers
- 被延迟的 residuals 按周期释放
- 释放出的同步 payload 在 collective communication 前进行 balanced bucket packing

因此，ImportanceGradient 并不是简单丢弃低重要性梯度，而是延迟并保留其更新，从而降低同步负载并保持训练质量。

## 仓库内容

本仓库包含 ImportanceGradient 的核心实现和实验 workflow，包括：

- Transformer 组件映射和 sparse-gradient importance profiling
- 面向低重要性组件的 residual-preserving 周期同步
- balanced bucket planning 和真实 `torch.distributed.all_reduce` 通信路径
- 基于 ADTopk 的端到端训练 workflow
- RQ1-RQ5 实验脚本：重要性稳定性、阈值划分消融、释放频率消融、packing 消融、端到端评估
- 面向核心 CPU 逻辑的轻量测试

## 主要特性

- 面向 LLM 后训练的 importance-aware sparse-gradient synchronization
- 离线组件重要性 profiling
- 自适应重要/低重要性组件划分
- 低重要性梯度的 residual compensation
- released payload 的 balanced packing
- 兼容 PyTorch distributed 和 DeepSpeed-style 训练 workflow
- 提供论文实验复现脚本

## 实验结果

在代表性 LLM 后训练 workload 上，ImportanceGradient 的目标是：

- 传输约为 baseline sparse-gradient traffic 的 **四分之一**
- 降低 synchronization 和 all-reduce 时间
- 保持 validation loss 和任务 accuracy
- 提升端到端训练 throughput

本仓库提供用于复现论文 RQ 实验的脚本。

## 安装和测试

```bash
git clone https://github.com/rice379/importance-gradient.git
cd importance-gradient

pip install -e .
pip install -r requirements.txt

# 轻量测试，不需要分布式 GPU 环境
pytest tests
```

## 实验 workflow

主要实验目录如下：

```text
experiments/rq1_workload_stability   # 跨 workload 的组件重要性稳定性
experiments/rq2_partition_ablation   # 阈值/组件划分方法消融
experiments/rq3_release_frequency    # 低重要性组件释放频率消融
experiments/rq4_packing_ablation     # balanced packing 消融
experiments/rq5_adtopk_e2e           # ADTopk 端到端评估
```

示例命令：

```bash
# RQ1: profile 不同 workload 下的组件重要性
GPUS=4 LOCAL_FILES_ONLY=0 bash experiments/rq1_workload_stability/run_rq1_workload_stability.sh

# RQ4: 对比不同 payload packing 策略
GPUS=2 LOCAL_FILES_ONLY=0 bash experiments/rq4_packing_ablation/run_rq4_packing_ablation.sh

# RQ5: ADTopk 端到端评估
LOCAL_FILES_ONLY=0 SEEDS=42 bash experiments/rq5_adtopk_e2e/run_rq5_adtopk_seed42_windowavg.sh
```

更多说明见 [`docs/reproduce.md`](docs/reproduce.md)。

## 集成说明

ImportanceGradient 当前提供的是 PyTorch distributed 原型实现。运行时在 Python 通信层重新组织被选中的梯度张量，然后调用 `torch.distributed.all_reduce` 执行真实通信；它不需要修改 NCCL 内部实现。

大规模实验需要准备 CUDA 兼容的 PyTorch、Hugging Face 模型/数据集访问权限，以及多 GPU 分布式训练环境。

