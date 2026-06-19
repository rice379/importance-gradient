# ImportanceGradient

ImportanceGradient 是一个面向大语言模型后训练的 importance-aware 稀疏梯度同步原型系统。

系统采用离线-在线两阶段流程：

1. 离线 profiling：根据保留下来的 sparse gradients 统计组件级重要性。
2. 在线训练：重要组件每轮同步。
3. 低重要性组件延迟同步，并通过 residual accumulation 保存被延迟的更新。
4. 释放出来的同步 payload 通过 balanced bucket packing 后再执行 collective communication。

## 目录说明

```text
importance_gradient/              核心运行时代码
experiments/rq1_workload_stability RQ1: workload 间重要性稳定性
experiments/rq2_partition_ablation RQ2: 阈值/划分方法消融
experiments/rq3_release_frequency  RQ3: 低重要性组件释放频率和 residual
experiments/rq4_packing_ablation   RQ4: balanced packing 消融
experiments/rq5_adtopk_e2e         RQ5: ADTopk 端到端实验
configs/                           配置文件
docs/                              复现和上传说明
tests/                             轻量测试
```

## 安装

```bash
git clone https://github.com/rice379/importance-gradient.git
cd importance-gradient
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Windows PowerShell 激活虚拟环境可用：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 快速检查

```bash
pytest tests
```

完整实验需要 GPU、PyTorch、Hugging Face 模型/数据集，以及可用的 `torchrun`/NCCL 环境。
