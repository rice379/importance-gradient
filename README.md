# ImportanceGradient

An importance-aware sparse-gradient synchronization engine for large language model post-training.

ImportanceGradient is motivated by an empirical observation on post-training dynamics: after a conventional sparse-gradient selector such as Top-k or ADTopk has already removed most dense gradients, a large fraction of the remaining sparse gradients still contributes little to final model quality. These low-importance gradients do not need to be synchronized at every training iteration. Based on this observation, ImportanceGradient designs an **importance-aware synchronization system** that reduces sparse-gradient communication while preserving model accuracy.

## Core idea

ImportanceGradient operates in two phases:

### 1. Offline profiling phase

In the offline phase, ImportanceGradient profiles retained sparse gradients and builds component-level synchronization knowledge, including:

- component importance scores
- an adaptive importance threshold
- important and low-importance component sets

The profiling workflow maps retained sparse gradients to Transformer components and combines magnitude and cross-iteration consistency to identify components that should remain on the every-iteration synchronization path.

### 2. Online synchronization phase

In the online phase, ImportanceGradient applies an importance-aware synchronization policy:

- important component gradients are synchronized every iteration
- low-importance component gradients are delayed and accumulated in residual buffers
- delayed residuals are released periodically
- released payloads are balanced across communication buckets before collective synchronization

This allows ImportanceGradient to reduce the synchronized payload without dropping deferred gradient updates.

## What this repository contains

This repository contains the core implementation and experimental workflow for ImportanceGradient, including:

- component mapping and sparse-gradient importance profiling
- residual-preserving periodic synchronization for low-importance components
- balanced bucket planning and real `torch.distributed.all_reduce` communication
- ADTopk-based end-to-end training workflow
- RQ1-RQ5 experiment scripts for profiling, threshold ablation, release-frequency sweep, packing ablation, and end-to-end evaluation
- lightweight tests for core CPU-side logic

## Key features

- importance-aware sparse-gradient synchronization for LLM post-training
- offline component-importance profiling
- adaptive important/low-importance component partitioning
- residual compensation for delayed gradients
- balanced packing for released synchronization payloads
- compatibility with PyTorch distributed and DeepSpeed-style training workflows
- reproducibility scripts for paper experiments

## Results

Across representative LLM post-training workloads, ImportanceGradient is designed to:

- transmit roughly **one quarter** of the baseline sparse-gradient traffic
- reduce synchronization and all-reduce time
- preserve validation loss and task accuracy
- improve end-to-end training throughput

The repository provides the experimental scripts used to reproduce the paper's RQ studies.

## Install and test

```bash
git clone https://github.com/rice379/importance-gradient.git
cd importance-gradient

pip install -e .
pip install -r requirements.txt

# Lightweight tests, no distributed GPU environment required
pytest tests
```

## Experiment workflow

The main experiment directories are:

```text
experiments/rq1_workload_stability   # cross-workload importance stability
experiments/rq2_partition_ablation   # threshold/partition ablation
experiments/rq3_release_frequency    # low-importance release-frequency sweep
experiments/rq4_packing_ablation     # balanced packing ablation
experiments/rq5_adtopk_e2e           # ADTopk end-to-end evaluation
```

Example:

```bash
# RQ1: profile component importance across workloads
GPUS=4 LOCAL_FILES_ONLY=0 bash experiments/rq1_workload_stability/run_rq1_workload_stability.sh

# RQ4: compare payload packing policies
GPUS=2 LOCAL_FILES_ONLY=0 bash experiments/rq4_packing_ablation/run_rq4_packing_ablation.sh

# RQ5: end-to-end ADTopk evaluation
LOCAL_FILES_ONLY=0 SEEDS=42 bash experiments/rq5_adtopk_e2e/run_rq5_adtopk_seed42_windowavg.sh
```

See [`docs/reproduce.md`](docs/reproduce.md) for more details.

## Integration notes

ImportanceGradient currently provides a PyTorch distributed prototype implementation. The runtime path reorganizes selected gradient tensors at the Python communication layer and then invokes `torch.distributed.all_reduce`. It does not require modifying NCCL internals.

For large-scale experiments, prepare CUDA-compatible PyTorch, Hugging Face model/dataset access, and a multi-GPU distributed training environment.
