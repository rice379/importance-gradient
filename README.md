# ImportanceGradient

ImportanceGradient is an importance-aware sparse-gradient synchronization prototype for distributed LLM post-training.

The system follows an offline-online workflow:

1. Offline profiling builds component-level importance knowledge from retained sparse gradients.
2. Online training synchronizes important components every iteration.
3. Low-importance components are delayed with residual accumulation and released periodically.
4. Released payloads are materialized through balanced bucket packing before collective communication.

This repository contains the core runtime modules and the experiment scripts used for the paper's RQ studies.

## Repository Layout

```text
importance_gradient/              Core runtime modules
experiments/rq1_workload_stability Cross-workload importance stability
experiments/rq2_partition_ablation Threshold/partition ablation
experiments/rq3_release_frequency  Low-importance release frequency sweep
experiments/rq4_packing_ablation   Balanced packing ablation
experiments/rq5_adtopk_e2e         ADTopk end-to-end experiments
configs/                           Example DeepSpeed/runtime configs
docs/                              Reproduction and upload notes
tests/                             CPU-friendly sanity tests
```

## Install

```bash
git clone https://github.com/rice379/importance-gradient.git
cd importance-gradient

python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

For distributed GPU experiments, install CUDA-compatible PyTorch and configure `torchrun`, NCCL, and your local model/data cache.

## Quick Sanity Test

```bash
pytest tests
```

These tests only check lightweight CPU-side logic. Full experiments require GPUs and Hugging Face model/dataset access.

## Core Modules

- `importance_gradient.periodic_sync_gate`: component-aware synchronization gate with residual accumulation.
- `importance_gradient.bucket_runtime_planner`: balanced/risk-aware bucket planning.
- `importance_gradient.real_bucket_comm`: real bucketized `torch.distributed.all_reduce` path.
- `importance_gradient.release_policy`: minimal release-policy logic for RQ3.
- `importance_gradient.component_mapper`: parameter-name to Transformer-component mapping.

## Experiment Mapping

| Paper Question | Directory | Purpose |
| --- | --- | --- |
| RQ1 | `experiments/rq1_workload_stability` | Profile component importance across PubMed, WikiText-103, and Alpaca. |
| RQ2 | `experiments/rq2_partition_ablation` | Compare Knee, Otsu, GMM, KMeans, fixed-ratio, and percentile partitioning. |
| RQ3 | `experiments/rq3_release_frequency` | Sweep low-importance release period and residual compensation. |
| RQ4 | `experiments/rq4_packing_ablation` | Compare direct filtering, sequential packing, greedy packing, and risk-aware packing. |
| RQ5 | `experiments/rq5_adtopk_e2e` | End-to-end ADTopk baseline, importance-aware delay, and full ImportanceGradient. |

## Example Commands

RQ1:

```bash
GPUS=4 LOCAL_FILES_ONLY=0 bash experiments/rq1_workload_stability/run_rq1_workload_stability.sh
```

RQ4:

```bash
GPUS=2 LOCAL_FILES_ONLY=0 bash experiments/rq4_packing_ablation/run_rq4_packing_ablation.sh
```

RQ5 quick run:

```bash
LOCAL_FILES_ONLY=0 SEEDS=42 bash experiments/rq5_adtopk_e2e/run_rq5_adtopk_seed42_windowavg.sh
```

See `docs/reproduce.md` for more details.

## Notes

- This code does not include model weights, datasets, checkpoints, or large raw logs.
- The scripts assume public Hugging Face models/datasets by default.
- Some experiments are expensive and were designed for multi-GPU A100-class machines.
