# Reproduction Notes

This document explains how the experiment scripts map to the paper.

## Environment

Recommended:

- Python 3.9+
- PyTorch with CUDA
- Hugging Face `transformers` and `datasets`
- Multi-GPU machine for distributed experiments

Install:

```bash
pip install -e .
pip install -r requirements.txt
```

## RQ1: Workload Stability

```bash
GPUS=4 LOCAL_FILES_ONLY=0 bash experiments/rq1_workload_stability/run_rq1_workload_stability.sh
```

Outputs are written under `runs/rq1_workload_stability` by default.

## RQ2: Partition Ablation

RQ2 expects threshold/profile outputs from the offline profiling stage.

```bash
bash experiments/rq2_partition_ablation/run_rq2_partition_ablation.sh
```

Important variables:

- `THRESHOLD_OUTPUT_DIR`: directory containing selection/profile files.
- `DATASET_PATH`: local dataset path if using a preprocessed dataset.

## RQ3: Release Frequency

```bash
bash experiments/rq3_release_frequency/run_rq3_frequency_sweep.sh
```

This sweeps `R_low` and compares residual compensation versus dropping delayed updates.

## RQ4: Packing Ablation

```bash
GPUS=2 LOCAL_FILES_ONLY=0 bash experiments/rq4_packing_ablation/run_rq4_packing_ablation.sh
```

Compared policies:

- `direct_filtering`
- `sequential`
- `greedy`
- `risk_aware`

## RQ5: ADTopk End-to-End

```bash
LOCAL_FILES_ONLY=0 SEEDS=42 bash experiments/rq5_adtopk_e2e/run_rq5_adtopk_seed42_windowavg.sh
```

Compared methods:

- `adtopk_sync`: synchronize all ADTopk-retained gradients every iteration.
- `iad_adtopk`: importance-aware delay without balanced packing.
- `ig_adtopk`: full ImportanceGradient with balanced packing.

## Data and Model Access

The scripts default to public Hugging Face names such as `facebook/opt-1.3b` and `uiyunkim-hub/pubmed-abstract`. If your machine is offline, prepare local Hugging Face caches and use `LOCAL_FILES_ONLY=1`.
