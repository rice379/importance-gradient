# Paper-to-Code Mapping

| Paper Component | Code |
| --- | --- |
| Offline component importance profiling | `experiments/rq1_workload_stability/rq1_profile_workload_importance.py` |
| Component importance comparison | `experiments/rq1_workload_stability/rq1_analyze_workload_stability.py` |
| Importance threshold and partition | `experiments/rq2_partition_ablation/rq2_partition_ablation.py` |
| Online periodic synchronization | `importance_gradient/periodic_sync_gate.py` |
| Residual compensation | `importance_gradient/periodic_sync_gate.py`, `importance_gradient/release_policy.py` |
| Balanced bucket packing | `importance_gradient/bucket_runtime_planner.py` |
| Real distributed all-reduce path | `importance_gradient/real_bucket_comm.py` |
| RQ4 packing ablation | `experiments/rq4_packing_ablation` |
| RQ5 end-to-end ADTopk evaluation | `experiments/rq5_adtopk_e2e` |
