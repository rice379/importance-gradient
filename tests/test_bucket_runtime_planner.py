from importance_gradient.bucket_runtime_planner import BlockMeta, RiskAwareBucketPlanner


def test_bucket_planner_assigns_all_blocks():
    blocks = [
        BlockMeta(i, f"p{i}", ("g", i), 0, 10, 10, est_cost_mean=float(10 - i), est_cost_std=0.0, nnz_proxy=10)
        for i in range(6)
    ]
    planner = RiskAwareBucketPlanner(bucket_num=3, mode_requested="risk_aware")

    plan = planner.plan(blocks, step=1)

    assert set(plan.assignment) == {b.block_id for b in blocks}
    assert len(plan.bucket_states) == 3
    ratio, cv, overflow = planner.compute_plan_metrics(plan)
    assert ratio >= 1.0
    assert cv >= 0.0
    assert overflow >= 0
