import torch

from importance_gradient.release_policy import RQ3ReleasePolicy


def test_release_policy_accumulates_and_releases_residual():
    policy = RQ3ReleasePolicy(
        important_components={"0:Q"},
        low_importance_components={"0:K"},
        release_period=2,
        use_residual_compensation=True,
    )

    should_comm, update = policy.process_component_update("0:K", torch.tensor([1.0]), global_step=0)
    assert should_comm is False
    assert update is None

    should_comm, update = policy.process_component_update("0:K", torch.tensor([2.0]), global_step=1)
    assert should_comm is True
    assert torch.equal(update, torch.tensor([3.0]))
